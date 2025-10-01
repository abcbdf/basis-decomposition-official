import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from collections import defaultdict



# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV2Config"

def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@torch.no_grad()
def replace_deepseek(model, device, mode):
    named_modules = list(model.named_modules())
    for name, module in tqdm(named_modules):
        if module.__class__.__name__ == "DeepseekV2Attention":
            print(f"DeepseekV2Attention: {name}")
            module.to(device)
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            child_name = name.split(".")[-1]
            new_module = DeepseekV2AttentionSimple.copy_from_original(module, mode)
            module.cpu()
            new_module.to(device)
            setattr(parent, child_name, new_module)
            del module

def relax_factorization(weight, rank, basis_top, sparse="right"):  # weight m * n
    weight_float = weight.float()
    if sparse == "right":
        pass
    elif sparse == "left":
        weight_float = weight_float.T
    if basis_top:
        left = weight_float[:, :rank]   # m * r
    else:
        left = weight_float[:, -rank:]   # m * r
    result = torch.linalg.lstsq(left, weight_float)   # left: m * r, weight: m * n， result：r * n
    residual = torch.mean(result.residuals).item()
    right = result.solution
    left, right = left.to(weight.dtype), right.to(weight.dtype)
    if basis_top:
        right = right[:, rank:]  # r * (n - r)
    else:
        right = right[:, :-rank]  # r * (n - r)
    if sparse == "right":
        return residual, left, right
    elif sparse == "left":
        return residual, right.T, left.T


class DeepseekV2AttentionSimple(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, q_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj, rotary_emb, layer_idx, mode):
        super().__init__()
        assert config.q_lora_rank is None
        self.start_events = defaultdict(list)
        self.end_events = defaultdict(list)
        self.record_timing = False
        self.mode = mode
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size  # d
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank  # dr
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim  # dkv/h
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        q_proj_weight = q_proj.weight.view(self.num_heads, self.q_head_dim, self.hidden_size)


        self.q_pe_weight = nn.Parameter(torch.reshape(q_proj_weight[:, self.qk_nope_head_dim:, :], [-1, self.hidden_size]))

        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.kv_a_layernorm = kv_a_layernorm

        kv_b_proj_weight = kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, config.kv_lora_rank)

        if "qk" in mode:
            q_nope_weight = q_proj_weight[:, :self.qk_nope_head_dim, :]  # h * (dkv/h) * d
            k_b_proj_weight = kv_b_proj_weight[:, :self.qk_nope_head_dim, :]  # h * (dkv/h) * dr
            qk = torch.matmul(q_nope_weight.transpose(1, 2), k_b_proj_weight)  # h * d * dr
            if "no_select" in mode:
                k_basis_top = True
                print(f"set k_basis_top to True")
            else:
                mean_residual_list = []
                for cur_basis_top in [True, False]:
                    residual_list = []
                    for i in range(self.num_heads):
                        residual, _, _ = relax_factorization(qk[i, :, :], self.qk_nope_head_dim, cur_basis_top)
                        residual_list.append(residual)
                    mean_residual_list.append(np.mean(residual_list))
                if mean_residual_list[0] <= mean_residual_list[1]:
                    k_basis_top = True
                else:
                    k_basis_top = False
                print(f"k_basis_top: {k_basis_top}, mean_residual_list: {mean_residual_list}")

            q_list = []
            k_list = []
            for i in range(self.num_heads):
                print(f"qk head {i}")
                residual, left, right = relax_factorization(qk[i, :, :], self.qk_nope_head_dim,
                                                            k_basis_top)  # 1, d * (dkv/h), (dkv/h) * (dr - dkv/h)
                print(f"residual: {residual}")
                q_list.append(left)
                k_list.append(right)

            q_w = torch.cat(q_list, dim=1)  # d * (dkv/h* h)
            k_w = torch.cat(k_list, dim=0).transpose(0, 1)  # (dr-dkv/h) * (h * dkv/h)

            self.q_w = nn.Parameter(q_w.contiguous())  # d * (dkv/h* h)
            self.k_w = nn.Parameter(k_w.contiguous())  # (dr-dkv/h) * (h * dkv/h)
            self.k_basis_top = k_basis_top
        else:
            self.q_nope_weight = nn.Parameter(torch.reshape(q_proj_weight[:, :self.qk_nope_head_dim, :], [-1, self.hidden_size]))  # dkv * d
            self.k_b_proj_weight = nn.Parameter(torch.reshape(kv_b_proj_weight[:, :self.qk_nope_head_dim, :], [-1, config.kv_lora_rank]))  # dkv * dr

        if "vo" in mode:
            v_w_reshape = kv_b_proj_weight[:, self.qk_nope_head_dim:, :]  # h * (dkv/h) * dr
            o_w_reshape = o_proj.weight.view(self.hidden_size, self.num_heads, self.v_head_dim)  # d * h * (dkv/h)
            vo = torch.matmul(v_w_reshape.transpose(1, 2),
                              o_w_reshape.permute(1, 2, 0))  # h*dr*(dkv/h) @ h*(dkv/h)*d = h*dr*d
            v_list = []
            o_list = []
            if "no_select" in mode:
                v_basis_top = True
                print(f"set v_basis_top to True")
            else:
                mean_residual_list = []
                for cur_basis_top in [True, False]:
                    residual_list = []
                    for i in range(self.num_heads):
                        residual, _, _ = relax_factorization(vo[i, :, :], self.v_head_dim, cur_basis_top, sparse="left")
                        residual_list.append(residual)
                    mean_residual_list.append(np.mean(residual_list))
                if mean_residual_list[0] <= mean_residual_list[1]:
                    v_basis_top = True
                else:
                    v_basis_top = False
                print(f"v_basis_top: {v_basis_top}, mean_residual_list: {mean_residual_list}")
            for i in range(self.num_heads):
                print(f"vo head {i}")
                residual, left, right = relax_factorization(vo[i, :, :], self.v_head_dim, v_basis_top,
                                                            sparse="left")  # 1, (dr-dkv/h)*(dkv/h), (dkv/h)*d
                print(f"residual: {residual}")
                o_list.append(right)
                v_list.append(left)

            o_w = torch.cat(o_list, dim=0)  # dkv*d
            v_w = torch.cat(v_list, dim=1)  # (dr-dkv/h)*dkv

            self.v_w = nn.Parameter(v_w.contiguous())  # (dr-dkv/h)*dkv
            self.o_w = nn.Parameter(o_w.contiguous())  # dkv*d
            self.v_basis_top = v_basis_top
        else:
            self.v_b_proj_weight = nn.Parameter(torch.reshape(kv_b_proj_weight[:, self.qk_nope_head_dim:, :], [-1, config.kv_lora_rank]))  # dkv * dr
            self.o_proj = o_proj  # d * dkv


        self.rotary_emb = rotary_emb

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    @staticmethod
    def copy_from_original(layer, mode):
        return DeepseekV2AttentionSimple(
            config = layer.config,
            q_proj = layer.q_proj,
            kv_a_proj_with_mqa = layer.kv_a_proj_with_mqa,
            kv_a_layernorm = layer.kv_a_layernorm,
            kv_b_proj = layer.kv_b_proj,
            o_proj = layer.o_proj,
            rotary_emb = layer.rotary_emb,
            layer_idx = layer.layer_idx,
            mode = mode
        )


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def get_key(self, key, batch_size):
        if self.k_basis_top:
            key_basis_channels = torch.unsqueeze(key[:, :, :self.qk_nope_head_dim], 2)  # b * s * 1 * dkv/h
            key_nonbasis_channels = key[:, :, self.qk_nope_head_dim:]  # b * s * (d - dkv/h)
        else:
            key_basis_channels = torch.unsqueeze(key[:, :, -self.qk_nope_head_dim:], 2)  # b * s * 1 * dkv/h
            key_nonbasis_channels = key[:, :, :-self.qk_nope_head_dim]  # b * s * (d - dkv/h)
        key_nonbasis_channels = torch.matmul(key_nonbasis_channels, self.k_w).view(batch_size, -1, self.num_heads, self.qk_nope_head_dim)  # (b * s * (d - dkv/h)) @ ((d-dkv/h) * dkv) = b * s * dkv
        key = key_basis_channels + key_nonbasis_channels
        key = key.transpose(1, 2)  # b * h * s * (dkv/h)
        return key

    def get_value(self, value, batch_size):
        if self.v_basis_top:
            value_basis_channels = torch.unsqueeze(value[:, :, :self.v_head_dim], 2)  # b * s * 1 * (dkv/h)
            value_nonbasis_channels = value[:, :, self.v_head_dim:]
        else:
            value_basis_channels = torch.unsqueeze(value[:, :, -self.v_head_dim:], 2)  # b * s * 1 * (dkv/h)
            value_nonbasis_channels = value[:, :, :-self.v_head_dim]
        value_nonbasis_channels = torch.matmul(value_nonbasis_channels, self.v_w).view(batch_size, -1, self.num_heads, self.v_head_dim)  # (b * s * (d - dkv/h)) @ ((d-dkv/h) * dkv) = b * s * dkv
        value = value_basis_channels + value_nonbasis_channels
        value = value.transpose(1, 2)  # b * h * s * (dkv/h)
        return value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self.record_timing:  # Only record after warm-up
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            k_start_event = torch.cuda.Event(enable_timing=True)
            k_end_event = torch.cuda.Event(enable_timing=True)
            v_start_event = torch.cuda.Event(enable_timing=True)
            v_end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        # print(f"hidden states device: {hidden_states.device}")

        q_pe = torch.matmul(hidden_states, self.q_pe_weight.T).view(bsz, q_len, self.num_heads, -1).transpose(1, 2)


        # q = self.q_proj(hidden_states)
        # q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # q_nope, q_pe = torch.split(
        #     q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        # )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_a = self.kv_a_layernorm(compressed_kv)

        if "qk" in self.mode:
            q_nope = torch.matmul(hidden_states, self.q_w).view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
            if self.record_timing:
                k_start_event.record()
            k_nope = self.get_key(kv_a, bsz)
        else:
            q_nope = torch.matmul(hidden_states, self.q_nope_weight.T).view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
            if self.record_timing:
                k_start_event.record()
            k_nope = torch.matmul(kv_a, self.k_b_proj_weight.T).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        if self.record_timing:
            k_end_event.record()
            v_start_event.record()

        if "vo" in self.mode:
            value_states = self.get_value(kv_a, bsz)
        else:
            value_states = torch.matmul(kv_a, self.v_b_proj_weight.T).view(bsz, q_len, self.num_heads, self.v_head_dim).transpose(1, 2)

        if self.record_timing:
            v_end_event.record()

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        if "vo" in self.mode:
            attn_output = torch.matmul(attn_output, self.o_w)
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if self.record_timing:
            end_event.record()
            self.start_events["all"].append(start_event)
            self.end_events["all"].append(end_event)
            self.start_events["k"].append(k_start_event)
            self.end_events["k"].append(k_end_event)
            self.start_events["v"].append(v_start_event)
            self.end_events["v"].append(v_end_event)

        return attn_output, attn_weights, past_key_value