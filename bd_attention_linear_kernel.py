import torch

import triton
import triton.language as tl
import numpy as np



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device('cuda')
DTYPE = torch.float16

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=8),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'DIM', 'OUT_DIM', 'HEAD_DIM'],
)
@triton.jit
def bd_attention_matmul_kernel(
        # Pointers to matrices
        x_ptr, w_ptr, c_ptr,
        # Matrix dimensions
        M, DIM, DIM_C, OUT_DIM,
        HEAD_DIM: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(OUT_DIM, HEAD_DIM)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    accumulator = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)

    # accumulator += I*X[:, 1:d_h]
    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k1 = tl.arange(0, HEAD_DIM)
    x_ptrs1 = x_ptr + (offs_xm[:, None] * DIM + offs_k1[None, :])
    x1 = tl.load(x_ptrs1)
    accumulator += x1


    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_wn = (pid_n * HEAD_DIM + tl.arange(0, HEAD_DIM)) % OUT_DIM
    x_ptrs = x_ptr + (offs_xm[:, None] * DIM + offs_k[None, :]) + HEAD_DIM
    w_ptrs = w_ptr + (offs_k[:, None] * OUT_DIM + offs_wn[None, :])
    for k in range(0, tl.cdiv(DIM_C, BLOCK_SIZE_K)):

        # accumulator += C*X[:, d_h:d]
        x = tl.load(x_ptrs, mask=offs_k[None, :] < DIM_C - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < DIM_C - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(x, w, accumulator)
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K * OUT_DIM

    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * HEAD_DIM + tl.arange(0, HEAD_DIM)
    c_ptrs = c_ptr + OUT_DIM * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < OUT_DIM)
    tl.store(c_ptrs, c, mask=c_mask)


def BDA(x, w, HEAD_DIM):
    assert x.is_contiguous(), "Matrix x must be contiguous"
    assert w.is_contiguous(), "Matrix w must be contiguous"

    M, DIM = x.shape  # (b * s) * d
    DIM_C, OUT_DIM = w.shape  # (d-dkv/h) * (h * dkv/h)

    # Allocates output.
    c = torch.empty((M, OUT_DIM), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(OUT_DIM, HEAD_DIM), )
    bd_attention_matmul_kernel[grid](
        x, w, c,  #
        M, DIM, DIM_C, OUT_DIM, HEAD_DIM
    )
    return c   # (b*s) * dkv

def BDA_torch(x, k_w, BATCH_SIZE, HEAD_COUNT, HEAD_DIM):
    key_pivot_channels = torch.unsqueeze(x[:, :, :HEAD_DIM], 2)  # b * s * 1 * dkv/h
    key_nonpivot_channels = x[:, :, HEAD_DIM:]  # b * s * (d - dkv/h)
    key_nonpivot_channels = torch.matmul(key_nonpivot_channels, k_w).view(BATCH_SIZE, -1, HEAD_COUNT, HEAD_DIM)  # (b * s * (d - dkv/h)) @ ((d-dkv/h) * dkv) = b * s * dkv
    key = key_pivot_channels + key_nonpivot_channels
    key = key.transpose(1, 2)  # b * h * s * (dkv/h)

    return key



def validate_correctness():
    torch.manual_seed(0)
    HEAD_COUNT = 8
    MODEL_DIM = 512
    QK_DIM = 512
    QK_DIM_PER_HEAD = QK_DIM // HEAD_COUNT
    BATCH_SIZE = 16
    SEQ_LEN = 128
    DIM_C = MODEL_DIM - QK_DIM_PER_HEAD

    k_w = torch.empty([QK_DIM, DIM_C], dtype=DTYPE, device=DEVICE)
    torch.nn.init.xavier_uniform_(k_w)
    k_w = k_w.transpose(0, 1).contiguous()


    x = torch.randn([BATCH_SIZE, SEQ_LEN, MODEL_DIM], dtype=DTYPE, device=DEVICE)

    # torch result
    torch_output = BDA_torch(x, k_w, BATCH_SIZE, HEAD_COUNT, QK_DIM_PER_HEAD)

    # triton result
    x_view = x.view(BATCH_SIZE * SEQ_LEN, MODEL_DIM)
    k_w_view = k_w
    triton_output = BDA(x_view, k_w_view, QK_DIM_PER_HEAD)
    triton_output = triton_output.view(BATCH_SIZE, SEQ_LEN, HEAD_COUNT, QK_DIM_PER_HEAD).transpose(1, 2)

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


if __name__ == "__main__":

    validate_correctness()

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M"],
            x_vals=[2 ** i for i in range(6, 17)], 
            line_arg="provider", 

            line_vals=['MHA', 'BDA'], 
            line_names=['MHA', 'BDA'], 
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            xlabel="Sequence Length (M)",
            ylabel="Million Token/s", 
            plot_name="MHAvsBDA-fp16",
            x_log=True,
            args={}, 
        ))


    @triton.testing.perf_report(configs)
    def benchmark(M, provider):
        # M = 128 * 128
        HEAD_COUNT = 128
        MODEL_DIM = 512
        QK_DIM_PER_HEAD = 128
        QK_DIM = HEAD_COUNT * QK_DIM_PER_HEAD
        BATCH_SIZE = 1
        SEQ_LEN = M
        DIM_C = MODEL_DIM - QK_DIM_PER_HEAD

        WARMUP = 25
        REP = 100

        k_w = torch.randn([QK_DIM, DIM_C], dtype=DTYPE, device=DEVICE).transpose(0, 1).contiguous()

        x = torch.randn([BATCH_SIZE, SEQ_LEN, MODEL_DIM], dtype=DTYPE, device=DEVICE)
        w = torch.randn([QK_DIM, MODEL_DIM], dtype=DTYPE, device=DEVICE)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "MHA":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(x, w.T), quantiles=quantiles, warmup=WARMUP, rep=REP)
            print(f"MHA M {M}: ms {ms}")
        elif provider == 'BDA':
            x_view = x.view(BATCH_SIZE * SEQ_LEN, MODEL_DIM)
            k_w_view = k_w
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: BDA(x_view, k_w_view, QK_DIM_PER_HEAD), quantiles=quantiles, warmup=WARMUP, rep=REP)
            print(f"BDA M {M}: ms {ms}, {bd_attention_matmul_kernel.best_config}")

        perf = lambda ms: BATCH_SIZE * SEQ_LEN / (ms * 1e-3) / 1e6
        return perf(ms), perf(max_ms), perf(min_ms)


    result = benchmark.run(return_df=True, save_path="result_image/")[0]
    result['ratio'] = (result["BDA"] - result["MHA"]) / result["MHA"]
    print(result)
