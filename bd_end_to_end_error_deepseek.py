import torch
import torch.nn as nn
import argparse
import os
import numpy as np

from lib.data import get_loaders
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepseek_model import replace_deepseek

DEVICE = torch.device("cuda")
DTYPE = torch.float16


def get_llm(model_name, seqlen=2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=DEVICE, trust_remote_code=True, dtype=DTYPE
    )
    model.seqlen = seqlen
    return model



@torch.no_grad()
def compute_batch_ppl_from_logits(logits, input_ids):
    # logits: [B, T, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:]
    loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)),
                                 shift_labels.reshape(-1))

    return loss.float() * input_ids.size(1) * input_ids.size(0)


@torch.no_grad()
def collect_original_pass(model, tokenizer, dataset_name, batch_size):

    model.eval()
    tokenizer = tokenizer
    print(f"[Stage-1] evaluating on {dataset_name}")

    _, testenc = get_loaders(dataset_name, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    print(f"[Stage-1] nsamples {nsamples}")


    nlls = []
    model.to(DEVICE)
    with torch.no_grad():
        for i in range(0, nsamples, batch_size):
            if i + batch_size > nsamples:
                break
            if i % 50 == 0:
                print(f"[Stage-1] sample {i}")

            j = i + batch_size
            inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(DEVICE)
            inputs = inputs.reshape(j - i, model.seqlen)

            
            logits = model(inputs, use_cache=False).logits
            nlls.append(compute_batch_ppl_from_logits(logits, inputs))


    # model.to("cpu")
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"[Stage-1] original ppl: {ppl.item():.6f}")
    return ppl.item(), nsamples, testenc


@torch.no_grad()
def compare_bd_pass(bd_model, tokenizer, dataset_name, batch_size, nsamples, testenc):
    bd_model.eval()
    print(f"[Stage-2] evaluating on {dataset_name} for BD model")

    bd_nlls = []

    bd_model.to(DEVICE)
    with torch.no_grad():
        for i in range(0, nsamples, batch_size):
            if i + batch_size > nsamples:
                break
            if i % 50 == 0:
                print(f"[Stage-2] sample {i}")

            j = i + batch_size
            inputs = testenc[:, (i * bd_model.seqlen):(j * bd_model.seqlen)].to(DEVICE)
            inputs = inputs.reshape(j - i, bd_model.seqlen)

            
            bd_logits = bd_model(inputs, use_cache=False).logits
            bd_nlls.append(compute_batch_ppl_from_logits(bd_logits, inputs))

    bd_ppl = torch.exp(torch.stack(bd_nlls).sum() / (nsamples * bd_model.seqlen))
    print(f"[Stage-2] bd ppl: {bd_ppl.item():.6f}")


    return bd_ppl.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='deepseek model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="wikitext2")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--mode', type=str, default=None, help='qk|vo|qkvo')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"current dtype: {DTYPE}")

    with torch.no_grad():
        model = get_llm(args.model, seqlen=args.seqlen)

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        orig_ppl, nsamples, testenc = collect_original_pass(
            model, tokenizer, args.dataset, args.batch_size
        )

        replace_deepseek(model, DEVICE, args.mode)


        bd_ppl = compare_bd_pass(
            model, tokenizer, args.dataset, args.batch_size, nsamples, testenc
        )

        print(f"[Summary] original ppl: {orig_ppl:.6f} | bd ppl: {bd_ppl:.6f}")


if __name__ == '__main__':
    main()
