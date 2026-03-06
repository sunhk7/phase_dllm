import argparse

import numpy as np
import torch
from transformers import AutoTokenizer

from generate import generate
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM
from plot_dynamics import plot_attention_dynamics


def build_dummy_model(device: str) -> LLaDAModelLM:
    config = LLaDAConfig(
        n_layers=4,
        d_model=256,
        n_heads=8,
        n_kv_heads=8,
        mlp_ratio=4,
        rope=True,
        alibi=False,
        max_sequence_length=512,
        vocab_size=130000,
        embedding_size=130048,
    )
    model = LLaDAModelLM(config=config, init_params=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def main():
    parser = argparse.ArgumentParser(description="Run dummy LLaDA dynamics test")
    parser.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--output", type=str, default="dummy_dynamics.npy")
    parser.add_argument("--plot", type=str, default="dummy_dynamics.png")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    dummy_model = build_dummy_model(device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    prompt = "Test attention locality."
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    _ = generate(
        dummy_model,
        input_ids,
        attention_mask=attention_mask,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        collect_attention_dynamics=True,
        save_dynamics_path=args.output,
    )

    dynamics = np.load(args.output)
    expected_shape = (args.steps, 4)
    if dynamics.shape != expected_shape:
        raise RuntimeError(f"Unexpected dynamics shape: {dynamics.shape}, expected {expected_shape}")

    plot_attention_dynamics(
        args.output,
        output_path=args.plot,
        title="Dummy LLaDA Spatio-Temporal Attention Dynamics",
    )

    print(f"Saved dynamics: {args.output} shape={dynamics.shape}")
    print(f"Saved heatmap: {args.plot}")


if __name__ == "__main__":
    main()
