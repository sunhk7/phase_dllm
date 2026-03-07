import torch
import numpy as np
import torch.nn.functional as F
import argparse
import json
import os

from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from generate import generate


def main():
    parser = argparse.ArgumentParser(description="Run LLaDA generation for manual prompts and collect attention dynamics")
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = LLaDAModelLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    assert tokenizer.pad_token_id != 126336

    os.makedirs(args.results_dir, exist_ok=True)
    dataset_results_dir = os.path.join(args.results_dir, "prompt")
    os.makedirs(dataset_results_dir, exist_ok=True)

    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
    ]

    records = []
    
    # Process prompts in batches
    for start in range(0, len(prompts), args.batch_size):
        end = min(start + args.batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        
        messages = [{"role": "user", "content": p} for p in batch_prompts]
        formatted_prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

        encoded_outputs = tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded_outputs['input_ids'].to(device)
        attention_mask = encoded_outputs['attention_mask'].to(device)

        dynamics_path = os.path.join(dataset_results_dir, f"prompt_dynamics_{start:05d}_{end - 1:05d}.npy")
        
        out = generate(
            model, 
            input_ids, 
            attention_mask, 
            steps=args.steps, 
            gen_length=args.gen_length, 
            block_length=args.block_length, 
            temperature=args.temperature, 
            cfg_scale=args.cfg_scale, 
            remasking=args.remasking,
            save_dynamics_path=dynamics_path
        )
        output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
        
        for local_idx, (prompt, prediction) in enumerate(zip(batch_prompts, output_text)):
            sample_idx = start + local_idx
            records.append(
                {
                    "index": sample_idx,
                    "prompt": prompt,
                    "prediction": prediction,
                    "dynamics_path": dynamics_path,
                }
            )
            print(f"[{sample_idx}] {prediction}")
            print("-" * 50)
            
    output_jsonl = os.path.join(dataset_results_dir, "prompt_outputs.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} samples to {output_jsonl}")


if __name__ == '__main__':
    main()
