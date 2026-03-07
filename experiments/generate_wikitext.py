import torch
import numpy as np
import argparse
import json
import os
import random

from transformers import AutoTokenizer
from datasets import load_dataset
from model.modeling_llada import LLaDAModelLM
from generate import generate

def main():
    parser = argparse.ArgumentParser(description="Run LLaDA generation for WikiText-103 and collect attention dynamics")
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=768)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--local-half-window", type=int, default=64, help="Local window size for calculating global ratio.")
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
        device_map="auto" if device == "cuda" else None,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    assert tokenizer.pad_token_id != 126336

    os.makedirs(args.results_dir, exist_ok=True)
    wikitext_results_dir = os.path.join(args.results_dir, "wikitext")
    os.makedirs(wikitext_results_dir, exist_ok=True)

    print("Loading WikiText-103 dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    except Exception as e:
        print(f"Error loading wikitext dataset: {e}")
        return

    # Filter out empty or very short lines to speed up logic
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 500)
    
    valid_samples = []
    # Find passages that have at least prompt_length tokens
    for row in dataset:
        text = row.get("text", "").strip()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= args.prompt_length:
            valid_samples.append(tokens)
            if len(valid_samples) >= args.samples * 3:
                break
                
    if len(valid_samples) == 0:
        print("Error: Could not find any text snippets long enough in the dataset.")
        return

    sample_indices = random.sample(range(len(valid_samples)), min(args.samples, len(valid_samples)))

    records = []
    for local_idx in sample_indices:
        tokens = valid_samples[local_idx]
        prompt_tokens = tokens[:args.prompt_length]
        prompt_text = tokenizer.decode(prompt_tokens)

        # Standard text completion prompt
        messages = [{"role": "user", "content": f"Please continue the following text:\n\n{prompt_text}"}]
        prompt = tokenizer.apply_chat_template([messages[0]], add_generation_prompt=True, tokenize=False)

        encoded_outputs = tokenizer(
            [prompt],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        
        first_device = model.device
        input_ids = encoded_outputs["input_ids"].to(first_device)
        attention_mask = encoded_outputs["attention_mask"].to(first_device)

        dynamics_path = os.path.join(wikitext_results_dir, f"wikitext_dynamics_{local_idx}.npy")

        print(f"Generating for wikitext index {local_idx}...")
        
        with torch.no_grad():
            out = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                save_dynamics_path=dynamics_path,
                local_half_window=args.local_half_window,
            )
            
        out = out.cpu()
        torch.cuda.empty_cache()
        output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        record = {
            "dataset": "wikitext-103",
            "index": local_idx,
            "prompt": prompt_text,
            "prediction": output_text,
            "dynamics_path": dynamics_path,
        }
        records.append(record)

        print(f"Prediction: {output_text[:150]}...")
        print("-" * 50)

    output_jsonl = os.path.join(wikitext_results_dir, "wikitext_outputs.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} samples to {output_jsonl}")

if __name__ == '__main__':
    main()
