import torch
import numpy as np
import torch.nn.functional as F
import argparse
import json
import os
import random

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from model.modeling_llada import LLaDAModelLM
from generate import generate

def main():
    parser = argparse.ArgumentParser(description="Run LLaDA generation for LongBench and collect attention dynamics")
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--datasets", type=str, default="narrativeqa,hotpotqa,qasper,gov_report")
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=512)
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
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    assert tokenizer.pad_token_id != 126336

    os.makedirs(args.results_dir, exist_ok=True)
    longbench_results_dir = os.path.join(args.results_dir, "longbench")
    os.makedirs(longbench_results_dir, exist_ok=True)

    target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    records = []

    for dataset_name in target_datasets:
        print(f"Loading LongBench dataset: {dataset_name}")
        try:
            dataset = load_dataset("THUDM/LongBench", dataset_name, split="test")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue

        total_samples = len(dataset)
        sample_indices = random.sample(range(total_samples), min(args.samples_per_dataset, total_samples))

        for local_idx in sample_indices:
            row = dataset[local_idx]
            context = row.get("context", "")
            question = row.get("input", "")
            answers = row.get("answers", [])

            messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nPlease answer the question based on the context."}]
            prompt = tokenizer.apply_chat_template([messages[0]], add_generation_prompt=True, tokenize=False)

            encoded_outputs = tokenizer(
                [prompt],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded_outputs["input_ids"].to(device)
            attention_mask = encoded_outputs["attention_mask"].to(device)

            dynamics_path = os.path.join(longbench_results_dir, f"{dataset_name}_dynamics_{local_idx}.npy")

            print(f"Generating for {dataset_name} index {local_idx}...")
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
            
            output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            record = {
                "dataset": dataset_name,
                "index": local_idx,
                "question": question,
                "answers": answers,
                "prediction": output_text,
                "dynamics_path": dynamics_path,
            }
            records.append(record)

            print(f"Prediction: {output_text}")
            print("-" * 50)

    output_jsonl = os.path.join(longbench_results_dir, "longbench_outputs.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} samples to {output_jsonl}")

if __name__ == '__main__':
    main()
