import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from generate import generate
from model.modeling_llada import LLaDAModelLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaDA generation on gsm8k and save entropy-ratio pairs")
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=768)
    parser.add_argument("--block-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--logits-eos-inf", action="store_true", help="Set EOS logit to -inf")
    parser.add_argument("--confidence-eos-eot-inf", action="store_true", help="Set EOS/EoT confidence to -inf")
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

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == 126336:
        raise RuntimeError("pad_token_id equals mask_id(126336), current generate() path does not support this case.")

    os.makedirs(args.results_dir, exist_ok=True)
    gsm8k_results_dir = os.path.join(args.results_dir, "gsm8k")
    os.makedirs(gsm8k_results_dir, exist_ok=True)

    dataset = load_dataset("gsm8k", "main", split=args.split)
    total_samples = min(len(dataset), args.max_samples)
    records = []

    print(f"[INFO] dataset=gsm8k/main:{args.split}, total_samples={total_samples}")

    for start in range(0, total_samples, args.batch_size):
        end = min(start + args.batch_size, total_samples)
        batch = dataset[start:end]
        questions = batch["question"]
        answers = batch["answer"]

        messages = [{"role": "user", "content": f"Solve the math problem.\\n\\nQuestion: {q}"} for q in questions]
        prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

        encoded_outputs = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded_outputs["input_ids"].to(device)
        attention_mask = encoded_outputs["attention_mask"].to(device)

        pairs_path = os.path.join(gsm8k_results_dir, f"gsm8k_pairs_{start:05d}_{end - 1:05d}.npy")
        print(f"[RUN] gsm8k batch {start}-{end - 1}, pairs={pairs_path}")

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
            logits_eos_inf=args.logits_eos_inf,
            confidence_eos_eot_inf=args.confidence_eos_eot_inf,
            collect_attention_dynamics=False,
            save_dynamics_path=None,
            collect_entropy_ratio_pairs=True,
            save_entropy_ratio_path=pairs_path,
            entropy_layer_index=24,
        )
        output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)

        for local_idx, (question, answer, prediction) in enumerate(zip(questions, answers, output_text)):
            sample_idx = start + local_idx
            records.append(
                {
                    "dataset": "gsm8k",
                    "index": sample_idx,
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                    "pairs_path": pairs_path,
                }
            )
            print(f"[{sample_idx}] {prediction}")
            print("-" * 50)

    output_jsonl = os.path.join(gsm8k_results_dir, "gsm8k_outputs.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} samples to {output_jsonl}")


if __name__ == "__main__":
    main()
