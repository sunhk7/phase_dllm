import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from generate import generate
from model.modeling_llada import LLaDAModelLM


DEFAULT_PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
]


def load_prompts(prompts_file: str, prompt_key: str = "prompt") -> list[str]:
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    if prompts_file.endswith(".jsonl"):
        prompts = []
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if prompt_key in obj:
                    prompts.append(str(obj[prompt_key]).strip())
                elif "content" in obj:
                    prompts.append(str(obj["content"]).strip())
                else:
                    raise KeyError(f"Line {line_no} has no '{prompt_key}' or 'content' field")
        return [p for p in prompts if p]

    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaDA generation for prompts loaded from a local file")
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompts-file", type=str, default=None, help="Optional path to .txt or .jsonl prompts file")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Prompt field name when using .jsonl")
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

    if args.prompts_file:
        if os.path.isfile(args.prompts_file):
            prompts = load_prompts(args.prompts_file, args.prompt_key)
            print(f"[INFO] Loaded {len(prompts)} prompts from file: {args.prompts_file}")
        else:
            prompts = DEFAULT_PROMPTS
            print(f"[WARN] prompts file not found: {args.prompts_file}, fallback to DEFAULT_PROMPTS")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"[INFO] Using built-in DEFAULT_PROMPTS ({len(prompts)} prompts)")

    if len(prompts) == 0:
        raise RuntimeError("No valid prompts available.")

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
    prompt_results_dir = os.path.join(args.results_dir, "prompt")
    os.makedirs(prompt_results_dir, exist_ok=True)

    records = []
    total_samples = len(prompts)
    print(f"[INFO] total_prompts={total_samples}")

    for start in range(0, total_samples, args.batch_size):
        end = min(start + args.batch_size, total_samples)
        batch_prompts = prompts[start:end]

        messages = [{"role": "user", "content": p} for p in batch_prompts]
        formatted_prompts = [
            tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
            for message in messages
        ]

        encoded_outputs = tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded_outputs["input_ids"].to(device)
        attention_mask = encoded_outputs["attention_mask"].to(device)

        pairs_path = os.path.join(prompt_results_dir, f"prompt_pairs_{start:05d}_{end - 1:05d}.npy")
        print(f"[RUN] prompt batch {start}-{end - 1}, pairs={pairs_path}")

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

        for local_idx, (prompt, prediction) in enumerate(zip(batch_prompts, output_text)):
            sample_idx = start + local_idx
            records.append(
                {
                    "index": sample_idx,
                    "prompt": prompt,
                    "prediction": prediction,
                    "pairs_path": pairs_path,
                }
            )
            print(f"[{sample_idx}] {prediction}")
            print("-" * 50)

    output_jsonl = os.path.join(prompt_results_dir, "prompt_outputs.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} samples to {output_jsonl}")


if __name__ == "__main__":
    main()
