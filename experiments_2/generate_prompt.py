import torch
import numpy as np
import torch.nn.functional as F
import argparse
import json
import os

from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from generate import generate, _collect_and_save_attention_weights


DEFAULT_PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
]


def load_prompts_from_file(prompts_file: str, prompt_key: str = "prompt") -> list[str]:
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


def main():
    parser = argparse.ArgumentParser(description="Run LLaDA generation for manual prompts and collect attention dynamics")
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompts-file", type=str, default=None, help="Optional .txt/.jsonl prompts file")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Prompt field name when using .jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--logits-eos-inf", action="store_true", help="Set EOS logit to -inf")
    parser.add_argument("--confidence-eos-eot-inf", action="store_true", help="Set EOS/EoT confidence to -inf")
    parser.add_argument("--local-half-window", type=int, default=32, help="Local window size for calculating global ratio.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-txt", type=str, default=None, help="Optional txt output path")
    parser.add_argument("--dynamic-window-size", type=int, default=None, help="L-Shape mask window size W. Set to enable L-Shape mask intervention.")
    parser.add_argument("--record-attention", action="store_true", help="Record and save post-softmax attention weights for offline analysis.")
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

    # Configure L-Shape mask and attention recording for all relevant config objects
    configs_to_update = [model.config]
    if hasattr(model, "model") and hasattr(model.model, "config"):
        configs_to_update.append(model.model.config)
    if hasattr(model, "transformer") and hasattr(model.transformer, "config"):
        configs_to_update.append(model.transformer.config)

    for cfg in configs_to_update:
        if args.dynamic_window_size is not None:
            cfg.dynamic_window_size = args.dynamic_window_size
        if args.record_attention:
            cfg.record_attention = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    assert tokenizer.pad_token_id != 126336

    os.makedirs(args.results_dir, exist_ok=True)
    dataset_results_dir = os.path.join(args.results_dir, "prompt")
    os.makedirs(dataset_results_dir, exist_ok=True)

    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file, args.prompt_key)
        print(f"[INFO] Loaded {len(prompts)} prompts from file: {args.prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"[INFO] Using built-in {len(prompts)} default prompts")
    if len(prompts) == 0:
        raise RuntimeError("No prompts to run.")

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

        # Set prompt_length on config BEFORE generation
        prompt_len = input_ids.shape[1]
        for cfg in configs_to_update:
            cfg.prompt_length = prompt_len

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
            logits_eos_inf=args.logits_eos_inf,
            confidence_eos_eot_inf=args.confidence_eos_eot_inf,
            save_dynamics_path=dynamics_path,
            local_half_window=args.local_half_window
        )
        output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)

        # Save attention weights if recording is enabled
        if args.record_attention:
            _collect_and_save_attention_weights(
                model, dataset_results_dir,
                tag=f"{start:05d}_{end - 1:05d}",
                prompt_length=prompt_len,
            )
        
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

    output_txt = args.output_txt or os.path.join(dataset_results_dir, "prompt_outputs.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        for record in records:
            f.write(f"[{record['index']}] Prompt:\n{record['prompt']}\n\n")
            f.write(f"Prediction:\n{record['prediction']}\n")
            f.write("-" * 80 + "\n")
    print(f"Saved txt outputs to {output_txt}")


if __name__ == '__main__':
    main()
