import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from generate import generate
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM


def sanitize_dataset_name(name: str) -> str:
    """将数据集名转换为文件系统安全前缀。"""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.-")
    return safe or "dataset"


def build_task_prompt(task_name: str, context: str) -> str:
    """把原始上下文包装成明确任务提示，减少模型输出乱码概率。"""
    if task_name == "text_continuation":
        return (
            "Continue the following passage in fluent, coherent English. "
            "Keep the same topic and style. Only output the continuation.\n\n"
            f"Passage:\n{context}\n\nContinuation:"
        )
    return f"Task: {task_name}\n\nInput:\n{context}\n\nOutput:"


def build_dummy_model(device: str) -> LLaDAModelLM:
    """构建一个极小的随机权重模型，用于验证完整数据闭环。"""
    config = LLaDAConfig(
        n_layers=4,
        d_model=256,
        n_heads=8,
        n_kv_heads=8,
        mlp_ratio=4,
        rope=True,
        alibi=False,
        max_sequence_length=2048,
        vocab_size=130000,
        embedding_size=130048,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
    )
    model = LLaDAModelLM(config=config, init_params=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool value: {text}")


def resolve_torch_dtype(device: str, dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def build_model(
    model_source: str,
    model_id: str,
    trust_remote_code: bool,
    model_dtype_name: str,
    device: str,
) -> LLaDAModelLM:
    if model_source == "dummy":
        return build_dummy_model(device)

    model_dtype = resolve_torch_dtype(device, model_dtype_name)
    model = LLaDAModelLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=model_dtype,
    )
    return model.to(device).eval()


def select_wikitext_samples(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_split: str,
    samples: int,
    prompt_length: int,
) -> list[dict]:
    """从 wikitext 连续拼接文本中切分多个 prompt 样本。"""
    dataset = load_dataset("wikitext", dataset_name, split=dataset_split)

    min_required = prompt_length
    stride = max(64, prompt_length // 2)
    newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]

    valid_samples = []
    token_buffer: list[int] = []
    last_row_index = -1

    for idx, row in enumerate(dataset):
        text = row.get("text", "").strip()
        if not text:
            continue

        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if newline_ids:
            token_buffer.extend(newline_ids)
        last_row_index = idx

        # 使用滑动窗口从拼接后的长 token 序列中切分样本 prompt。
        while len(token_buffer) >= min_required:
            prompt_ids = token_buffer[:prompt_length]
            valid_samples.append(
                {
                    "index": last_row_index,
                    "prompt_ids": prompt_ids,
                }
            )
            if len(valid_samples) >= samples * 12:
                break
            token_buffer = token_buffer[stride:]

        if len(valid_samples) >= samples * 12:
            break

    if len(valid_samples) < samples:
        raise RuntimeError(
            "可用样本不足: "
            f"需要 {samples} 条，实际仅 {len(valid_samples)} 条。"
            f"dataset={dataset_name}/{dataset_split}, min_required_tokens={min_required}"
        )

    selected_indices = random.sample(range(len(valid_samples)), samples)
    return [valid_samples[i] for i in selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect entropy/global-ratio pairs and save wikitext inference jsonl")
    parser.add_argument("--model-source", type=str, default="pretrained", choices=["pretrained", "dummy"])
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--trust-remote-code", type=str, default="true")
    parser.add_argument("--model-dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--dataset-name", type=str, default="wikitext-103-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tokenizer-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=768)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=768)
    parser.add_argument("--output-dir", type=str, default="results/wikitext")
    parser.add_argument("--pairs-name-suffix", type=str, default="entropy_ratio_pairs")
    parser.add_argument("--output-jsonl", type=str, default="results/wikitext/wikitext_outputs.jsonl")
    parser.add_argument("--task-name", type=str, default="text_continuation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 固定随机种子，保证同一配置下结果可复现。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    safe_dataset_name = sanitize_dataset_name(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=parse_bool(args.trust_remote_code))
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id == 126336:
        raise RuntimeError("pad_token_id equals mask_id(126336), current generate() path does not support this case.")

    selected_samples = select_wikitext_samples(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        samples=args.samples,
        prompt_length=args.prompt_length,
    )
    print(
        f"[INFO] dataset={args.dataset_name}/{args.dataset_split}, selected_samples={len(selected_samples)}, "
        f"prompt_length={args.prompt_length}, gen_length={args.gen_length}, steps={args.steps}",
        flush=True,
    )
    model = build_model(
        model_source=args.model_source,
        model_id=args.model_id,
        trust_remote_code=parse_bool(args.trust_remote_code),
        model_dtype_name=args.model_dtype,
        device=device,
    )
    print(
        f"[INFO] model_source={args.model_source}, model_id={args.model_id}, device={device}, dtype={args.model_dtype}",
        flush=True,
    )

    with open(args.output_jsonl, "a", encoding="utf-8") as f_jsonl:
        for sample_id, sample in enumerate(selected_samples):
            source_context = tokenizer.decode(sample["prompt_ids"], skip_special_tokens=True)
            task_prompt = build_task_prompt(args.task_name, source_context)
            if hasattr(tokenizer, "apply_chat_template"):
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": task_prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                formatted_prompt = task_prompt

            encoded = tokenizer(
                [formatted_prompt],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            print(
                f"[RUN] dataset={args.dataset_name} sample={sample_id + 1}/{len(selected_samples)} "
                f"index={sample['index']} task={args.task_name}",
                flush=True,
            )
            sample_key = f"{safe_dataset_name}_sample_{sample_id:03d}_{args.pairs_name_suffix}"
            sample_pairs_path = str(Path(args.output_dir) / f"{sample_key}.npy")

            # 推理阶段使用 inference_mode，减少显存与 autograd 开销。
            with torch.inference_mode():
                out = generate(
                    model,
                    input_ids,
                    attention_mask=attention_mask,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    collect_attention_dynamics=False,
                    save_dynamics_path=None,
                    collect_entropy_ratio_pairs=True,
                    save_entropy_ratio_path=sample_pairs_path,
                    entropy_layer_index=24,
                )

            sample_pairs = np.load(sample_pairs_path).astype(np.float32)

            prediction_ids = out[:, input_ids.shape[1]:].detach().cpu().numpy()[0].tolist()
            prediction_text = tokenizer.decode(prediction_ids, skip_special_tokens=True)

            record = {
                "dataset": args.dataset_name,
                "index": int(sample["index"]),
                "task": args.task_name,
                "inputs": {
                    "source_context": source_context,
                    "task_prompt": task_prompt,
                    "formatted_prompt": formatted_prompt,
                },
                "outputs": {
                    "generated_text": prediction_text,
                },
                "artifacts": {
                    "pairs_path": sample_pairs_path,
                },
            }
            # 追加写入：每个样本一行 JSON。
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[DONE] dataset={args.dataset_name} sample={sample_id + 1}/{len(selected_samples)} "
                f"pairs_shape={sample_pairs.shape} pairs_path={sample_pairs_path}",
                flush=True,
            )

            del sample_pairs, out, input_ids, attention_mask
            if device == "cuda":
                torch.cuda.empty_cache()

    expected_points = args.steps * args.gen_length
    print(f"Saved per-sample entropy/global-ratio npy files to: {args.output_dir}")
    print(f"Each sample shape is expected close to ({expected_points}, 2)")
    print(f"Appended {args.samples} inference records to: {args.output_jsonl}")

    # 显式释放大对象，避免后续实验显存累计。
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
