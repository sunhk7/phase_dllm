import torch
import numpy as np
import torch.nn.functional as F
import argparse
import json
import os
from typing import Optional

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from model.modeling_llada import LLaDAModelLM


def _iter_model_layers(model):
    if not hasattr(model, "model"):
        return []

    if hasattr(model.model, "layers"):
        return list(model.model.layers)

    if hasattr(model.model, "transformer"):
        transformer = model.model.transformer
        if hasattr(transformer, "blocks"):
            return list(transformer.blocks)

        if hasattr(transformer, "block_groups"):
            modules = []
            for group in transformer.block_groups:
                modules.extend(list(group))
            return modules

    return []


def _get_attention_module(layer):
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    return layer


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
             confidence_eos_eot_inf=False, collect_attention_dynamics=True,
             save_dynamics_path='llada_8b_attention_dynamics.npy', local_half_window=32,
             collect_entropy_ratio_pairs=True, save_entropy_ratio_path='entropy_ratio_pairs.npy',
             entropy_layer_index=24, strategy_mode='none',
             entropy_threshold: Optional[float] = None, entropy_quantile: float = 0.5,
             collect_entropy_ratio_meta=False, save_entropy_ratio_meta_path='entropy_ratio_meta.npy'):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        collect_attention_dynamics: Whether to collect per-layer global attention ratios each diffusion step.
        save_dynamics_path: Path to save attention dynamics matrix (.npy). Set to None to disable saving.
        collect_entropy_ratio_pairs: Whether to collect token-level entropy/global-ratio pairs.
        save_entropy_ratio_path: Path to save paired data as shape (N, 2) numpy array.
        entropy_layer_index: 1-based transformer layer index used to extract global ratios (default: 24).
        strategy_mode: Selective context strategy. 'none'|'A'|'B'|'C'.
            A: low-entropy tokens use local attention in the next step.
            B: high-entropy tokens use local attention in the next step.
            C: all tokens use global attention (baseline).
        entropy_threshold: Absolute entropy threshold. If None, use entropy_quantile per step.
        entropy_quantile: Quantile used to split low/high entropy groups when threshold is None.
        collect_entropy_ratio_meta: Save 4-col meta [entropy, ratio, is_updated, used_local].
        save_entropy_ratio_meta_path: Path to save the meta matrix.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    layer_modules = _iter_model_layers(model) if (collect_attention_dynamics or collect_entropy_ratio_pairs) else []
    attention_modules = []
    for layer_module in layer_modules:
        attention_module = _get_attention_module(layer_module)
        if attention_module is None:
            continue
        attention_module.global_ratio_tracker = []
        attention_module.local_half_window = local_half_window
        attention_module.context_length = prompt.shape[1]
        attention_modules.append(attention_module)

    attention_dynamics = [] if (collect_attention_dynamics and attention_modules) else None
    entropy_ratio_pairs = [] if collect_entropy_ratio_pairs else None
    entropy_ratio_meta = [] if collect_entropy_ratio_meta else None
    if attention_modules and entropy_layer_index > 0:
        target_entropy_layer_idx = min(entropy_layer_index - 1, len(attention_modules) - 1)
    else:
        target_entropy_layer_idx = None

    strategy_mode = str(strategy_mode).upper()
    if strategy_mode not in {"NONE", "A", "B", "C"}:
        raise ValueError(f"Unknown strategy_mode: {strategy_mode}")
    next_step_local_mask = None

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            # Apply selective local/global policy (decided from previous step entropy).
            for attention_module in attention_modules:
                attention_module.query_local_mask = next_step_local_mask
            if next_step_local_mask is not None:
                used_local_np = next_step_local_mask.detach().cpu().numpy().astype(np.float32)
            else:
                used_local_np = None

            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            entropy_np = None
            ratio_np = None
            if entropy_ratio_pairs is not None or entropy_ratio_meta is not None or strategy_mode in {"A", "B"}:
                gen_logits = logits[:, prompt.shape[1]:, :]
                gen_probs = F.softmax(gen_logits.float(), dim=-1)
                entropy = -(gen_probs * torch.log(gen_probs + 1e-9)).sum(dim=-1)
                entropy_np = entropy.detach().cpu().numpy().astype(np.float32)

                if (
                    target_entropy_layer_idx is not None
                    and attention_modules[target_entropy_layer_idx].global_ratio_tracker
                ):
                    ratio_np = np.asarray(
                        attention_modules[target_entropy_layer_idx].global_ratio_tracker[-1],
                        dtype=np.float32,
                    )
                else:
                    ratio_np = np.full_like(entropy_np, np.nan, dtype=np.float32)

                if entropy_ratio_pairs is not None:
                    flat_entropy = entropy_np.reshape(-1)
                    flat_ratio = ratio_np.reshape(-1)
                    pair_count = min(flat_entropy.shape[0], flat_ratio.shape[0])
                    if pair_count > 0:
                        entropy_ratio_pairs.append(
                            np.stack((flat_entropy[:pair_count], flat_ratio[:pair_count]), axis=-1)
                        )
                    del flat_entropy, flat_ratio

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # Build next-step local mask from current entropy according to strategy.
            if strategy_mode in {"A", "B"} and entropy_np is not None:
                if entropy_threshold is not None:
                    cur_threshold = float(entropy_threshold)
                else:
                    cur_threshold = float(np.nanquantile(entropy_np, entropy_quantile))
                if strategy_mode == "A":
                    next_local_np = entropy_np <= cur_threshold
                else:
                    next_local_np = entropy_np > cur_threshold
                next_step_local_mask = torch.from_numpy(next_local_np).to(device=logits.device, dtype=torch.bool)
                del next_local_np
            elif strategy_mode in {"C", "NONE"}:
                next_step_local_mask = None

            # Optional meta dump for hypothesis testing:
            # col0=entropy, col1=ratio, col2=is_updated, col3=used_local.
            if entropy_ratio_meta is not None and entropy_np is not None and ratio_np is not None:
                updated_np = transfer_index[:, prompt.shape[1]:].detach().cpu().numpy().astype(np.float32)
                if used_local_np is None:
                    used_local_flat = np.zeros_like(updated_np.reshape(-1), dtype=np.float32)
                else:
                    used_local_flat = used_local_np.reshape(-1).astype(np.float32)

                flat_entropy = entropy_np.reshape(-1)
                flat_ratio = ratio_np.reshape(-1)
                flat_updated = updated_np.reshape(-1)
                pair_count = min(flat_entropy.shape[0], flat_ratio.shape[0], flat_updated.shape[0], used_local_flat.shape[0])
                if pair_count > 0:
                    entropy_ratio_meta.append(
                        np.stack(
                            (
                                flat_entropy[:pair_count],
                                flat_ratio[:pair_count],
                                flat_updated[:pair_count],
                                used_local_flat[:pair_count],
                            ),
                            axis=-1,
                        )
                    )
                del updated_np, used_local_flat, flat_entropy, flat_ratio, flat_updated

            if attention_dynamics is not None:
                step_ratio = []
                for attention_module in attention_modules:
                    if hasattr(attention_module, "global_ratio_tracker") and attention_module.global_ratio_tracker:
                        value = attention_module.global_ratio_tracker[-1]
                        if isinstance(value, torch.Tensor):
                            value = value.float().mean().cpu().item()
                        else:
                            value_arr = np.asarray(value, dtype=np.float32)
                            value = float(np.nanmean(value_arr)) if value_arr.size > 0 else float("nan")
                    else:
                        value = float("nan")
                    step_ratio.append(value)
                attention_dynamics.append(step_ratio)
            if entropy_np is not None:
                del gen_logits, gen_probs, entropy

    if attention_dynamics is not None and save_dynamics_path:
        np.save(save_dynamics_path, np.asarray(attention_dynamics, dtype=np.float32))
    if entropy_ratio_pairs is not None and save_entropy_ratio_path:
        if entropy_ratio_pairs:
            paired_array = np.concatenate(entropy_ratio_pairs, axis=0).astype(np.float32)
        else:
            paired_array = np.empty((0, 2), dtype=np.float32)
        np.save(save_entropy_ratio_path, paired_array)
    if entropy_ratio_meta is not None and save_entropy_ratio_meta_path:
        if entropy_ratio_meta:
            meta_array = np.concatenate(entropy_ratio_meta, axis=0).astype(np.float32)
        else:
            meta_array = np.empty((0, 4), dtype=np.float32)
        np.save(save_entropy_ratio_meta_path, meta_array)

    return x


def main():
    parser = argparse.ArgumentParser(description="Run LLaDA generation and collect attention dynamics")
    parser.add_argument("--model-id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=100)
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

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    os.makedirs(args.results_dir, exist_ok=True)
    dataset_results_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(dataset_results_dir, exist_ok=True)




    if args.dataset != "gsm8k":
        raise NotImplementedError(args.dataset)

    dataset = load_dataset("gsm8k", "main", split=args.split)
    total_samples = min(len(dataset), args.max_samples)
    records = []

    for start in range(0, total_samples, args.batch_size):
        end = min(start + args.batch_size, total_samples)
        batch = dataset[start:end]
        questions = batch["question"]
        answers = batch["answer"]

        messages = [{"role": "user", "content": f"Solve the math problem.\n\nQuestion: {q}"} for q in questions]
        prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

        encoded_outputs = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded_outputs["input_ids"].to(device)
        attention_mask = encoded_outputs["attention_mask"].to(device)

        dynamics_path = os.path.join(dataset_results_dir, f"gsm8k_dynamics_{start:05d}_{end - 1:05d}.npy")
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
            save_dynamics_path=dynamics_path,
            local_half_window=args.local_half_window,
        )
        output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)

        for local_idx, (question, answer, prediction) in enumerate(zip(questions, answers, output_text)):
            sample_idx = start + local_idx
            records.append(
                {
                    "index": sample_idx,
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                    "dynamics_path": dynamics_path,
                }
            )
            print(f"[{sample_idx}] {prediction}")
            print("-" * 50)

    output_jsonl = os.path.join(dataset_results_dir, "gsm8k_outputs.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} samples to {output_jsonl}")

if __name__ == '__main__':
    main()
