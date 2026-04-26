"""
Fused Swin Windowed Attention via OpenAI Triton — with native GQA support.

Replaces the fragmented PyTorch ops (repeat_interleave + F.pad → view → matmul → max → exp → sum → div)
with a single GPU kernel launch per window batch. K/V heads are indexed via h_kv = h_q // gqa_ratio,
avoiding all physical memory copies.
"""
import torch
import triton
import triton.language as tl
import math


@triton.jit
def _swin_win_attn_fwd(
    Q, K, V, Out, Lse,
    D,  # original sequence length
    GQA_RATIO: tl.constexpr,  # num_q_heads // num_kv_heads (e.g. 4)
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    SHIFT: tl.constexpr,
    W: tl.constexpr,
    DHEAD: tl.constexpr,
    NUM_WIN: tl.constexpr,
    H_Q: tl.constexpr,   # number of Q heads (e.g. 32)
):
    """Each program instance handles one (batch, q_head, window)."""
    pid = tl.program_id(0)
    b   = pid // (H_Q * NUM_WIN)
    rem = pid % (H_Q * NUM_WIN)
    h_q = rem // NUM_WIN
    wid = rem % NUM_WIN

    # GQA: map Q head → KV head
    h_kv = h_q // GQA_RATIO

    scale = 1.0 / tl.sqrt(DHEAD + 0.0)

    pos   = tl.arange(0, W)
    d_idx = tl.arange(0, DHEAD)

    # Map padded window positions → original positions
    orig  = wid * W + pos - SHIFT
    valid = (orig >= 0) & (orig < D)
    safe  = tl.where(valid, orig, 0)

    # ---- Load Q [W, DHEAD] from Q heads ----
    q = tl.load(Q + b*stride_qb + h_q*stride_qh + safe[:, None]*stride_qs + d_idx[None, :]*stride_qd,
                mask=valid[:, None], other=0.0)

    # ---- Load K, V [W, DHEAD] from KV heads (no repeat_interleave!) ----
    k = tl.load(K + b*stride_kb + h_kv*stride_kh + safe[:, None]*stride_ks + d_idx[None, :]*stride_kd,
                mask=valid[:, None], other=0.0)
    v = tl.load(V + b*stride_vb + h_kv*stride_vh + safe[:, None]*stride_vs + d_idx[None, :]*stride_vd,
                mask=valid[:, None], other=0.0)

    # ---- Attention scores [W, W] ----
    qk = tl.dot(q, tl.trans(k)) * scale
    qk = tl.where(valid[None, :] & valid[:, None], qk, float('-inf'))

    # ---- Softmax ----
    row_max = tl.max(qk, axis=1)
    row_max = tl.where(valid, row_max, 0.0)
    exp_qk  = tl.exp(qk - row_max[:, None])
    exp_qk  = tl.where(valid[None, :] & valid[:, None], exp_qk, 0.0)
    row_sum  = tl.sum(exp_qk, axis=1)
    row_sum  = tl.where(valid, row_sum, 1.0)

    # ---- Output [W, DHEAD] ----
    out = tl.dot(exp_qk.to(v.dtype), v)
    out = out / row_sum[:, None]

    lse = row_max + tl.log(row_sum)
    lse = tl.where(valid, lse, float('-inf'))

    # ---- Store (indexed by Q head) ----
    tl.store(Out + b*stride_ob + h_q*stride_oh + safe[:, None]*stride_os + d_idx[None, :]*stride_od,
             out.to(tl.bfloat16), mask=valid[:, None])
    tl.store(Lse + b*stride_lb + h_q*stride_lh + safe*stride_ls,
             lse, mask=valid)


# ---------------------------------------------------------------------------
#  Python wrapper
# ---------------------------------------------------------------------------
def swin_triton_attention(q, k_block, v_block, k_prefix, v_prefix, w, S, layer_id):
    """
    Args (GQA-aware: Q has more heads than K/V)
        q           : [B, H_Q,  D, d_head]   (e.g. H_Q=32)
        k_block/v_block : [B, H_KV, D, d_head]   (e.g. H_KV=8, NOT expanded)
        k_prefix/v_prefix : [B, H_KV, P, d_head]
        w  : window size (>= 16, power-of-2)
        S  : shift amount
        layer_id : even/odd determines shift
    Returns
        att : [B, H_Q, D, d_head]
    """
    B, H_Q, D, d_head = q.shape
    H_KV = k_block.shape[1]
    GQA_RATIO = H_Q // H_KV

    shift   = S if layer_id % 2 == 1 else 0
    padded  = D + 2 * shift if shift > 0 else D
    num_win = padded // w

    # ---- 1. Triton kernel: windowed attention (native GQA) ----
    out_win = torch.zeros_like(q)
    lse_win = torch.full((B, H_Q, D), float('-inf'), device=q.device, dtype=torch.float32)

    grid = (B * H_Q * num_win,)
    _swin_win_attn_fwd[grid](
        q, k_block, v_block, out_win, lse_win,
        D, GQA_RATIO,
        *q.stride(), *k_block.stride(), *v_block.stride(), *out_win.stride(),
        lse_win.stride(0), lse_win.stride(1), lse_win.stride(2),
        SHIFT=shift, W=w, DHEAD=d_head, NUM_WIN=num_win, H_Q=H_Q,
    )

    # ---- 2. Prefix attention (native GQA via expand, no physical copy) ----
    scale = 1.0 / math.sqrt(d_head)
    # Expand KV to match Q heads via broadcast (zero-copy view)
    k_pref_exp = k_prefix.unsqueeze(2).expand(B, H_KV, GQA_RATIO, -1, d_head).reshape(B, H_Q, -1, d_head)
    v_pref_exp = v_prefix.unsqueeze(2).expand(B, H_KV, GQA_RATIO, -1, d_head).reshape(B, H_Q, -1, d_head)

    s_p      = torch.matmul(q, k_pref_exp.transpose(-1, -2)) * scale
    max_p    = s_p.max(dim=-1).values
    exp_p    = torch.exp(s_p - max_p.unsqueeze(-1))
    sum_p    = exp_p.sum(dim=-1)
    lse_pref = max_p + torch.log(sum_p)
    out_pref = torch.matmul(exp_p, v_pref_exp) / sum_p.unsqueeze(-1)

    # ---- 3. LogSumExp combination ----
    lse_total = torch.logaddexp(lse_win, lse_pref)
    alpha = torch.exp(lse_win  - lse_total).unsqueeze(-1)
    beta  = torch.exp(lse_pref - lse_total).unsqueeze(-1)

    return (alpha * out_win + beta * out_pref).to(q.dtype)
