"""
Fused Swin Windowed Attention via OpenAI Triton.

Replaces the fragmented PyTorch ops (F.pad → view → matmul → max → exp → sum → div)
with a single GPU kernel launch per window batch.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _swin_win_attn_fwd(
    Q, K, V, Out, Lse,
    D,  # original sequence length (not constexpr, can vary)
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    SHIFT: tl.constexpr,
    W: tl.constexpr,
    DHEAD: tl.constexpr,
    NUM_WIN: tl.constexpr,
    H: tl.constexpr,
):
    """Each program instance handles one (batch, head, window)."""
    pid = tl.program_id(0)
    b  = pid // (H * NUM_WIN)
    rem = pid % (H * NUM_WIN)
    h  = rem // NUM_WIN
    wid = rem % NUM_WIN

    scale = 1.0 / tl.sqrt(DHEAD + 0.0)

    pos   = tl.arange(0, W)          # [W]
    d_idx = tl.arange(0, DHEAD)      # [DHEAD]

    # Map padded window positions → original positions
    orig = wid * W + pos - SHIFT     # [W]
    valid = (orig >= 0) & (orig < D)
    safe  = tl.where(valid, orig, 0) # clamp for safe mem access

    # ---- Load Q, K, V tiles  [W, DHEAD] ----
    q = tl.load(Q + b*stride_qb + h*stride_qh + safe[:, None]*stride_qs + d_idx[None, :]*stride_qd,
                mask=valid[:, None], other=0.0)
    k = tl.load(K + b*stride_kb + h*stride_kh + safe[:, None]*stride_ks + d_idx[None, :]*stride_kd,
                mask=valid[:, None], other=0.0)
    v = tl.load(V + b*stride_vb + h*stride_vh + safe[:, None]*stride_vs + d_idx[None, :]*stride_vd,
                mask=valid[:, None], other=0.0)

    # ---- Attention scores [W, W] (fp32 via tensor-cores) ----
    qk = tl.dot(q, tl.trans(k)) * scale
    qk = tl.where(valid[None, :] & valid[:, None], qk, float('-inf'))

    # ---- Online softmax ----
    row_max = tl.max(qk, axis=1)
    row_max = tl.where(valid, row_max, 0.0)
    exp_qk  = tl.exp(qk - row_max[:, None])
    exp_qk  = tl.where(valid[None, :] & valid[:, None], exp_qk, 0.0)
    row_sum  = tl.sum(exp_qk, axis=1)
    row_sum  = tl.where(valid, row_sum, 1.0)

    # ---- Weighted value [W, DHEAD] ----
    out = tl.dot(exp_qk.to(v.dtype), v)
    out = out / row_sum[:, None]

    lse = row_max + tl.log(row_sum)
    lse = tl.where(valid, lse, float('-inf'))

    # ---- Store (only valid original positions) ----
    tl.store(Out + b*stride_ob + h*stride_oh + safe[:, None]*stride_os + d_idx[None, :]*stride_od,
             out.to(tl.bfloat16), mask=valid[:, None])
    tl.store(Lse + b*stride_lb + h*stride_lh + safe*stride_ls,
             lse, mask=valid)


# ---------------------------------------------------------------------------
#  Python wrapper – called from modeling_llada.py
# ---------------------------------------------------------------------------
def swin_triton_attention(q, k_block, v_block, k_prefix, v_prefix, w, S, layer_id):
    """
    Args
        q, k_block, v_block : [B, H, D, d_head]  (block portion, GQA already expanded)
        k_prefix, v_prefix  : [B, H, P, d_head]  (prompt KV)
        w  : window size (must be >= 16 and power-of-2)
        S  : shift amount
        layer_id : determines even/odd shift
    Returns
        att : [B, H, D, d_head]
    """
    B, H, D, d_head = q.shape
    shift   = S if layer_id % 2 == 1 else 0
    padded  = D + 2 * shift if shift > 0 else D
    num_win = padded // w

    # ---- 1. Triton kernel: windowed attention → (out_win, lse_win) ----
    out_win = torch.zeros_like(q)
    lse_win = torch.full((B, H, D), float('-inf'), device=q.device, dtype=torch.float32)

    grid = (B * H * num_win,)
    _swin_win_attn_fwd[grid](
        q, k_block, v_block, out_win, lse_win,
        D,
        *q.stride(), *k_block.stride(), *v_block.stride(), *out_win.stride(),
        lse_win.stride(0), lse_win.stride(1), lse_win.stride(2),
        SHIFT=shift, W=w, DHEAD=d_head, NUM_WIN=num_win, H=H,
    )

    # ---- 2. Prefix attention (vanilla PyTorch, prefix is short) ----
    scale    = 1.0 / math.sqrt(d_head)
    s_p      = torch.matmul(q, k_prefix.transpose(-1, -2)) * scale   # [B,H,D,P]
    max_p    = s_p.max(dim=-1).values                                  # [B,H,D]
    exp_p    = torch.exp(s_p - max_p.unsqueeze(-1))
    sum_p    = exp_p.sum(dim=-1)                                       # [B,H,D]
    lse_pref = max_p + torch.log(sum_p)
    out_pref = torch.matmul(exp_p, v_prefix) / sum_p.unsqueeze(-1)

    # ---- 3. LogSumExp combination ----
    lse_total = torch.logaddexp(lse_win, lse_pref)
    alpha = torch.exp(lse_win  - lse_total).unsqueeze(-1)
    beta  = torch.exp(lse_pref - lse_total).unsqueeze(-1)

    return (alpha * out_win + beta * out_pref).to(q.dtype)
