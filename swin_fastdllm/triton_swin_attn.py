"""
Fully-optimised Swin Windowed Attention — Triton kernels + zero-copy GQA.

Three fused stages:
  1. _swin_win_attn_fwd  : pad → window partition → attention  (Triton, native GQA)
  2. Prefix attention     : grouped-broadcast matmul (zero-copy GQA, no expand/reshape)
  3. _lse_combine_fwd     : α·out_win + β·out_pref with fused LogSumExp (Triton)
"""
import torch
import triton
import triton.language as tl
import math


# =====================================================================
#  Kernel 1: Windowed Attention with native GQA
# =====================================================================
@triton.jit
def _swin_win_attn_fwd(
    Q, K, V, Out, Lse,
    D,
    GQA_RATIO: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    SHIFT: tl.constexpr,
    W: tl.constexpr,
    DHEAD: tl.constexpr,
    NUM_WIN: tl.constexpr,
    H_Q: tl.constexpr,
):
    pid = tl.program_id(0)
    b   = pid // (H_Q * NUM_WIN)
    rem = pid %  (H_Q * NUM_WIN)
    h_q = rem // NUM_WIN
    wid = rem %  NUM_WIN
    h_kv = h_q // GQA_RATIO

    scale = 1.0 / tl.sqrt(DHEAD + 0.0)
    pos   = tl.arange(0, W)
    d_idx = tl.arange(0, DHEAD)

    orig  = wid * W + pos - SHIFT
    valid = (orig >= 0) & (orig < D)
    safe  = tl.where(valid, orig, 0)

    q = tl.load(Q + b*stride_qb + h_q*stride_qh  + safe[:, None]*stride_qs + d_idx[None, :]*stride_qd,
                mask=valid[:, None], other=0.0)
    k = tl.load(K + b*stride_kb + h_kv*stride_kh + safe[:, None]*stride_ks + d_idx[None, :]*stride_kd,
                mask=valid[:, None], other=0.0)
    v = tl.load(V + b*stride_vb + h_kv*stride_vh + safe[:, None]*stride_vs + d_idx[None, :]*stride_vd,
                mask=valid[:, None], other=0.0)

    qk = tl.dot(q, tl.trans(k)) * scale
    qk = tl.where(valid[None, :] & valid[:, None], qk, float('-inf'))

    row_max = tl.max(qk, axis=1)
    row_max = tl.where(valid, row_max, 0.0)
    exp_qk  = tl.exp(qk - row_max[:, None])
    exp_qk  = tl.where(valid[None, :] & valid[:, None], exp_qk, 0.0)
    row_sum  = tl.sum(exp_qk, axis=1)
    row_sum  = tl.where(valid, row_sum, 1.0)

    out = tl.dot(exp_qk.to(v.dtype), v)
    out = out / row_sum[:, None]

    lse = row_max + tl.log(row_sum)
    lse = tl.where(valid, lse, float('-inf'))

    tl.store(Out + b*stride_ob + h_q*stride_oh + safe[:, None]*stride_os + d_idx[None, :]*stride_od,
             out.to(tl.bfloat16), mask=valid[:, None])
    tl.store(Lse + b*stride_lb + h_q*stride_lh + safe*stride_ls,
             lse, mask=valid)


# =====================================================================
#  Kernel 2: Fused LogSumExp Combination
#    result = exp(lse_w - lse_total) * out_w + exp(lse_p - lse_total) * out_p
# =====================================================================
@triton.jit
def _lse_combine_fwd(
    OUT_W, OUT_P, LSE_W, LSE_P, RESULT,
    DHEAD: tl.constexpr,
):
    """Each program handles one (b, h, s) position → DHEAD values."""
    row = tl.program_id(0)
    d_idx = tl.arange(0, DHEAD)

    lw = tl.load(LSE_W + row)
    lp = tl.load(LSE_P + row)

    # Numerically stable logaddexp
    m = tl.maximum(lw, lp)
    lse = m + tl.log(tl.exp(lw - m) + tl.exp(lp - m))

    alpha = tl.exp(lw - lse)
    beta  = tl.exp(lp - lse)

    ow = tl.load(OUT_W + row * DHEAD + d_idx).to(tl.float32)
    op = tl.load(OUT_P + row * DHEAD + d_idx).to(tl.float32)

    r = alpha * ow + beta * op
    tl.store(RESULT + row * DHEAD + d_idx, r.to(tl.bfloat16))


# =====================================================================
#  Python wrapper
# =====================================================================
def swin_triton_attention(q, k_block, v_block, k_prefix, v_prefix, w, S, layer_id):
    """
    Fully-fused Swin attention with native GQA everywhere.

    Args (GQA-aware: Q has more heads than K/V):
        q               : [B, H_Q,  D, d_head]
        k_block/v_block : [B, H_KV, D, d_head]   (NOT expanded)
        k_prefix/v_prefix : [B, H_KV, P, d_head]
        w  : window size (>= 16, power-of-2)
        S  : shift amount
        layer_id : even/odd determines shift
    """
    B, H_Q, D, d_head = q.shape
    H_KV = k_block.shape[1]
    GQA  = H_Q // H_KV
    P    = k_prefix.shape[2]

    shift   = S if layer_id % 2 == 1 else 0
    padded  = D + 2 * shift if shift > 0 else D
    num_win = padded // w

    # ================================================================
    #  Stage 1: Windowed attention (Triton kernel, native GQA)
    # ================================================================
    out_win = torch.zeros((B, H_Q, D, d_head), device=q.device, dtype=q.dtype)
    lse_win = torch.full((B, H_Q, D), float('-inf'), device=q.device, dtype=torch.float32)

    grid = (B * H_Q * num_win,)
    _swin_win_attn_fwd[grid](
        q, k_block, v_block, out_win, lse_win,
        D, GQA,
        *q.stride(), *k_block.stride(), *v_block.stride(), *out_win.stride(),
        lse_win.stride(0), lse_win.stride(1), lse_win.stride(2),
        SHIFT=shift, W=w, DHEAD=d_head, NUM_WIN=num_win, H_Q=H_Q,
    )

    # ================================================================
    #  Stage 2: Prefix attention (grouped broadcast — zero-copy GQA)
    # ================================================================
    scale = 1.0 / math.sqrt(d_head)

    # q grouped: [B, H_KV, GQA, D, d]
    q_g = q.view(B, H_KV, GQA, D, d_head)

    # K^T broadcast: [B, H_KV, 1, d, P]  (unsqueeze is zero-copy)
    k_t = k_prefix.transpose(-1, -2).unsqueeze(2)

    # Scores via broadcast: [B, H_KV, GQA, D, P]  (no expand/reshape copy!)
    s_p = torch.matmul(q_g, k_t) * scale

    # Softmax in f32 for stability
    s_p_f32 = s_p.float()
    max_s   = s_p_f32.max(dim=-1, keepdim=True).values
    exp_s   = (s_p_f32 - max_s).exp()
    sum_s   = exp_s.sum(dim=-1, keepdim=True)

    lse_pref = (max_s + sum_s.log()).squeeze(-1).reshape(B, H_Q, D)  # [B, H_Q, D]

    # V broadcast: [B, H_KV, 1, P, d]
    v_u = v_prefix.unsqueeze(2)
    out_pref = torch.matmul(exp_s.to(q.dtype), v_u) / sum_s.to(q.dtype)  # [B, H_KV, GQA, D, d]
    out_pref = out_pref.reshape(B, H_Q, D, d_head)

    # ================================================================
    #  Stage 3: Fused LogSumExp combination (Triton kernel)
    # ================================================================
    N = B * H_Q * D
    result = torch.empty((B, H_Q, D, d_head), device=q.device, dtype=q.dtype)

    _lse_combine_fwd[(N,)](
        out_win,
        out_pref.contiguous(),
        lse_win,
        lse_pref.contiguous(),
        result,
        DHEAD=d_head,
    )

    # Return result with original `q` strides to match standard PyTorch flow
    return result.view(q.shape)
