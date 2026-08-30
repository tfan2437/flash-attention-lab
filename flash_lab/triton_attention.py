"""FlashAttention-2 forward in Triton, registered as torch.ops.flash_lab.attention_triton.

One program owns BLOCK_M query rows of one (batch, head) and walks the keys in tiles of BLOCK_N.
Scores are scaled into the log2 domain so the online softmax uses exp2. Tiles where every row
sees every key run without masks; only the tiles on the causal diagonal or past the end of the
keys pay for the comparison, so the key loop is split in two.

Block sizes, warps, and pipeline stages are autotuned per (head_dim, causal, length bucket).
FLASH_LAB_TRITON_AUTOTUNE=0 pins one configuration instead, which keeps test runs from compiling
every candidate for every shape.
"""

import math
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton on this platform (macOS); the op is simply not registered
    triton = None

LOG2E = 1.4426950408889634

if triton is not None:

    @triton.jit
    def _tile_loop(
        acc,
        m_i,
        l_i,
        q,
        k_base,
        v_base,
        stride_ks,
        stride_vs,
        rows,
        d,
        start,
        end,
        seqlen_k,
        causal_offset,
        scale_log2,
        CAUSAL: tl.constexpr,
        MASKED: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        for start_n in range(start, end, BLOCK_N):
            cols = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_base + cols[:, None] * stride_ks + d[None, :]
            v_ptrs = v_base + cols[:, None] * stride_vs + d[None, :]
            if MASKED:
                in_range = cols < seqlen_k
                k = tl.load(k_ptrs, mask=in_range[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=in_range[:, None], other=0.0)
            else:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)

            s = tl.dot(q, tl.trans(k)) * scale_log2
            if MASKED:
                keep = in_range[None, :]
                if CAUSAL:
                    keep = keep & (cols[None, :] <= rows[:, None] + causal_offset)
                s = tl.where(keep, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, 1))
            # Rows that have only seen masked keys keep a max of -inf; 0 as the reference
            # point makes their exponentials exactly 0 instead of nan.
            ref = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2(m_i - ref)
            p = tl.math.exp2(s - ref[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_new
        return acc, m_i, l_i

    @triton.jit
    def _attention_fwd_kernel(
        Q,
        K,
        V,
        Out,
        LSE,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_ob,
        stride_os,
        stride_oh,
        heads,
        group,
        seqlen_q,
        seqlen_k,
        scale_log2,
        SEQ_BUCKET,  # only used as an autotune key
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // heads
        h = pid_bh % heads
        h_kv = h // group
        causal_offset = seqlen_k - seqlen_q  # bottom-right alignment

        row0 = pid_m * BLOCK_M
        rows = row0 + tl.arange(0, BLOCK_M)
        d = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + b * stride_qb + h * stride_qh + rows[:, None] * stride_qs + d[None, :]
        q = tl.load(q_ptrs, mask=rows[:, None] < seqlen_q, other=0.0)
        k_base = K + b * stride_kb + h_kv * stride_kh
        v_base = V + b * stride_vb + h_kv * stride_vh

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        if CAUSAL:
            # The last row of the block sees keys up to its index + offset; tiles that end at or
            # before the first row's limit are visible to every row and need no mask.
            end = tl.maximum(tl.minimum(seqlen_k, row0 + BLOCK_M + causal_offset), 0)
            full_end = tl.maximum(tl.minimum(row0 + causal_offset + 1, end), 0)
        else:
            end = seqlen_k
            full_end = seqlen_k
        full_end = (full_end // BLOCK_N) * BLOCK_N

        acc, m_i, l_i = _tile_loop(
            acc,
            m_i,
            l_i,
            q,
            k_base,
            v_base,
            stride_ks,
            stride_vs,
            rows,
            d,
            0,
            full_end,
            seqlen_k,
            causal_offset,
            scale_log2,
            CAUSAL=CAUSAL,
            MASKED=False,
            BLOCK_N=BLOCK_N,
        )
        acc, m_i, l_i = _tile_loop(
            acc,
            m_i,
            l_i,
            q,
            k_base,
            v_base,
            stride_ks,
            stride_vs,
            rows,
            d,
            full_end,
            end,
            seqlen_k,
            causal_offset,
            scale_log2,
            CAUSAL=CAUSAL,
            MASKED=True,
            BLOCK_N=BLOCK_N,
        )

        # Rows that saw no key (causal with S_q > S_k) produce zeros and an LSE of -inf.
        seen = l_i > 0
        out = acc / tl.where(seen, l_i, 1.0)[:, None]
        o_ptrs = Out + b * stride_ob + h * stride_oh + rows[:, None] * stride_os + d[None, :]
        tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=rows[:, None] < seqlen_q)
        # m is in the log2 domain; convert back to natural log.
        lse = m_i * 0.6931471805599453 + tl.log(tl.where(seen, l_i, 1.0))
        lse = tl.where(seen, lse, -float("inf"))
        tl.store(LSE + pid_bh * seqlen_q + rows, lse, mask=rows < seqlen_q)

    _CONFIGS = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in [
            (64, 32, 4, 4),
            (64, 64, 4, 3),
            (64, 64, 4, 4),
            (64, 128, 4, 3),
            (128, 32, 4, 4),
            (128, 64, 4, 3),
            (128, 64, 8, 3),
            (128, 128, 8, 3),
        ]
    ]
    _tuned_kernel = triton.autotune(configs=_CONFIGS, key=["SEQ_BUCKET", "HEAD_DIM", "CAUSAL"])(
        _attention_fwd_kernel
    )
    _FIXED_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}

    def _seq_bucket(seqlen_k: int) -> int:
        return min(1 << max(seqlen_k - 1, 1).bit_length(), 16384)

    @torch.library.custom_op("flash_lab::attention_triton", mutates_args=())
    def attention_triton(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, softmax_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen_q, heads, head_dim = q.shape
        seqlen_k, heads_kv = k.shape[1], k.shape[2]
        o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = torch.empty(batch, heads, seqlen_q, dtype=torch.float32, device=q.device)
        args = (
            q,
            k,
            v,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            heads,
            heads // heads_kv,
            seqlen_q,
            seqlen_k,
            softmax_scale * LOG2E,
            _seq_bucket(seqlen_k),
        )
        if os.environ.get("FLASH_LAB_TRITON_AUTOTUNE", "1") == "0":
            grid = (math.ceil(seqlen_q / _FIXED_CONFIG["BLOCK_M"]), batch * heads)
            _attention_fwd_kernel[grid](*args, HEAD_DIM=head_dim, CAUSAL=causal, **_FIXED_CONFIG)
        else:

            def grid(meta):
                return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * heads)

            _tuned_kernel[grid](*args, HEAD_DIM=head_dim, CAUSAL=causal)
        return o, lse

    def autotune_choice(seqlen_k: int, head_dim: int, causal: bool) -> dict | None:
        """The configuration autotuning picked for a key, once it has run."""
        for key, config in _tuned_kernel.cache.items():
            if key[:3] == (_seq_bucket(seqlen_k), head_dim, causal):
                return {
                    **config.kwargs,
                    "num_warps": config.num_warps,
                    "num_stages": config.num_stages,
                }
        return None
