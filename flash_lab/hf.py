"""Hugging Face transformers integration: attn_implementation="flash_lab".

    import flash_lab.hf
    flash_lab.hf.register()
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="flash_lab")

Prompt processing (several query tokens) runs flash_lab.attention with causal masking, and each
generated token (one query token) runs flash_lab.decode directly on the KV cache tensors. With
cache_implementation="static" the cache of each layer is a fixed [B, H_kv, max_len, D] buffer
that the decode kernels read in place, and generate() compiles the decode step with
torch.compile, which the ops support through their fake (shape-only) kernels.

transformers builds no attention mask for an attention implementation it does not know, so the
number of valid cache rows comes from position_ids, and batches must be unpadded: every sequence
starts at position 0 and all have the same length. Prefill checks this.
"""

from functools import partial

import torch

from flash_lab.ops import attention, decode

NAME = "flash_lab"


def attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    *,
    prefill_impl: str = "auto",
    decode_impl: str = "auto",
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """transformers' attention-function signature: query is [B, H, S_q, D], key and value are
    [B, H_kv, S_k, D] (the whole buffer for a static cache), and the output is [B, S_q, H, D]."""
    if dropout:
        raise ValueError("flash_lab attention has no dropout; call model.eval() first")
    if kwargs.get("sliding_window") is not None:
        raise NotImplementedError("flash_lab attention has no sliding-window masking")
    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        raise ValueError("flash_lab attention needs position_ids to find the valid cache rows")

    batch, _, seqlen_q, _ = query.shape
    # [B, S, H, D] views of the [B, H, S, D] tensors; the kernels read through the strides.
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    last = position_ids[:, -1]
    if seqlen_q == 1:
        # No host sync on this path: it is the step generate() compiles and replays.
        seq_lens = (last + 1).to(torch.int32).expand(batch).contiguous()
        out = decode(q, k, v, seq_lens, softmax_scale=scaling, impl=decode_impl)
        return out, None

    ends = set(last.tolist())
    if len(ends) != 1:
        raise ValueError("flash_lab attention supports unpadded batches only")
    seqlen_k = ends.pop() + 1
    if not seqlen_q <= seqlen_k <= k.shape[1]:
        raise ValueError(
            f"position_ids end at {seqlen_k - 1}, which does not fit {seqlen_q} new tokens "
            f"and a cache of {k.shape[1]} rows"
        )
    out = attention(
        q,
        k[:, :seqlen_k],
        v[:, :seqlen_k],
        causal=getattr(module, "is_causal", True),
        softmax_scale=scaling,
        impl=prefill_impl,
    )
    return out, None


def register(name: str = NAME, prefill_impl: str = "auto", decode_impl: str = "auto") -> str:
    """Registers (or re-registers) attention_forward with transformers under `name`, with the
    given kernel choices, and returns the name to pass as attn_implementation."""
    from transformers import AttentionInterface

    AttentionInterface.register(
        name, partial(attention_forward, prefill_impl=prefill_impl, decode_impl=decode_impl)
    )
    return name
