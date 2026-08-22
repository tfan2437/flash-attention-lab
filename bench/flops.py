"""FLOP and byte models, and per-GPU peak numbers for roofline placement."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ceilings:
    fp32_tflops: float
    tensor_tflops: float  # dense bf16/fp16 tensor-core peak, no sparsity
    hbm_gbps: float
    source: str


# From NVIDIA datasheets. Keys are torch.cuda.get_device_name() strings.
CEILINGS = {
    "NVIDIA H100 80GB HBM3": Ceilings(
        67.0, 989.4, 3350.0, "https://www.nvidia.com/en-us/data-center/h100/ (H100 SXM)"
    ),
    "NVIDIA H100 PCIe": Ceilings(
        51.2, 756.0, 2000.0, "https://www.nvidia.com/en-us/data-center/h100/ (H100 PCIe)"
    ),
    "NVIDIA H200": Ceilings(
        67.0, 989.4, 4800.0, "https://www.nvidia.com/en-us/data-center/h200/ (H200 SXM)"
    ),
    "NVIDIA A100-SXM4-80GB": Ceilings(
        19.5, 312.0, 2039.0, "https://www.nvidia.com/en-us/data-center/a100/ (A100 SXM 80GB)"
    ),
}


def attention_flops(
    batch: int, heads: int, seqlen_q: int, seqlen_k: int, head_dim: int, causal: bool
) -> float:
    """Matmul FLOPs of one forward pass: 2 * S_q * S_k * D for QK^T and again for PV, per head.

    Causal counts half of the full product, the convention of the FlashAttention-2 benchmarks.
    """
    flops = 4.0 * batch * heads * seqlen_q * seqlen_k * head_dim
    return flops / 2 if causal else flops


def prefill_min_bytes(
    batch: int, heads: int, heads_kv: int, seqlen_q: int, seqlen_k: int, head_dim: int, elem: int
) -> float:
    """HBM traffic if Q, K, V are read once and O is written once."""
    q_and_o = 2.0 * batch * seqlen_q * heads * head_dim
    k_and_v = 2.0 * batch * seqlen_k * heads_kv * head_dim
    return (q_and_o + k_and_v) * elem


def decode_bytes(
    batch: int, heads: int, heads_kv: int, context: int, head_dim: int, elem: int
) -> float:
    """HBM traffic of one decode step: every cached K and V row read once, q read, o written."""
    k_and_v = 2.0 * batch * heads_kv * context * head_dim
    q_and_o = 2.0 * batch * heads * head_dim
    return (k_and_v + q_and_o) * elem
