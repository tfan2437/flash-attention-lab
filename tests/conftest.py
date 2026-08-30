import os

import pytest
import torch

# One fixed Triton configuration unless a test asks for autotuning; otherwise every new shape
# would compile all candidates (tests/test_triton.py covers the autotuned path).
os.environ.setdefault("FLASH_LAB_TRITON_AUTOTUNE", "0")

from flash_lab import ops  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(autouse=True, scope="session")
def _no_tf32():
    # fp32 baselines must really be fp32: TF32 matmuls would loosen the reference comparisons.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@pytest.fixture(scope="session")
def flash_ops():
    """torch.ops.flash_lab, failing loudly if a GPU is present but the extension did not load."""
    if not ops.HAS_EXTENSION:
        pytest.fail(f"CUDA is available but flash_lab._C did not import: {ops.EXTENSION_ERROR}")
    return torch.ops.flash_lab
