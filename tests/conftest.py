import pytest
import torch

from flash_lab import ops


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


@pytest.fixture(scope="session")
def flash_ops():
    """torch.ops.flash_lab, failing loudly if a GPU is present but the extension did not load."""
    if not ops.HAS_EXTENSION:
        pytest.fail(f"CUDA is available but flash_lab._C did not import: {ops.EXTENSION_ERROR}")
    return torch.ops.flash_lab
