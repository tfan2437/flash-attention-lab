"""Checks that the extension built for this GPU and that op registration works."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_extension_registers_ops(flash_ops):
    assert hasattr(flash_ops, "smoke_axpy")


@pytest.mark.parametrize("n", [1, 1000, 1 << 20])
def test_smoke_kernel(flash_ops, n):
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    torch.testing.assert_close(flash_ops.smoke_axpy(x, y, 2.5), 2.5 * x + y)
