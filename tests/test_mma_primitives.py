"""ldmatrix + mma.sync fragment handling on a single tile, before it is buried in a kernel."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("k", [16, 32, 64])
def test_tile_times_transposed_tile(flash_ops, k):
    # B given as [16, K] rows, like K in Q K^T.
    a = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    c = flash_ops.mma_tile_test(a, b, False)
    torch.testing.assert_close(c, a.float() @ b.float().T, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("k", [16, 32, 64])
def test_tile_times_tile_through_ldmatrix_trans(flash_ops, k):
    # B given as [K, 16] rows, like V in P V.
    a = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, 16, device="cuda", dtype=torch.bfloat16)
    c = flash_ops.mma_tile_test(a, b, True)
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-5, atol=1e-5)


def test_fragment_positions_with_an_identity(flash_ops):
    # A = I picks rows of B, so any lane or register mix-up shows as a permuted result.
    a = torch.eye(16, device="cuda", dtype=torch.bfloat16)
    b = torch.arange(256, device="cuda", dtype=torch.float32).view(16, 16).to(torch.bfloat16)
    torch.testing.assert_close(flash_ops.mma_tile_test(a, b, True), b.float())
    torch.testing.assert_close(flash_ops.mma_tile_test(a, b, False), b.float().T)
