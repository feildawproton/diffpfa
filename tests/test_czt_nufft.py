import torch
import numpy as np
import pytest
from diffpfa.IFA.channel.czt_torch import czt_1d_torch, czt_resample_kspace_1d
from diffpfa.IFA.channel.nufft_torch import kaiser_bessel_kernel_1d, nufft_grid_1d

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_czt_1d_vs_direct_sum():
    N, M = 64, 100
    r_min, r_max = -40.0, 40.0
    k_start_val = 1e8 / 3e8
    k_step_val = 1e6 / 3e8

    torch.manual_seed(42)
    x = torch.randn(N, dtype=torch.complex128, device=device)
    k_start = torch.tensor(k_start_val, dtype=torch.float64, device=device)
    k_step = torch.tensor(k_step_val, dtype=torch.float64, device=device)

    czt_out = czt_1d_torch(x, M=M, r_min=r_min, r_max=r_max, k_step=k_step, k_start=k_start)

    # Direct analytical summation
    n_idx = torch.arange(N, dtype=torch.float64, device=device)
    m_idx = torch.arange(M, dtype=torch.float64, device=device)
    dr = (r_max - r_min) / max(M - 1, 1)
    r_m = r_min + m_idx * dr
    k_n = k_start + n_idx * k_step
    dft_kernel = torch.exp(1j * 2.0 * np.pi * torch.outer(r_m, k_n))
    direct_out = torch.matmul(dft_kernel, x)

    rel_err = (torch.norm(czt_out - direct_out) / torch.norm(direct_out)).item()
    assert rel_err < 1e-12, f"CZT relative error {rel_err} exceeds 1e-12 threshold"

def test_kaiser_bessel_kernel_properties():
    # Peak at center x = 0 should be 1.0
    x_zero = torch.tensor([0.0], dtype=torch.float64, device=device)
    w_zero = kaiser_bessel_kernel_1d(x_zero, J=6, beta=13.9086)
    assert np.isclose(w_zero.item(), 1.0, atol=1e-6)

    # Truncation outside [-J/2, J/2] should be 0.0
    x_outside = torch.tensor([3.5, -4.0], dtype=torch.float64, device=device)
    w_outside = kaiser_bessel_kernel_1d(x_outside, J=6, beta=13.9086)
    assert torch.all(w_outside == 0.0)
