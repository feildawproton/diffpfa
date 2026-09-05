import torch
import numpy as np
import math

from diffpfa.IFA.channel.czt_torch import czt_1d_torch, czt_resample_kspace_1d
from diffpfa.IFA.channel.nufft_torch import nufft_grid_1d
from diffpfa.IFA.channel.pfa_channel import process_cztnufft, _deskew_rvp
from diffpfa.IFA.PFA import _apply_ifft_and_deconv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing differentiability on {device}...")

# -------------------------------------------------------------
# Test 1: CZT 1D Differentiability & Gradcheck
# -------------------------------------------------------------
print("\n--- Test 1: czt_1d_torch autograd & gradcheck ---")
N = 32
M = 32
x = torch.randn(4, N, dtype=torch.complex128, device=device, requires_grad=True)
k_start = torch.full((4, 1), 60.0, dtype=torch.float64, device=device)
k_step = torch.full((4, 1), 0.1, dtype=torch.float64, device=device)

try:
    out = czt_1d_torch(x, M=M, r_min=-50.0, r_max=50.0, k_step=k_step, k_start=k_start, dim=-1)
    loss = (out.abs() ** 2).sum()
    loss.backward()
    assert x.grad is not None
    print(f"  czt_1d_torch backward SUCCESS. Grad norm: {x.grad.norm().item():.4e}")
    
    # Run gradcheck on czt_1d_torch
    def czt_func(x_in):
        return czt_1d_torch(x_in, M=M, r_min=-50.0, r_max=50.0, k_step=k_step, k_start=k_start, dim=-1)
    
    test_passed = torch.autograd.gradcheck(czt_func, (x,), eps=1e-6, atol=1e-4)
    print(f"  czt_1d_torch gradcheck: {test_passed}")
except Exception as e:
    print(f"  czt_1d_torch FAILED: {e}")

# -------------------------------------------------------------
# Test 2: czt_resample_kspace_1d Differentiability
# -------------------------------------------------------------
print("\n--- Test 2: czt_resample_kspace_1d autograd ---")
x = torch.randn(4, N, dtype=torch.complex128, device=device, requires_grad=True)
try:
    resampled = czt_resample_kspace_1d(
        x,
        k_start=k_start,
        k_step=k_step,
        M_out=M,
        k_out_start=59.0,
        k_out_step=0.1,
        spatial_extent=100.0,
        batch_size=2
    )
    loss = (resampled.abs() ** 2).sum()
    loss.backward()
    assert x.grad is not None
    print(f"  czt_resample_kspace_1d backward SUCCESS. Grad norm: {x.grad.norm().item():.4e}")
except Exception as e:
    print(f"  czt_resample_kspace_1d FAILED: {e}")

# -------------------------------------------------------------
# Test 3: nufft_grid_1d Differentiability
# -------------------------------------------------------------
print("\n--- Test 3: nufft_grid_1d autograd ---")
N_pts, B = 16, 32
sig = torch.randn(N_pts, B, dtype=torch.complex64, device=device, requires_grad=True)
cot_theta = torch.linspace(-0.1, 0.1, N_pts, device=device)
Kr_cart = torch.linspace(59.0, 62.0, B, device=device)

try:
    grid = nufft_grid_1d(
        signal=sig,
        kx=(cot_theta, Kr_cart),
        grid_size=32,
        L_x=100.0,
        k_center=0.0,
        batch_size=16
    )
    loss = (grid.abs() ** 2).sum()
    loss.backward()
    assert sig.grad is not None
    print(f"  nufft_grid_1d backward SUCCESS. Grad norm: {sig.grad.norm().item():.4e}")
except Exception as e:
    print(f"  nufft_grid_1d FAILED: {e}")

# -------------------------------------------------------------
# Test 4: _apply_ifft_and_deconv In-Place Mutation Test
# -------------------------------------------------------------
print("\n--- Test 4: _apply_ifft_and_deconv autograd ---")
grid = torch.randn(32, 32, dtype=torch.complex64, device=device, requires_grad=True)
try:
    img = _apply_ifft_and_deconv(grid, M_u=32, M_r=32, device=device)
    loss = (img.abs() ** 2).sum()
    loss.backward()
    assert grid.grad is not None
    print(f"  _apply_ifft_and_deconv backward SUCCESS. Grad norm: {grid.grad.norm().item():.4e}")
except Exception as e:
    print(f"  _apply_ifft_and_deconv FAILED with in-place error: {e}")

# -------------------------------------------------------------
# Test 5: Full End-to-End process_cztnufft autograd
# -------------------------------------------------------------
print("\n--- Test 5: End-to-end process_cztnufft autograd ---")
sig_full = torch.randn(16, 32, dtype=torch.complex64, device=device, requires_grad=True)
pvp_dummy = {}
Ku_dummy = torch.randn(16, 32, dtype=torch.float64, device=device)
Kr_dummy = torch.linspace(59.0, 62.0, 32, dtype=torch.float64, device=device).unsqueeze(0).expand(16, 32)

try:
    grid_out = process_cztnufft(
        signal=sig_full,
        fxc=9.6e9,
        pvp=pvp_dummy,
        Ku=Ku_dummy,
        Kr=Kr_dummy,
        N_u=32,
        N_r=32,
        L_u=100.0,
        L_r=100.0,
        k_ctr_u=0.0,
        k_ctr_r=60.5,
        batch_size=16,
        device=device
    )
    img_out = _apply_ifft_and_deconv(grid_out, M_u=32, M_r=32, device=device)
    target = torch.randn_like(img_out)
    pixel_loss = torch.nn.functional.mse_loss(img_out.real, target.real) + torch.nn.functional.mse_loss(img_out.imag, target.imag)
    pixel_loss.backward()
    assert sig_full.grad is not None
    print(f"  Full pipeline pixel_loss -> sig_full backward SUCCESS!")
    print(f"  Grad norm: {sig_full.grad.norm().item():.4e}")
    print(f"  Gradient min: {sig_full.grad.abs().min().item():.4e}, max: {sig_full.grad.abs().max().item():.4e}")
except Exception as e:
    print(f"  Full pipeline FAILED: {e}")
