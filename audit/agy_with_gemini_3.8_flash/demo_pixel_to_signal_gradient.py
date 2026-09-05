import torch
import numpy as np
import math

from diffpfa.IFA.channel.pfa_channel import process_cztnufft
from diffpfa.IFA.PFA import _apply_ifft_and_deconv
from diffpfa.IFA.kspace import compute_kspace

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Pixel-to-Signal Gradient Demo on {device}...")

# 1. Setup synthetic radar signal with known geometry
N_pulses = 32
N_samples = 64
N_u = 32
N_r = 32
L_u = 100.0
L_r = 100.0

# Create input signal tensor requiring gradients
torch.manual_seed(42)
sig_raw = torch.randn(N_pulses, N_samples, dtype=torch.complex64, device=device, requires_grad=True)

# Generate realistic K-space bounds
fxc = 9.6e9
k_ctr_u = 0.0
k_ctr_r = 64.0
cot_theta = torch.linspace(-0.002, 0.002, N_pulses, device=device, dtype=torch.float64)
Kr_radial = torch.linspace(63.0, 65.0, N_samples, device=device, dtype=torch.float64).unsqueeze(0).expand(N_pulses, N_samples)
Ku_radial = cot_theta.unsqueeze(1) * Kr_radial

pvp_dummy = {}

def forward_pfa(sig):
    grid_2d = process_cztnufft(
        signal=sig,
        fxc=fxc,
        pvp=pvp_dummy,
        Ku=Ku_radial,
        Kr=Kr_radial,
        N_u=N_u,
        N_r=N_r,
        L_u=L_u,
        L_r=L_r,
        k_ctr_u=k_ctr_u,
        k_ctr_r=k_ctr_r,
        batch_size=N_r,
        device=device
    )
    img = _apply_ifft_and_deconv(grid_2d, N_u, N_r, device)
    # Transpose to match SICD row (range) and col (azimuth)
    img = img.T
    return img

# 2. Forward pass to form initial image
img_init = forward_pfa(sig_raw)
print(f"Initial formed image shape: {img_init.shape}, mean magnitude: {img_init.abs().mean().item():.4e}")

# 3. Define target modification on SICD pixels (e.g. inject an artificial target at pixel (16, 16))
img_target = img_init.detach().clone()
img_target[16, 16] += 10.0 + 5.0j  # Desired pixel edit

# 4. Compute loss on SICD pixels
loss = torch.sum(torch.abs(img_init - img_target) ** 2)
print(f"Initial Pixel Loss: {loss.item():.4e}")

# 5. Backpropagate to raw channel signal
loss.backward()

assert sig_raw.grad is not None
print("Gradient backpropagation SUCCESS!")
print(f"sig_raw.grad shape: {sig_raw.grad.shape}")
print(f"sig_raw.grad norm: {sig_raw.grad.norm().item():.4e}")
print(f"Fraction of non-zero gradient elements: {(sig_raw.grad != 0).float().mean().item() * 100:.1f}%")

# 6. Verify gradient descent optimization step
optimizer = torch.optim.Adam([sig_raw], lr=0.01)
optimizer.step()

# Recompute loss after 1 step
img_after = forward_pfa(sig_raw)
loss_after = torch.sum(torch.abs(img_after - img_target) ** 2)
print(f"Pixel Loss after 1 gradient step: {loss_after.item():.4e} (Reduced: {loss.item() > loss_after.item()})")
