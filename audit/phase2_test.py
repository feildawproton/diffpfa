import torch
import numpy as np

# Simulate two disjoint frequency bands
N = 1000
x = torch.linspace(-10, 10, N)
# Band 1: freq from 10 to 20
# Band 2: freq from 20 to 30
# In K-space
k = torch.fft.fftfreq(N, d=(x[1]-x[0]).item())
k = torch.fft.fftshift(k)

# Ideal point target at x=0
target_x = 0.0

# Band 1 K-space (rect from 10 to 20)
mask1 = (k >= 10) & (k < 20)
K1 = torch.zeros_like(k, dtype=torch.complex64)
K1[mask1] = torch.exp(-1j * 2 * np.pi * k[mask1] * target_x)

# Band 2 K-space (rect from 20 to 30) with a constant phase offset
phase_offset = np.pi / 3.0
mask2 = (k >= 20) & (k < 30)
K2 = torch.zeros_like(k, dtype=torch.complex64)
K2[mask2] = torch.exp(-1j * 2 * np.pi * k[mask2] * target_x) * np.exp(1j * phase_offset)

# 1. Inner product in K-space
inner_prod_k = torch.sum(K1 * torch.conj(K2))
print(f"K-space inner product: {inner_prod_k.item()}")

# 2. Inner product in Image space
I1 = torch.fft.ifft(torch.fft.ifftshift(K1))
I2 = torch.fft.ifft(torch.fft.ifftshift(K2))
inner_prod_i = torch.sum(I1 * torch.conj(I2))
print(f"Image-space inner product: {inner_prod_i.item()}")

# Phase calculated:
est_phase = torch.angle(inner_prod_k)
print(f"True phase offset: {phase_offset}")
print(f"Estimated phase offset (via inner product): {est_phase.item()}")
