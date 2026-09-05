import torch
import math
from diffpfa.IFA.channel.nufft_torch import nufft_grid_1d
from diffpfa.IFA.PFA import _apply_ifft_and_deconv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running gradcheck on {device}...")

# Gradcheck on nufft_grid_1d
N_pts, B = 8, 8
sig = torch.randn(N_pts, B, dtype=torch.complex128, device=device, requires_grad=True)
cot_theta = torch.linspace(-0.0005, 0.0005, N_pts, device=device, dtype=torch.float64)
Kr_cart = torch.linspace(59.5, 60.5, B, device=device, dtype=torch.float64)

def nufft_wrapper(s):
    return nufft_grid_1d(
        signal=s,
        kx=(cot_theta, Kr_cart),
        grid_size=16,
        L_x=100.0,
        k_center=0.0,
        batch_size=8
    )

passed_nufft = torch.autograd.gradcheck(nufft_wrapper, (sig,), eps=1e-6, atol=1e-4)
print(f"nufft_grid_1d gradcheck passed: {passed_nufft}")

# Gradcheck on _apply_ifft_and_deconv
grid = torch.randn(16, 16, dtype=torch.complex128, device=device, requires_grad=True)
def deconv_wrapper(g):
    return _apply_ifft_and_deconv(g.clone(), M_u=16, M_r=16, device=device)

passed_deconv = torch.autograd.gradcheck(deconv_wrapper, (grid,), eps=1e-6, atol=1e-4)
print(f"_apply_ifft_and_deconv gradcheck passed: {passed_deconv}")
