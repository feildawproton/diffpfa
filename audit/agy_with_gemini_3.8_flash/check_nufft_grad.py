import torch
from diffpfa.IFA.channel.nufft_torch import nufft_grid_1d

device = "cuda" if torch.cuda.is_available() else "cpu"
N_pts, B = 16, 32
sig = torch.randn(N_pts, B, dtype=torch.complex64, device=device, requires_grad=True)

# Choose realistic angles that fall inside grid:
# Grid: grid_size = 32, L_x = 100.0, dK = 1/100 = 0.01 cycles/m.
# Max kx supported by grid: (32/2) * 0.01 = 0.16 cycles/m.
# If Kr ~ 60, cot_theta should be <= 0.16 / 60 ~ 0.0026 rad!
cot_theta = torch.linspace(-0.001, 0.001, N_pts, device=device)
Kr_cart = torch.linspace(59.0, 61.0, B, device=device)

grid = nufft_grid_1d(
    signal=sig,
    kx=(cot_theta, Kr_cart),
    grid_size=32,
    L_x=100.0,
    k_center=0.0,
    batch_size=16
)

print("Grid norm:", grid.norm().item())
loss = (grid.abs() ** 2).sum()
loss.backward()

print("sig.grad is None?", sig.grad is None)
if sig.grad is not None:
    print("sig.grad norm:", sig.grad.norm().item())
    print("Non-zero grad count:", (sig.grad != 0).sum().item(), "out of", sig.numel())
