import torch
import math
import matplotlib.pyplot as plt

def kaiser_bessel_kernel_1d(x: torch.Tensor, J: int = 6, beta: float = 13.9086) -> torch.Tensor:
    abs_x = torch.abs(x)
    mask = abs_x <= (J / 2.0)
    val = torch.zeros_like(x)
    arg = torch.clamp(1.0 - (2.0 * abs_x[mask] / J) ** 2, min=1e-12)
    bessel_arg = beta * torch.sqrt(arg)
    val[mask] = torch.i0(bessel_arg) / torch.i0(torch.tensor(beta, dtype=x.dtype, device=x.device))
    return val

J = 6
beta = 13.9086
M = 512
k = torch.arange(M, dtype=torch.float64) - M / 2.0
# The kernel in K-space
wk = kaiser_bessel_kernel_1d(k, J=J, beta=beta)
# Compute its FT
Wk_ft = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(wk)))

# The deconv functions used in code:
x_coords = (torch.arange(M, dtype=torch.float64) - M / 2.0) / M

# from nufft_1d_type1_torch
arg = torch.clamp(1.0 - (2.0 * x_coords) ** 2, min=1e-12)
deconv_1d = torch.i0(beta * torch.sqrt(arg)) / torch.i0(torch.tensor(beta))

# from nufft_2d_type1_torch / pfa_engine
deconv_2d = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=torch.float64)**2 - (math.pi * J * x_coords)**2, min=1e-12))) / torch.i0(torch.tensor(beta, dtype=torch.float64))

# True analytic FT of Kaiser-Bessel
z = torch.sqrt(torch.complex(torch.tensor(beta)**2 - (math.pi * J * x_coords)**2, torch.zeros_like(x_coords)))
true_deconv = torch.real(torch.sinh(z) / z)

print("Midpoint Wk_ft: ", torch.abs(Wk_ft[M//2]).item())
Wk_ft_normalized = torch.abs(Wk_ft) / torch.abs(Wk_ft[M//2])
true_deconv_normalized = true_deconv / true_deconv[M//2]

print("Diff between normalized numerical FT and true analytic FT (sinh(z)/z):", torch.max(torch.abs(Wk_ft_normalized - true_deconv_normalized)).item())

print("Diff between normalized numerical FT and nufft_1d deconv:", torch.max(torch.abs(Wk_ft_normalized - (deconv_1d / deconv_1d[M//2]))).item())
print("Diff between normalized numerical FT and nufft_2d/pfa deconv:", torch.max(torch.abs(Wk_ft_normalized - (deconv_2d / deconv_2d[M//2]))).item())
