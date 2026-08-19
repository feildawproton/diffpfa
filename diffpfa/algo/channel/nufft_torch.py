import math
from typing import Tuple, Union
import numpy as np
import torch
from scipy.fft import next_fast_len

def kaiser_bessel_kernel_1d(x: torch.Tensor, J: int = 6, beta: float = 13.9086) -> torch.Tensor:
    """
    Evaluates 1D Kaiser-Bessel window kernel at distances x (in grid units).
    Kernel is truncated at |x| > J / 2.
    Uses torch.i0 for zeroth-order modified Bessel function.
    """
    abs_x = torch.abs(x)
    mask = abs_x <= (J / 2.0)

    val = torch.zeros_like(x)
    # sqrt(1 - (2x/J)^2)
    arg = torch.clamp(1.0 - (2.0 * abs_x[mask] / J) ** 2, min=1e-12)
    bessel_arg = beta * torch.sqrt(arg)

    # I0(bessel_arg) / I0(beta)
    val[mask] = torch.i0(bessel_arg) / torch.i0(torch.tensor(beta, dtype=x.dtype, device=x.device))
    return val

def nufft_grid_1d(
    signal: torch.Tensor,        # (N_pts, Batch) complex
    kx: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], # (N_pts,) or factored tuple
    grid_size: int,
    L_x: float,                  # Spatial extent (meters)
    k_center: float,             # Target center of K-space grid
    oversample: float = 1.0,
    J: int = 6,
    beta: float = 13.9086,
    batch_size: int = None
) -> torch.Tensor:
    """
    1D Type-1 NUFFT K-space gridding step (Non-uniform K-space -> Uniform K-space).
    No IFFT is applied. Returns the oversampled K-space grid.
    """
    N_pts, B = signal.shape
    device = signal.device
    real_dtype = torch.float64 if signal.dtype == torch.complex128 else torch.float32
    
    if isinstance(kx, tuple):
        kx_scale, kx_base = kx
        kx_scale = kx_scale.to(real_dtype)
        kx_base = kx_base.to(real_dtype)
    else:
        kx = kx.to(real_dtype)
        if kx.ndim == 1:
            kx = kx.unsqueeze(1).expand(N_pts, B)
     
    M = next_fast_len(int(math.ceil(grid_size * oversample)))
    
    # 1. Calibrate grid scaling
    dK = grid_size / (M * max(L_x, 1e-12))
    
    grid = torch.zeros((M, B), dtype=signal.dtype, device=device)
    
    batch_size = batch_size or B
    
    for b in range(0, B, batch_size):
        b_end = min(b + batch_size, B)
        
        if isinstance(kx, tuple):
            kx_b = kx_scale.unsqueeze(1) * kx_base[b:b_end].unsqueeze(0)
        else:
            kx_b = kx[:, b:b_end]
            
        sig_b = signal[:, b:b_end]
        
        gx_idx_b = ((kx_b - k_center) / dK) + (M / 2.0)
        
        half_J = J / 2.0
        j_offsets = torch.arange(-math.floor(half_J), math.ceil(half_J), device=device, dtype=real_dtype)
        
        col_idx = torch.arange(b_end - b, device=device).unsqueeze(0).expand(N_pts, b_end - b)
        grid_b = torch.zeros((M, b_end - b), dtype=signal.dtype, device=device)
        
        for jx in j_offsets:
            ix = torch.floor(gx_idx_b + jx).to(torch.long)
            mask = (ix >= 0) & (ix < M)
            
            wx = kaiser_bessel_kernel_1d((gx_idx_b - ix.to(real_dtype)), J=J, beta=beta)
            
            flat_idx = ix[mask] * (b_end - b) + col_idx[mask]
            grid_b.view(-1).index_add_(0, flat_idx, sig_b[mask] * wx[mask])
            
        grid[:, b:b_end] = grid_b
        
    return grid
