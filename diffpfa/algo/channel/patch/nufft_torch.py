import math
from typing import Tuple
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
    kx: torch.Tensor,            # (N_pts,) or (N_pts, Batch) in cycles/meter
    grid_size: int,
    L_x: float,                  # Spatial extent (meters)
    k_center: float,             # Target center of K-space grid
    oversample: float = 2.0,
    J: int = 6,
    beta: float = 13.9086
) -> torch.Tensor:
    """
    1D Type-1 NUFFT K-space gridding step (Non-uniform K-space -> Uniform K-space).
    No IFFT is applied. Returns the oversampled K-space grid.
    """
    N_pts, B = signal.shape
    device = signal.device
    real_dtype = torch.float64 if signal.dtype == torch.complex128 else torch.float32
    
    kx = kx.to(real_dtype)
     
    M = next_fast_len(int(math.ceil(grid_size * oversample)))
    
    # 1. Calibrate grid scaling
    dK = grid_size / (M * max(L_x, 1e-12))
    
    if kx.ndim == 1:
        kx = kx.unsqueeze(1).expand(N_pts, B)
    
    # Normalizing coordinates
    gx_idx = ((kx - k_center) / dK) + (M / 2.0)
    
    grid = torch.zeros((M, B), dtype=signal.dtype, device=device)
    
    half_J = J / 2.0
    j_offsets = torch.arange(-math.floor(half_J), math.ceil(half_J), device=device, dtype=real_dtype)
    
    col_idx = torch.arange(B, device=device).unsqueeze(0).expand(N_pts, B)
    
    for jx in j_offsets:
        ix = torch.floor(gx_idx + jx).to(torch.long)
        mask = (ix >= 0) & (ix < M)
        
        wx = kaiser_bessel_kernel_1d((gx_idx - ix.to(real_dtype)), J=J, beta=beta)
        
        flat_idx = ix[mask] * B + col_idx[mask]
        grid.view(-1).index_add_(0, flat_idx, signal[mask] * wx[mask])
        
    return grid


def nufft_1d_type1_torch(
    signal: torch.Tensor,        # (N_pts, Batch) complex
    kx: torch.Tensor,            # (N_pts,) in cycles/meter
    grid_size: int,
    x_min: float,
    x_max: float,
    oversample: float = 2.0,
    J: int = 6
) -> torch.Tensor:
    """
    1D Type-1 NUFFT for non-uniform kx -> uniform x.
    """
    N_pts, B = signal.shape
    device = signal.device
    real_dtype = torch.float64
    
    L_x = x_max - x_min
    from scipy.fft import next_fast_len
    M = next_fast_len(int(math.ceil(grid_size * oversample)))
    
    # 1. Calibrate grid scaling
    dK = grid_size / (M * max(L_x, 1e-12))
    
    k_min = kx.min().item()
    k_max = kx.max().item()
    k_center = (k_min + k_max) / 2.0
    
    # Normalizing coordinates
    gx_idx = ((kx - k_center) / dK) + (M // 2)
    
    grid = torch.zeros((M, B), dtype=signal.dtype, device=device)
    
    j_offsets = torch.arange(-J//2 + 1, J//2 + 1, device=device, dtype=real_dtype)
    beta = 13.9086
    
    for jx in j_offsets:
        ix = torch.floor(gx_idx + jx).to(torch.long)
        mask = (ix >= 0) & (ix < M)
        
        wx = kaiser_bessel_kernel_1d((gx_idx - ix.to(real_dtype)), J=J, beta=beta)
        
        ix_valid = ix[mask]
        wx_valid = wx[mask]
        
        # signal has shape (N_pts, B), wx_valid is (N_pts,)
        weighted_sig = signal[mask] * wx_valid.unsqueeze(1)
        
        # Accumulate into grid using index_add_ (much cleaner than index_put_ with accumulate=True)
        grid.index_add_(0, ix_valid, weighted_sig)
        
    # Deconvolution weights
    x_idx = torch.arange(M, dtype=real_dtype, device=device)
    alpha = (x_idx - M // 2) / M
    arg = torch.clamp(1.0 - (2.0 * alpha) ** 2, min=1e-12)
    deconv_w = torch.i0(beta * torch.sqrt(arg)) / torch.i0(torch.tensor(beta))
    # Clip to avoid division by zero
    deconv_w = torch.clamp(deconv_w, min=1e-6)
    
    # IFFT
    grid = torch.fft.ifftshift(grid, dim=0)
    grid = torch.fft.ifft(grid, dim=0)
    img_deconv = torch.fft.ifftshift(grid, dim=0)
    del grid
    img_deconv = img_deconv / deconv_w.unsqueeze(1)
    
    # Phase shift to center at x_c
    x_c = (x_min + x_max) / 2.0
    x_coords = torch.linspace(x_min, x_max, grid_size, device=device, dtype=real_dtype)
    shift_phase = torch.exp(torch.complex(torch.zeros_like(x_coords), -2.0 * math.pi * k_center * x_coords))
    
    start_x = (M - grid_size) // 2
    img_cropped = img_deconv[start_x : start_x + grid_size, :]
    
    return img_cropped * shift_phase.unsqueeze(1)


def nufft_2d_type1_torch(
    signal: torch.Tensor,        # (N_pts,) or (N_pulses, N_samples) complex
    ku: torch.Tensor,            # (N_pts,) in cycles/meter
    kr: torch.Tensor,            # (N_pts,) in cycles/meter
    grid_size_u: int,
    grid_size_r: int,
    u_min: float,
    u_max: float,
    r_min: float,
    r_max: float,
    oversample: float = 1.5,
    J: int = 6,
    beta: float = 13.9086,
    batch_size_pts: int = 1_000_000
) -> torch.Tensor:
    """
    2D Non-Uniform FFT Type 1 (Non-uniform K-space -> Uniform Image Space) using PyTorch.

    Args:
        signal: Complex non-uniform signal tensor.
        ku: Spatial frequencies along cross-range u (cycles/meter).
        kr: Spatial frequencies along range r (cycles/meter).
        grid_size_u: Target output image grid size along u (lines).
        grid_size_r: Target output image grid size along r (samples).
        u_min, u_max: Target spatial extent in meters along u.
        r_min, r_max: Target spatial extent in meters along r.
        oversample: Oversampling factor for Cartesian grid (default 1.5).
        J: Kernel footprint width in grid units (default 6).
        beta: Kaiser-Bessel shape parameter.
        batch_size_pts: Number of points to process simultaneously (default 1,000,000).

    Returns:
        Complex 2D image tensor of shape (grid_size_u, grid_size_r).
    """
    device = signal.device
    complex_dtype = signal.dtype
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32

    # Flatten inputs to 1D vectors
    sig_flat = signal.reshape(-1).to(complex_dtype)
    ku_flat = ku.reshape(-1).to(real_dtype)
    kr_flat = kr.reshape(-1).to(real_dtype)
    N_pts = sig_flat.shape[0]

    # Calculate oversampled grid dimensions
    from scipy.fft import next_fast_len
    M_u = next_fast_len(int(math.ceil(grid_size_u * oversample)))
    M_r = next_fast_len(int(math.ceil(grid_size_r * oversample)))

    # K-space extent in cycles/meter
    # For spatial domain extent L_u = u_max - u_min, dK_u = 1 / L_u
    du_meter = (u_max - u_min) / max(grid_size_u, 1)
    dr_meter = (r_max - r_min) / max(grid_size_r, 1)

    L_u = grid_size_u * du_meter
    L_r = grid_size_r * dr_meter

    dK_u = grid_size_u / (M_u * max(L_u, 1e-12))
    dK_r = grid_size_r / (M_r * max(L_r, 1e-12))

    # Convert non-uniform (ku, kr) coordinates to grid index units [0, M_u), [0, M_r)
    # Center K-space around 0. Since Ku/Kr are already globally centered by the caller,
    # we just use 0.0 unless the caller specifically passes a center.
    k_ctr_u = 0.0
    k_ctr_r = 0.0

    # 2D Cartesian K-space grid buffer
    grid = torch.zeros((M_u, M_r), dtype=complex_dtype, device=device)

    # Gridding loop over J x J neighbor offsets
    half_J = J / 2.0
    j_offsets = torch.arange(-math.floor(half_J), math.ceil(half_J), device=device, dtype=real_dtype)

    # Process points in batches to prevent GPU OOM
    for start_idx in range(0, N_pts, batch_size_pts):
        end_idx = min(start_idx + batch_size_pts, N_pts)
        
        ku_b = ku_flat[start_idx:end_idx]
        kr_b = kr_flat[start_idx:end_idx]
        sig_b = sig_flat[start_idx:end_idx]

        gu_idx = (ku_b - k_ctr_u) / dK_u + (M_u / 2.0)
        gr_idx = (kr_b - k_ctr_r) / dK_r + (M_r / 2.0)

        for ju in j_offsets:
            iu = torch.floor(gu_idx + ju).to(torch.long)
            mask_u = (iu >= 0) & (iu < M_u)
            wu = kaiser_bessel_kernel_1d((gu_idx - iu.to(real_dtype)), J=J, beta=beta)

            for jr in j_offsets:
                ir = torch.floor(gr_idx + jr).to(torch.long)
                mask_r = (ir >= 0) & (ir < M_r)
                mask = mask_u & mask_r

                wr = kaiser_bessel_kernel_1d((gr_idx - ir.to(real_dtype)), J=J, beta=beta)
                w_2d = (wu * wr).to(complex_dtype)

                # Scatter add weighted signal values into Cartesian grid (flattened for index_add_)
                flat_idx = iu[mask] * M_r + ir[mask]
                grid.view(-1).index_add_(0, flat_idx, sig_b[mask] * w_2d[mask])

    # 2D Inverse FFT to image domain
    # Shift center before IFFT
    grid_shifted = torch.fft.ifftshift(grid)
    del grid  # Free memory
    
    img_oversampled = torch.fft.ifft2(grid_shifted)
    del grid_shifted  # Free memory
    
    img_oversampled.mul_(M_u * M_r)

    img_shifted = torch.fft.fftshift(img_oversampled)
    del img_oversampled  # Free memory

    # Deconvolution to remove Kaiser-Bessel kernel roll-off
    grid_u_coords = (torch.arange(M_u, device=device, dtype=real_dtype) - M_u / 2.0) / M_u
    grid_r_coords = (torch.arange(M_r, device=device, dtype=real_dtype) - M_r / 2.0) / M_r

    deconv_u = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=device)**2 - (pi_val := math.pi * J * grid_u_coords)**2, min=1e-12))) / torch.i0(torch.tensor(beta, dtype=real_dtype, device=device))
    deconv_r = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=device)**2 - (pi_val_r := math.pi * J * grid_r_coords)**2, min=1e-12))) / torch.i0(torch.tensor(beta, dtype=real_dtype, device=device))

    deconv_2d = torch.outer(deconv_u, deconv_r).to(complex_dtype)
    img_deconv = img_shifted / (deconv_2d + 1e-12)
    del img_shifted  # Free memory

    # Crop to target image dimensions (grid_size_u, grid_size_r)
    start_u = (M_u - grid_size_u) // 2
    start_r = (M_r - grid_size_r) // 2

    img_cropped = img_deconv[start_u : start_u + grid_size_u, start_r : start_r + grid_size_r]
    return img_cropped
