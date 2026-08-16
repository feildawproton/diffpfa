import math
import numpy as np
import torch


def czt_1d_torch(
    x: torch.Tensor,
    M: int,
    r_min: float,
    r_max: float,
    k_step: torch.Tensor,
    k_start: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Computes 1D Chirp Z-Transform (CZT) along a specified dimension in PyTorch.
    Evaluates sum_n x[n] * exp(j * 2pi * r_m * (k_start + n * k_step))
    where r_m linearly spaces from r_min to r_max with M points.

    Args:
        x: Input tensor (real or complex).
        M: Number of output spatial bins.
        r_min: Minimum evaluation spatial coordinate (e.g., meters).
        r_max: Maximum evaluation spatial coordinate (e.g., meters).
        k_step: Spatial frequency step per sample (cycles/unit). Tensor broadcastable to x.
        k_start: Starting spatial frequency (cycles/unit). Tensor broadcastable to x.
        dim: Dimension along which to compute CZT (default -1).

    Returns:
        Complex tensor with size M along dimension `dim`.
    """
    if dim < 0:
        dim += x.ndim

    N = x.shape[dim]
    dr = (r_max - r_min) / max(M - 1, 1)

    device = x.device
    real_dtype = torch.float64 if x.dtype in (torch.float64, torch.complex128) else torch.float32
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    x_cplx = x.to(complex_dtype)

    n = torch.arange(N, dtype=real_dtype, device=device)
    m = torch.arange(M, dtype=real_dtype, device=device)

    # Reshape n and m to align with target dimension `dim`
    shape_n = [1] * x.ndim
    shape_n[dim] = N
    n_exp = n.view(*shape_n)

    shape_m = [1] * x.ndim
    shape_m[dim] = M
    m_exp = m.view(*shape_m)

    # Expand k_start and k_step to broadcast with n_exp
    k_start_exp = k_start
    k_step_exp = k_step

    pi = math.pi
    two_pi = 2.0 * pi

    # We evaluate: C(r) = sum_n x[n] exp(j * 2pi * r_m * (k_start + n * k_step))
    # where r_m = r_min + m * dr.
    # The Bluestein expansion of m*n is: m*n = (m^2 + n^2 - (m-n)^2)/2.
    # The term exp(j * 2pi * m * dr * n * k_step) becomes:
    # exp(j * pi * dr * k_step * m^2) * exp(j * pi * dr * k_step * n^2) * exp(-j * pi * dr * k_step * (m-n)^2)

    # 1. Pre-chirp:
    # phase_n = 2pi * r_min * (k_start + n * k_step) + pi * dr * k_step * n^2
    phase_n = two_pi * r_min * (k_start_exp + n_exp * k_step_exp) + pi * dr * k_step_exp * (n_exp**2)
    pre_chirp = torch.exp(torch.complex(torch.zeros_like(phase_n), phase_n))
    
    y = x_cplx * pre_chirp

    # Convolution kernel length L >= N + M - 1
    L = 2 ** math.ceil(math.log2(N + M - 1))

    # To support batching, v must broadcast over the batch dimensions if k_step is not a scalar
    # Actually, k_step might be different per pulse. We compute v with shape (..., L).
    # Since V is computed via FFT, we compute v exactly matching y's shape (except L in `dim`).
    
    # Evaluate phase for v: phase_v(l) = -pi * dr * k_step * l^2
    # We construct l in [0, M) and [L-N+1, L) like standard CZT.
    l_idx = torch.zeros(L, dtype=real_dtype, device=device)
    if M > 0:
        l_idx[:M] = torch.arange(M, dtype=real_dtype, device=device)
    if N > 1:
        # For l in [-N+1, -1] -> mapped to L-N+1 to L-1
        l_idx[L - N + 1:] = torch.arange(1, N, dtype=real_dtype, device=device).flip(0)

    # Reshape l_idx to align with dim
    shape_l = [1] * x.ndim
    shape_l[dim] = L
    l_exp = l_idx.view(*shape_l)

    # 2. Convolution Kernel V
    phase_v = -pi * dr * k_step_exp * (l_exp**2)
    v_exp = torch.exp(torch.complex(torch.zeros_like(phase_v), phase_v))

    # Perform FFT convolution along `dim`
    Y = torch.fft.fft(y, n=L, dim=dim)
    V = torch.fft.fft(v_exp, n=L, dim=dim)
    conv_full = torch.fft.ifft(Y * V, n=L, dim=dim)

    # Slice output to M points along `dim`
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(0, M)
    conv_m = conv_full[tuple(slices)]

    # 3. Post-chirp phase
    # phase_m = 2pi * k_start * m * dr + pi * dr * k_step * m^2
    phase_m = two_pi * k_start_exp * m_exp * dr + pi * dr * k_step_exp * (m_exp**2)
    post_chirp = torch.exp(torch.complex(torch.zeros_like(phase_m), phase_m))

    output = conv_m * post_chirp
    return output

def czt_resample_kspace_1d(
    signal: torch.Tensor,
    k_start: torch.Tensor,
    k_step: torch.Tensor,
    M_out: int,
    k_out_start: float,
    k_out_step: float,
    spatial_extent: float,
    oversample: float = 2.0
) -> torch.Tensor:
    """
    Resamples a K-space signal to a new uniform K-space grid using Chirp Scaling.
    1. K-space to Spatial Domain via CZT.
    2. Spatial Domain back to K-space via CZT on conjugate.
    """
    N = signal.shape[-1]
    N_spatial = max(int(N * oversample), M_out)
    
    # 1. K-space to Spatial Domain (Inverse Fourier-like)
    spatial = czt_1d_torch(
        signal,
        M=N_spatial,
        r_min=-spatial_extent/2.0,
        r_max=spatial_extent/2.0,
        k_step=k_step,
        k_start=k_start,
        dim=-1
    )
    
    r_step = spatial_extent / max(N_spatial - 1, 1)
    
    device = signal.device
    r_start_t = torch.tensor(-spatial_extent/2.0, device=device, dtype=torch.float64)
    r_step_t = torch.tensor(r_step, device=device, dtype=torch.float64)
    
    # 2. Spatial Domain to Cartesian K-space (Fourier-like)
    # The spatial locations are the "frequencies" of the second CZT
    spatial_conj = torch.conj(spatial)
    
    k_cart = czt_1d_torch(
        spatial_conj,
        M=M_out,
        r_min=k_out_start,
        r_max=k_out_start + (M_out - 1) * k_out_step,
        k_step=r_step_t,
        k_start=r_start_t,
        dim=-1
    )
    
    return torch.conj(k_cart) / float(N_spatial)
