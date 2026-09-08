import math
import torch

def _czt_range(
    x: torch.Tensor,
    M: int,
    r_min: float,
    r_max: float,
    k_step: torch.Tensor,
    k_start: torch.Tensor,
) -> torch.Tensor:
    """
    Computes 1D Chirp Z-Transform (CZT) along the range dimension.
    Evaluates sum_n x[n] * exp(j * 2pi * r_m * (k_start + n * k_step))
    where r_m linearly spaces from r_min to r_max with M points.

    Args:
        x: Input tensor (real or complex).
        M: Number of output spatial bins.
        r_min: Minimum evaluation spatial coordinate (e.g., meters).
        r_max: Maximum evaluation spatial coordinate (e.g., meters).
        k_step: Spatial frequency step per sample (cycles/unit). Tensor broadcastable to x.
        k_start: Starting spatial frequency (cycles/unit). Tensor broadcastable to x.

    Returns:
        Complex tensor with size M along dimension `dim`.
    """
    
    N = x.shape[1]
    dr = (r_max - r_min) / max(M - 1, 1)

    device = x.device

    # -- Pin Float64 Precision in CZT --
    # Oftern, Pre- and post-chirp phases evaluate 2*pi*r_min*k where r_min ~ -L/2 ~ -2500mand k ~ 64 cyc/m, 
    # resulting in absolute phases around ~1e6 rad.
    # Evaluating phases in float32 limits precision to ~0.06 rad (~3.4 degrees) of random phase noise per sample, 
    # reducing SNR by -21 dB.
    # Therefore, chirp phases and Bluestein convolution are always computed in float64/complex128.
    
    x_cplx = x.to(torch.complex128)

    n = torch.arange(N, dtype=torch.float64, device=device)
    m = torch.arange(M, dtype=torch.float64, device=device)

    # Reshape n and m to align with target dimension 1
    shape_n = [1] * x.ndim
    shape_n[1] = N
    n_exp = n.view(*shape_n)

    shape_m = [1] * x.ndim
    shape_m[1] = M
    m_exp = m.view(*shape_m)

    # Expand k_start and k_step to broadcast with n_exp in float64
    k_start_exp = k_start.to(torch.float64) if isinstance(k_start, torch.Tensor) else k_start
    k_step_exp = k_step.to(torch.float64) if isinstance(k_step, torch.Tensor) else k_step

    # We evaluate: C(r) = sum_n x[n] exp(j * 2pi * r_m * (k_start + n * k_step))
    # where r_m = r_min + m * dr.
    # The Bluestein expansion of m*n is: m*n = (m^2 + n^2 - (m-n)^2)/2.
    # The term exp(j * 2pi * m * dr * n * k_step) becomes:
    # exp(j * pi * dr * k_step * m^2) * exp(j * pi * dr * k_step * n^2) * exp(-j * pi * dr * k_step * (m-n)^2)

    # 1. Pre-chirp:
    # phase_n = 2pi * r_min * (k_start + n * k_step) + pi * dr * k_step * n^2
    phase_n = 2.0 * torch.pi * r_min * (k_start_exp + n_exp * k_step_exp) + torch.pi * dr * k_step_exp * (n_exp**2)
    pre_chirp = torch.exp(torch.complex(torch.zeros_like(phase_n), phase_n))
    
    y = x_cplx * pre_chirp

    # Convolution kernel length L >= N + M - 1
    L = 2 ** math.ceil(math.log2(N + M - 1))
        
    # To support batching, v must broadcast over the batch dimensions if k_step is not a scalar
    # Actually, k_step might be different per pulse. We compute v with shape (..., L).
    # Since V is computed via FFT, we compute v exactly matching y's shape (except L in dim 1).
    
    # Evaluate phase for v: phase_v(l) = -pi * dr * k_step * l^2
    # We construct l in [0, M) and [L-N+1, L) like standard CZT.
    l_idx = torch.zeros(L, dtype=torch.float64, device=device)
    if M > 0:
        l_idx[:M] = torch.arange(M, dtype=torch.float64, device=device)
    if N > 1:
        # For l in [-N+1, -1] -> mapped to L-N+1 to L-1
        l_idx[L - N + 1:] = torch.arange(1, N, dtype=torch.float64, device=device).flip(0)

    # Reshape l_idx to align with dim 1
    shape_l = [1] * x.ndim
    shape_l[1] = L
    l_exp = l_idx.view(*shape_l)

    # 2. Convolution Kernel V
    phase_v = -torch.pi * dr * k_step_exp * (l_exp**2)
    v_exp = torch.exp(torch.complex(torch.zeros_like(phase_v), phase_v))

    # Perform FFT convolution along dim 1
    Y = torch.fft.fft(y, n=L, dim=1)
    V = torch.fft.fft(v_exp, n=L, dim=1)
    conv_full = torch.fft.ifft(Y * V, n=L, dim=1)

    # Slice output to M points along dim 1
    slices = [slice(None)] * x.ndim
    slices[1] = slice(0, M)
    conv_m = conv_full[tuple(slices)]

    # 3. Post-chirp phase
    # phase_m = 2pi * k_start * m * dr + pi * dr * k_step * m^2
    phase_m = 2.0 * torch.pi * k_start_exp * m_exp * dr + torch.pi * dr * k_step_exp * (m_exp**2)
    post_chirp = torch.exp(torch.complex(torch.zeros_like(phase_m), phase_m))

    output = (conv_m * post_chirp).to(x.dtype if x.is_complex() else torch.complex128)
    return output

def batch_czt_range(
    signal: torch.Tensor,
    k_start: torch.Tensor,
    k_step: torch.Tensor,
    M_out: int,
    k_out_start: float,
    k_out_step: float,
    spatial_extent: float,
    oversample: float = 1.0,
    batch_size: int = None
) -> torch.Tensor:
    """
    Resamples a K-space signal to a new uniform K-space grid using Chirp Scaling.
    1. K-space to Spatial Domain via CZT.
    2. Spatial Domain back to K-space via CZT on conjugate.
    """
    N = signal.shape[-1]
    
    # Calculate input K-space bandwidth to prevent spatial aliasing
    BW_in = (torch.abs(k_step) * (N - 1)).max().item() if isinstance(k_step, torch.Tensor) else abs(k_step) * (N - 1)
    N_req = int(math.ceil(spatial_extent * BW_in)) + 1
    
    N_spatial = max(int(N * oversample), M_out, N_req)
    
    device = signal.device
    r_step = spatial_extent / max(N_spatial - 1, 1)
    r_start_t = torch.tensor(-spatial_extent/2.0, device=device, dtype=torch.float64)
    r_step_t = torch.tensor(r_step, device=device, dtype=torch.float64)
    
    k_cart = torch.zeros(signal.shape[:-1] + (M_out,), dtype=signal.dtype, device=device)
    
    batch_size = batch_size or (signal.shape[0] if signal.ndim > 1 else 1)
    
    for b in range(0, signal.shape[0] if signal.ndim > 1 else 1, batch_size):
        b_end = min(b + batch_size, signal.shape[0] if signal.ndim > 1 else 1)
        
        if signal.ndim > 1:
            sig_b = signal[b:b_end]
            k_start_b = k_start[b:b_end] if k_start.ndim > 0 else k_start
            k_step_b = k_step[b:b_end] if k_step.ndim > 0 else k_step
        else:
            sig_b = signal
            k_start_b = k_start
            k_step_b = k_step
            
        # 1. Polar K-space to Spatial Domain (Inverse Fourier-like)
        spatial_b = _czt_range(
            sig_b,
            M=N_spatial,
            r_min=-spatial_extent/2.0,
            r_max=spatial_extent/2.0,
            k_step=k_step_b,
            k_start=k_start_b,
        )
        
        # 2. Spatial Domain to Cartesian K-space (Fourier-like)
        spatial_conj_b = torch.conj(spatial_b)
        
        k_cart_b = _czt_range(
            spatial_conj_b,
            M=M_out,
            r_min=k_out_start,
            r_max=k_out_start + (M_out - 1) * k_out_step,
            k_step=r_step_t,
            k_start=r_start_t,
        )
        
        # -- NOTE (Audit C7 Remediated): CZT Resampler Normalization --
        # Previously, the resampled signal was normalized as:
        #   torch.conj(k_cart_b) / float(N_spatial)
        # However, the forward CZT sums over N_spatial discrete samples with spacing
        # r_step = spatial_extent / (N_spatial - 1), while the inverse CZT sums over
        # the input k-samples without multiplying by the measure |k_step|.
        # Dividing by N_spatial yielded an overall gain of 1 / (L * |k_step|), which caused
        # sub-bands with different sample spacing (e.g. SCSS) to be weighted inversely
        # to their sample spacing (a 2x denser subband received 2x the weight).
        # Multiplying by |k_step| * r_step weights the continuous integral appropriately:
        k_step_val = torch.abs(k_step_b) if isinstance(k_step_b, torch.Tensor) else abs(k_step_b)
        weight = k_step_val * r_step_t
        while isinstance(weight, torch.Tensor) and weight.ndim < k_cart_b.ndim:
            weight = weight.unsqueeze(-1)

        if signal.ndim > 1:
            k_cart[b:b_end] = (torch.conj(k_cart_b) * weight).to(k_cart.dtype)
        else:
            k_cart = (torch.conj(k_cart_b) * weight).to(k_cart.dtype)
    
    return k_cart
