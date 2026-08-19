import math
from scipy.fft import next_fast_len
import torch

from diffpfa.algo.channel.czt_torch import czt_resample_kspace_1d
from diffpfa.algo.channel.nufft_torch import nufft_grid_1d

def process_patch_cztnufft(
    sig_patch: torch.Tensor,
    Ku: torch.Tensor,
    Kr: torch.Tensor,
    p_N_u: int,
    p_N_r: int,
    L_u: float,
    L_r: float,
    k_ctr_u: float,
    k_ctr_r: float,
    czt_batch_size: int,
    device: str
) -> torch.Tensor:
    
    N_samples = sig_patch.shape[-1]
    
    M_u = next_fast_len(int(math.ceil(p_N_u)))
    M_r = next_fast_len(int(math.ceil(p_N_r)))
    dK_u = p_N_u / (M_u * max(L_u, 1e-12))
    dK_r = p_N_r / (M_r * max(L_r, 1e-12))
    
    # Orientation check
    # We use the center sample as the representative lookup
    cos_theta = Ku[:, N_samples//2] / (torch.sqrt(Ku[:, N_samples//2]**2 + Kr[:, N_samples//2]**2) + 1e-12)
    sin_theta = Kr[:, N_samples//2] / (torch.sqrt(Ku[:, N_samples//2]**2 + Kr[:, N_samples//2]**2) + 1e-12)
    
    is_rotated = abs(cos_theta.mean()) > abs(sin_theta.mean())
    
    if not is_rotated:
        k_out_start_r = k_ctr_r - (M_r / 2.0) * dK_r
        k_out_step_r = dK_r
        
        k_start = Kr[:, 0].unsqueeze(1)
        k_step = ((Kr[:, -1] - Kr[:, 0]) / max(N_samples - 1, 1)).unsqueeze(1)
        
        fast_resampled = czt_resample_kspace_1d(
            sig_patch, 
            k_start=k_start, 
            k_step=k_step,
            M_out=M_r,
            k_out_start=k_out_start_r,
            k_out_step=k_out_step_r,
            spatial_extent=L_r,
            batch_size=czt_batch_size,
        )
        
        cot_theta = Ku[:, N_samples//2] / Kr[:, N_samples//2]
        m_idx = torch.arange(M_r, device=device, dtype=torch.float64)
        Kr_cart = k_out_start_r + m_idx * dK_r
        
        grid_2d = nufft_grid_1d(
            signal=fast_resampled,
            kx=(cot_theta, Kr_cart),
            grid_size=p_N_u,
            L_x=L_u,
            k_center=k_ctr_u,
            batch_size=czt_batch_size,
        )
    else:
        k_out_start_u = k_ctr_u - (M_u / 2.0) * dK_u
        k_out_step_u = dK_u
        
        k_start = Ku[:, 0].unsqueeze(1)
        k_step = ((Ku[:, -1] - Ku[:, 0]) / max(N_samples - 1, 1)).unsqueeze(1)
        
        fast_resampled = czt_resample_kspace_1d(
            sig_patch, 
            k_start=k_start, 
            k_step=k_step,
            M_out=M_u,
            k_out_start=k_out_start_u,
            k_out_step=k_out_step_u,
            spatial_extent=L_u,
            batch_size=czt_batch_size,
        )
        
        tan_theta = Kr[:, N_samples//2] / Ku[:, N_samples//2]
        m_idx = torch.arange(M_u, device=device, dtype=torch.float64)
        Ku_cart = k_out_start_u + m_idx * dK_u
        
        grid_2d = nufft_grid_1d(
            signal=fast_resampled,
            kx=(tan_theta, Ku_cart),
            grid_size=p_N_r,
            L_x=L_r,
            k_center=k_ctr_r,
            batch_size=czt_batch_size,
        )
        grid_2d = grid_2d.T
    
    return grid_2d, is_rotated, M_u, M_r

def apply_ifft_and_deconv(grid_2d: torch.Tensor, is_rotated: bool, M_u: int, M_r: int, N_u: int, N_r: int, device: str) -> torch.Tensor:
    grid_shifted = torch.fft.ifftshift(grid_2d)
    img_oversampled = torch.fft.ifft2(grid_shifted)
    img_oversampled.mul_(M_u * M_r)
    img_shifted = torch.fft.fftshift(img_oversampled)
    
    beta = 13.9086
    J = 6
    real_dtype = torch.float64
    
    if not is_rotated:
        grid_coords = (torch.arange(M_u, device=device, dtype=real_dtype) - M_u / 2.0) / M_u
        deconv = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=device)**2 - (math.pi * J * grid_coords)**2, min=1e-12)))
        img_deconv = img_shifted / (deconv.unsqueeze(1) + 1e-12)
    else:
        grid_coords = (torch.arange(M_r, device=device, dtype=real_dtype) - M_r / 2.0) / M_r
        deconv = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=device)**2 - (math.pi * J * grid_coords)**2, min=1e-12)))
        img_deconv = img_shifted / (deconv.unsqueeze(0) + 1e-12)
    
    start_u = (M_u - N_u) // 2
    start_r = (M_r - N_r) // 2
    
    patch_img = img_deconv[start_u : start_u + N_u, start_r : start_r + N_r]
    return patch_img
