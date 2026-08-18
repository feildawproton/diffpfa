import math
from typing import Tuple 
import numpy as np
import torch
from scipy.fft import next_fast_len

from diffpfa.algo.channel.collection_channel import compute_fasttime_frequencies 
from diffpfa.algo.channel.geometry_channel import compute_look_vectors, get_image_plane_vectors, deskew_rvp

from diffpfa.algo.channel.patch.czt_torch import czt_resample_kspace_1d
from diffpfa.algo.channel.patch.nufft_torch import nufft_grid_1d
from diffpfa.algo.channel.patch.geometry_patch import compute_look_components, apply_scp_shift

from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.io.base import CPHDChannelData, CPHDMetadata

def process_channel_czt_nufft(ch_data: CPHDChannelData,
                              cphd_meta: CPHDMetadata,
                              u_min: float,
                              u_max: float,
                              r_min: float,
                              r_max: float,
                              N_u: int,
                              N_r: int,
                              czt_batch_size: int,
                              num_subpatches: int = 1,
                              global_k_ctr_u: float = None,
                              global_k_ctr_r: float = None, 
                              image_plane: str = "Ground",
                              device: str = "cuda",
                              ) -> torch.Tensor:
    """Processes a single CPHD channel using a CZTNUFFT approach: 1D CZT in Range, 1D NUFFT in Cross-Range."""
    signal = ch_data.signal.to(device)
    pvp = ch_data.pvp
    N_pulses, N_samples = signal.shape

    num_patches = max(1, num_subpatches)
    subpatch_size_u = int(np.ceil(N_u / num_patches))
    subpatch_size_r = int(np.ceil(N_r / num_patches))

    image_2d = torch.zeros((N_u, N_r), dtype=torch.complex64, device=device)
    
    P_vecs_orig = compute_look_vectors(pvp, device=device)
    F_hz_full = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=device)
    
    # RVP params are now handled internally by deskew_rvp
    
    uIAX, uIAY = get_image_plane_vectors(cphd_meta, image_plane, device)

    u_edges = np.linspace(u_min, u_max, N_u + 1)
    r_edges = np.linspace(r_min, r_max, N_r + 1)
    
    for i_u in range(0, N_u, subpatch_size_u):
        for i_r in range(0, N_r, subpatch_size_r):
            end_u = min(i_u + subpatch_size_u, N_u)
            end_r = min(i_r + subpatch_size_r, N_r)
            
            p_u_min, p_u_max = u_edges[i_u], u_edges[end_u]
            p_r_min, p_r_max = r_edges[i_r], r_edges[end_r]
            
            u_c = (p_u_min + p_u_max) / 2.0
            r_c = (p_r_min + p_r_max) / 2.0
            
            P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
            R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
            R_patch = torch.linalg.norm(P_patch, dim=-1)
            dR_full = R_patch - R_orig
            
            cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
            F_cpm = 2.0 * F_hz_full / SPEED_OF_LIGHT
            Ku = F_cpm * cos_theta.unsqueeze(1)
            Kr = F_cpm * sin_theta.unsqueeze(1)
            
            local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
            local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
            
            p_N_u, p_N_r = end_u - i_u, end_r - i_r
            
            M_u = next_fast_len(int(math.ceil(p_N_u)))
            M_r = next_fast_len(int(math.ceil(p_N_r)))
            L_u = local_u_max - local_u_min
            L_r = local_r_max - local_r_min
            dK_u = p_N_u / (M_u * max(L_u, 1e-12))
            dK_r = p_N_r / (M_r * max(L_r, 1e-12))
            k_ctr_u = global_k_ctr_u if global_k_ctr_u is not None else (Ku.min() + Ku.max()).item() / 2.0
            k_ctr_r = global_k_ctr_r if global_k_ctr_r is not None else (Kr.min() + Kr.max()).item() / 2.0
            
            sig_gpu = deskew_rvp(signal, pvp, N_samples, device)
            sig_patch, _ = apply_scp_shift(sig_gpu, F_hz_full, P_vecs_orig, u_c, r_c, uIAX, uIAY)
            
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
            
            start_u = (M_u - p_N_u) // 2
            start_r = (M_r - p_N_r) // 2
            
            patch_img = img_deconv[start_u : start_u + p_N_u, start_r : start_r + p_N_r]
            image_2d[i_u:end_u, i_r:end_r] = patch_img

    return image_2d
