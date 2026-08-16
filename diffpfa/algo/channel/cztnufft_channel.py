import math
from typing import Tuple 
import numpy as np
import torch
from scipy.fft import next_fast_len

from diffpfa.algo.channel.collection_channel import compute_fasttime_frequencies 
from diffpfa.algo.channel.geometry_channel import  compute_look_vectors, get_image_plane_vectors

from diffpfa.algo.channel.patch.czt_torch import czt_resample_kspace_1d
from diffpfa.algo.channel.patch.nufft_torch import nufft_grid_1d
from diffpfa.algo.channel.patch.geometry_patch import compute_look_components

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
                              return_kspace: bool = False,
                              image_plane: str = "Ground",
                              device: str = "cuda",
                              ) -> torch.Tensor:
    """Processes a single CPHD channel using a CZTNUFFT approach: 1D CZT in Range, 1D NUFFT in Cross-Range."""
    # Pin memory for async H2D copy
    if device.type == "cuda" and ch_data.signal.device.type == "cpu":
        ch_data.signal = ch_data.signal.pin_memory()
        
    pvp = ch_data.pvp
    N_pulses, N_samples = ch_data.signal.shape

    num_patches = max(1, num_subpatches)
    subpatch_size_u = int(np.ceil(N_u / num_patches))
    subpatch_size_r = int(np.ceil(N_r / num_patches))

    image_2d = torch.zeros((N_u, N_r), dtype=torch.complex64, device=device)
    
    P_vecs_orig = compute_look_vectors(pvp, device=device)
    F_hz_full = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=device)
    
    # Precompute RVP params if necessary
    has_rvp = "TxFMRate" in pvp and "SC0" in pvp and "SCSS" in pvp
    if has_rvp:
        gamma_full = torch.as_tensor(pvp["TxFMRate"], dtype=torch.float64, device=device)
    
    uIAX, uIAY = get_image_plane_vectors(cphd_meta, image_plane, device)

    u_edges = np.linspace(u_min, u_max, N_u + 1)
    r_edges = np.linspace(r_min, r_max, N_r + 1)
    
    czt_batch_size = czt_batch_size
    
    stream_compute = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    stream_copy = torch.cuda.Stream(device=device) if device.type == "cuda" else None

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
            
            oversample = 1.5
            M_u = next_fast_len(int(math.ceil(p_N_u * oversample)))
            M_r = next_fast_len(int(math.ceil(p_N_r * oversample)))
            L_u = local_u_max - local_u_min
            L_r = local_r_max - local_r_min
            dK_u = p_N_u / (M_u * max(L_u, 1e-12))
            dK_r = p_N_r / (M_r * max(L_r, 1e-12))
            k_ctr_u = global_k_ctr_u if global_k_ctr_u is not None else (Ku.min() + Ku.max()).item() / 2.0
            k_ctr_r = global_k_ctr_r if global_k_ctr_r is not None else (Kr.min() + Kr.max()).item() / 2.0
            
            # 1. Range CZT Resampling (K-space -> Cartesian K_r)
            k_out_start_r = k_ctr_r - (M_r / 2.0) * dK_r
            k_out_step_r = dK_r
            
            range_resampled = torch.zeros((N_pulses, M_r), dtype=torch.complex64, device=device)
            
            def get_batch(b_idx):
                if b_idx >= N_pulses: return None
                b_e = min(b_idx + czt_batch_size, N_pulses)
                kwargs = {"non_blocking": True} if device.type == "cuda" else {}
                return ch_data.signal[b_idx:b_e].to(device, **kwargs), b_idx, b_e

            if stream_copy:
                with torch.cuda.stream(stream_copy):
                    next_batch_info = get_batch(0)
            else:
                next_batch_info = get_batch(0)
            
            for b in range(0, N_pulses, czt_batch_size):
                if stream_compute and stream_copy:
                    stream_compute.wait_stream(stream_copy)
                    
                sig_gpu, b_start, b_end = next_batch_info
                
                if stream_copy and b + czt_batch_size < N_pulses:
                    with torch.cuda.stream(stream_copy):
                        next_batch_info = get_batch(b + czt_batch_size)
                elif not stream_copy and b + czt_batch_size < N_pulses:
                    next_batch_info = get_batch(b + czt_batch_size)
                        
                with torch.cuda.stream(stream_compute) if stream_compute else torch.autograd.profiler.record_function("cpu_compute"):
                    F_hz_b = F_hz_full[b_start:b_end]
                    
                    # Deskew RVP for batch
                    if has_rvp:
                        gamma_b = gamma_full[b_start:b_end]
                        rvp_phase = torch.pi * (F_hz_b ** 2) / gamma_b.unsqueeze(1)
                        rvp_term = torch.exp(torch.complex(torch.zeros_like(rvp_phase), rvp_phase))
                        sig_gpu = sig_gpu * rvp_term.to(sig_gpu.dtype)
                        
                    # Apply SCP Shift for batch
                    dR_b = dR_full[b_start:b_end]
                    phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz_b * dR_b.unsqueeze(1)
                    corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
                    sig_patch_b = sig_gpu * corr_term.to(sig_gpu.dtype)
                    
                    # Apply CZT for batch
                    kr_b = Kr[b_start:b_end]
                    k_start = kr_b[:, 0].unsqueeze(1)
                    k_step = ((kr_b[:, -1] - kr_b[:, 0]) / max(N_samples - 1, 1)).unsqueeze(1)
                    
                    batch_out = czt_resample_kspace_1d(
                        sig_patch_b, 
                        k_start=k_start, 
                        k_step=k_step,
                        M_out=M_r,
                        k_out_start=k_out_start_r,
                        k_out_step=k_out_step_r,
                        spatial_extent=L_r,
                        oversample=oversample
                    )
                    range_resampled[b_start:b_end, :] = batch_out

            if stream_compute:
                torch.cuda.current_stream(device=device).wait_stream(stream_compute)
            
            # 2. Cross-Range NUFFT Gridding (Cartesian K_r -> Cartesian K_u)
            cot_theta = Ku[:, N_samples//2] / Kr[:, N_samples//2]
            m_idx = torch.arange(M_r, device=device, dtype=torch.float64)
            Kr_cart = k_out_start_r + m_idx * dK_r
            
            del Ku, Kr, F_hz_full, dR_full
            
            nufft_batch_size = czt_batch_size
            grid_2d = torch.zeros((M_u, M_r), dtype=range_resampled.dtype, device=device)
            
            for b in range(0, M_r, nufft_batch_size):
                chunk_signal = range_resampled[:, b:b + nufft_batch_size]
                chunk_Kr_cart = Kr_cart[b:b + nufft_batch_size]
                chunk_Ku_cart = cot_theta.unsqueeze(1) * chunk_Kr_cart.unsqueeze(0)
                
                chunk_grid = nufft_grid_1d(
                    signal=chunk_signal,
                    kx=chunk_Ku_cart,
                    grid_size=p_N_u,
                    L_x=L_u,
                    k_center=k_ctr_u,
                    oversample=oversample
                )
                grid_2d[:, b:b+nufft_batch_size] = chunk_grid
            
            del range_resampled
            
            # 3. 2D IFFT and Deconvolution
            if return_kspace:
                if num_patches > 1:
                    raise ValueError("Cannot return kspace when num_subpatches > 1")
                return grid_2d
                
            grid_shifted = torch.fft.ifftshift(grid_2d)
            del grid_2d
            img_oversampled = torch.fft.ifft2(grid_shifted)
            del grid_shifted
            img_oversampled.mul_(M_u * M_r)
            img_shifted = torch.fft.fftshift(img_oversampled)
            del img_oversampled
            
            beta = 13.9086
            J = 6
            real_dtype = torch.float64
            grid_u_coords = (torch.arange(M_u, device=device, dtype=real_dtype) - M_u / 2.0) / M_u
            deconv_u = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=device)**2 - (math.pi * J * grid_u_coords)**2, min=1e-12)))
            
            img_deconv = img_shifted / (deconv_u.unsqueeze(1) + 1e-12)
            
            start_u = (M_u - p_N_u) // 2
            start_r = (M_r - p_N_r) // 2
            
            patch_img = img_deconv[start_u : start_u + p_N_u, start_r : start_r + p_N_r]
            image_2d[i_u:end_u, i_r:end_r] = patch_img

    return image_2d

