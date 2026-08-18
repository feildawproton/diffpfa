from typing import Tuple, Optional
import numpy as np
import torch

from diffpfa.algo.channel.geometry_channel import compute_look_vectors, get_image_plane_vectors, deskew_rvp
from diffpfa.algo.channel.collection_channel import compute_fasttime_frequencies

from diffpfa.algo.channel.patch.nufft_torch import nufft_2d_type1_torch
from diffpfa.algo.channel.patch.cztnufft_patch import process_patch_cztnufft
from diffpfa.algo.channel.patch.geometry_patch import compute_look_components, apply_scp_shift

from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.io.base import CPHDChannelData, CPHDMetadata

def process_channel(
    ch_data: CPHDChannelData,
    cphd_meta: CPHDMetadata,
    u_min: float,
    u_max: float,
    r_min: float,
    r_max: float,
    N_u: int,
    N_r: int,
    mode: str,
    batch_size_pts: Optional[int] = None,
    num_subpatches: int = 1,
    global_k_ctr_u: Optional[float] = None,
    global_k_ctr_r: Optional[float] = None,
    image_plane: str = "Ground",
    device: str = "cuda"
) -> torch.Tensor:
    """Processes a single CPHD channel using the specified PFA mode (nufft or cztnufft)."""
    
    signal = ch_data.signal.to(device)
    pvp = ch_data.pvp
    N_pulses, N_samples = signal.shape

    signal = deskew_rvp(signal, pvp, N_samples, device)

    num_patches = max(1, num_subpatches)
    subpatch_size_u = int(np.ceil(N_u / num_patches))
    subpatch_size_r = int(np.ceil(N_r / num_patches))

    P_vecs_orig = compute_look_vectors(pvp, device=device)
    F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=device)
    
    uIAX, uIAY = get_image_plane_vectors(cphd_meta, image_plane, device)

    u_edges = np.linspace(u_min, u_max, N_u + 1)
    r_edges = np.linspace(r_min, r_max, N_r + 1)

    image_rows = []
    for i_u in range(0, N_u, subpatch_size_u):
        row_patches = []
        for i_r in range(0, N_r, subpatch_size_r):
            end_u = min(i_u + subpatch_size_u, N_u)
            end_r = min(i_r + subpatch_size_r, N_r)
            
            p_u_min, p_u_max = u_edges[i_u], u_edges[end_u]
            p_r_min, p_r_max = r_edges[i_r], r_edges[end_r]
            
            u_c = (p_u_min + p_u_max) / 2.0
            r_c = (p_r_min + p_r_max) / 2.0
            
            sig_patch, P_patch = apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY)
            
            cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
            F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
            Ku = F_cpm * cos_theta.unsqueeze(1)
            Kr = F_cpm * sin_theta.unsqueeze(1)
            
            if global_k_ctr_u is not None:
                k_ctr_u = global_k_ctr_u
            else:
                k_ctr_u = (Ku.min() + Ku.max()).item() / 2.0
                
            if global_k_ctr_r is not None:
                k_ctr_r = global_k_ctr_r
            else:
                k_ctr_r = (Kr.min() + Kr.max()).item() / 2.0
                
            local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
            local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
            p_N_u, p_N_r = end_u - i_u, end_r - i_r
            
            if mode == "nufft":
                if global_k_ctr_u is not None:
                    Ku_centered = Ku - global_k_ctr_u
                else:
                    Ku_centered = Ku
                    
                if global_k_ctr_r is not None:
                    Kr_centered = Kr - global_k_ctr_r
                else:
                    Kr_centered = Kr
                    
                patch_img = nufft_2d_type1_torch(
                    signal=sig_patch,
                    ku=Ku_centered,
                    kr=Kr_centered,
                    grid_size_u=p_N_u,
                    grid_size_r=p_N_r,
                    u_min=local_u_min,
                    u_max=local_u_max,
                    r_min=local_r_min,
                    r_max=local_r_max,
                    batch_size_pts=batch_size_pts
                )
            elif mode == "cztnufft":
                L_u = local_u_max - local_u_min
                L_r = local_r_max - local_r_min
                
                patch_img = process_patch_cztnufft(
                    sig_patch=sig_patch,
                    Ku=Ku,
                    Kr=Kr,
                    p_N_u=p_N_u,
                    p_N_r=p_N_r,
                    L_u=L_u,
                    L_r=L_r,
                    k_ctr_u=k_ctr_u,
                    k_ctr_r=k_ctr_r,
                    czt_batch_size=batch_size_pts,
                    device=device
                )
            else:
                raise ValueError(f"Unknown IFP mode: {mode}")
                
            row_patches.append(patch_img)
        image_rows.append(torch.cat(row_patches, dim=1))
        
    image_2d = torch.cat(image_rows, dim=0)

    return image_2d
