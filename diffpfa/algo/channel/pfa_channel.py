from typing import Tuple, Optional, Union
import numpy as np
import torch

from diffpfa.algo.channel.geometry_channel import compute_look_vectors, get_image_plane_vectors, deskew_rvp
from diffpfa.algo.channel.collection_channel import compute_fasttime_frequencies

from diffpfa.algo.channel.cztnufft import process_patch_cztnufft, apply_ifft_and_deconv
from diffpfa.algo.channel.geometry_channel import compute_look_components, apply_scp_shift

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
    batch_size_pts: Optional[int] = None,
    global_k_ctr_u: Optional[float] = None,
    global_k_ctr_r: Optional[float] = None,
    image_plane: str = "Ground",
    device: str = "cuda"
) -> Tuple[torch.Tensor, bool, int, int]:
    """Processes a single CPHD channel using the cztnufft approach."""
    
    signal = ch_data.signal.to(device)
    pvp = ch_data.pvp
    N_pulses, N_samples = signal.shape

    signal = deskew_rvp(signal, pvp, N_samples, device)

    P_vecs_orig = compute_look_vectors(pvp, device=device)
    F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=device)
    
    uIAX, uIAY = get_image_plane_vectors(cphd_meta, image_plane, device)

    u_c = (u_min + u_max) / 2.0
    r_c = (r_min + r_max) / 2.0
    
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
        
    L_u = u_max - u_min
    L_r = r_max - r_min
    
    grid_2d, is_rotated, M_u, M_r = process_patch_cztnufft(
        sig_patch=sig_patch,
        Ku=Ku,
        Kr=Kr,
        p_N_u=N_u,
        p_N_r=N_r,
        L_u=L_u,
        L_r=L_r,
        k_ctr_u=k_ctr_u,
        k_ctr_r=k_ctr_r,
        czt_batch_size=batch_size_pts,
        device=device
    )
    
    return grid_2d, is_rotated, M_u, M_r
