from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
from scipy.fft import next_fast_len
import math

from diffpfa.IFA.channel.czt_torch import czt_resample_kspace_1d
from diffpfa.IFA.channel.nufft_torch import nufft_grid_1d

from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.types import CPHDMetadata

def _deskew_rvp(signal: torch.Tensor, pvp: dict, N_samples: int, device: str) -> torch.Tensor:
    if "TxFMRate" in pvp:
        gamma = torch.as_tensor(pvp["TxFMRate"], dtype=torch.float64, device=device)
        if "SC0" in pvp and "SCSS" in pvp:
            sc0 = torch.as_tensor(pvp["SC0"], dtype=torch.float64, device=device)
            scss = torch.as_tensor(pvp["SCSS"], dtype=torch.float64, device=device)
            k_idx = torch.arange(N_samples, dtype=torch.float64, device=device)
            F_hz = sc0.unsqueeze(1) + scss.unsqueeze(1) * k_idx.unsqueeze(0)
            rvp_phase = torch.pi * (F_hz ** 2) / gamma.unsqueeze(1)
            rvp_term = torch.exp(torch.complex(torch.zeros_like(rvp_phase), rvp_phase))
            signal = signal * rvp_term.to(signal.dtype)
    return signal
'''
def _get_image_plane_vectors(cphd_meta: CPHDMetadata, image_plane: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_plane == "Slant":
        srp = cphd_meta.srp_ecf
        arp = cphd_meta.arp_pos_coa
        arp_v = cphd_meta.arp_vel_coa
        p_vec = srp - arp
        u_row = p_vec / np.linalg.norm(p_vec)
        u_v = arp_v / np.linalg.norm(arp_v)
        u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
        u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
        uIAX = torch.as_tensor(u_col, dtype=torch.float64, device=device)
        uIAY = torch.as_tensor(u_row, dtype=torch.float64, device=device)
    else:
        uIAX = torch.as_tensor(cphd_meta.uIAX, dtype=torch.float64, device=device)
        uIAY = torch.as_tensor(cphd_meta.uIAY, dtype=torch.float64, device=device)
    return uIAX, uIAY
'''
def _apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY):
    """
    if we were to process this in patches we would need this
    with something like:

    signal = _deskew_rvp(signal, pvp, N_samples, device)

    P_vecs = _compute_look_vectors(pvp, device=device)
    F_hz = _compute_fasttime_frequencies(pvp, N_samples, domain_type, device=device)
    
    uIAX, uIAY = _get_image_plane_vectors(cphd_meta, image_plane, device)

    u_c = (u_min + u_max) / 2.0
    r_c = (r_min + r_max) / 2.0 
    sig_patch, P_patch = _apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY) # why
    
    cos_theta, sin_theta = _compute_look_components(P_vecs, uIAX, uIAY)
    F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
    Ku = F_cpm * cos_theta.unsqueeze(1)
    Kr = F_cpm * sin_theta.unsqueeze(1)
    
    """
    P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
    R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
    R_patch = torch.linalg.norm(P_patch, dim=-1)
    dR = R_patch - R_orig
    phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
    corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
    sig_patch = signal * corr_term.to(signal.dtype)
    return sig_patch, P_patch

def _process_patch_cztnufft(
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

def process_channel(
    signal: torch.Tensor,
    pvp: dict,
    domain_type: str,
    cphd_meta, 
    u_min: float,
    u_max: float,
    r_min: float,
    r_max: float,
    N_u: int,
    N_r: int,
    Ku: torch.tensor,
    Kr: torch.tensor,
    batch_size_pts: int = 1024,
    global_k_ctr_u: Optional[float] = None,
    global_k_ctr_r: Optional[float] = None,
    image_plane: str = "Ground",
    device: str = "cuda"
) -> Tuple[torch.Tensor, bool, int, int]:
    """Processes a single CPHD channel using the cztnufft approach."""
    
    signal = signal.to(device)
    N_pulses, N_samples = signal.shape

    signal = _deskew_rvp(signal, pvp, N_samples, device)
   
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
    
    grid_2d, is_rotated, M_u, M_r = _process_patch_cztnufft(
        sig_patch=signal,
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
