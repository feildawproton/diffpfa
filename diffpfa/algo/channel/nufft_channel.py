from typing import Tuple 
import numpy as np
import torch

from diffpfa.algo.channel.geometry_channel import compute_look_vectors, get_image_plane_vectors
from diffpfa.algo.channel.collection_channel import compute_fasttime_frequencies

from diffpfa.algo.channel.patch.nufft_torch import nufft_2d_type1_torch
from diffpfa.algo.channel.patch.geometry_patch import compute_look_components

from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.io.base import CPHDChannelData, CPHDMetadata

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
    
def _apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY):
    P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
    R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
    R_patch = torch.linalg.norm(P_patch, dim=-1)
    dR = R_patch - R_orig
    phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
    corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
    sig_patch = signal * corr_term.to(signal.dtype)
    return sig_patch, P_patch

def process_channel_nufft(ch_data: CPHDChannelData,
                          cphd_meta: CPHDMetadata,
                          u_min: float,
                          u_max: float,
                          r_min: float,
                          r_max: float,
                          N_u: int,
                          N_r: int,
                          nufft_batch_size_pts: int,
                          num_subpatches: int = 1,
                          global_k_ctr_u: float = None,
                          global_k_ctr_r: float = None,
                          image_plane: str = "Ground",
                          device: str = "cuda",
                         ) -> torch.Tensor:
    """Processes a single CPHD channel using 2D PyTorch NUFFT gridding."""
    signal = ch_data.signal.to(device)
    pvp = ch_data.pvp
    N_pulses, N_samples = signal.shape

    signal = _deskew_rvp(signal, pvp, N_samples, device)

    num_patches = max(1, num_subpatches)
    subpatch_size_u = int(np.ceil(N_u / num_patches))
    subpatch_size_r = int(np.ceil(N_r / num_patches))

    P_vecs_orig = compute_look_vectors(pvp, device=device)
    F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=device)
    
    uIAX, uIAY = _get_image_plane_vectors(cphd_meta, image_plane, device)

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
            
            sig_patch, P_patch = _apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY)
            
            cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
            F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
            Ku = F_cpm * cos_theta.unsqueeze(1)
            Kr = F_cpm * sin_theta.unsqueeze(1)
            
            if global_k_ctr_u is not None:
                Ku = Ku - global_k_ctr_u
            if global_k_ctr_r is not None:
                Kr = Kr - global_k_ctr_r
            
            local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
            local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
            p_N_u, p_N_r = end_u - i_u, end_r - i_r
            
            #print(f"going into nufft with u grid size {p_N_u} and r grid size {p_N_r}")
            print(f"going into nufft with u_min {local_u_min} and u_max {local_u_max} and r min {local_r_min} and r max {local_r_max}")
            print(f"woa, i though patches was 1?  i_u: {i_u}, i_r: {i_r}")
            patch_img = nufft_2d_type1_torch(
                signal=sig_patch,
                ku=Ku,
                kr=Kr,
                grid_size_u=p_N_u,
                grid_size_r=p_N_r,
                u_min=local_u_min,
                u_max=local_u_max,
                r_min=local_r_min,
                r_max=local_r_max,
                batch_size_pts=nufft_batch_size_pts
            )
            row_patches.append(patch_img)
        print("len of nufft row patches: " + str(len(row_patches)))
        image_rows.append(torch.cat(row_patches, dim=1))
    print("len of of nufft images: " + str(len(image_rows)))

    image_2d = torch.cat(image_rows, dim=0)

    return image_2d

