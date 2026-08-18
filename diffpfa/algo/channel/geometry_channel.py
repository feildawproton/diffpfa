from typing import Dict, Tuple
import numpy as np
import torch

from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.io.base import CPHDMetadata

from diffpfa.algo.channel.patch.geometry_patch import compute_look_components
from diffpfa.algo.channel.collection_channel import compute_fasttime_frequencies

def compute_look_vectors(pvp: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Computes look vectors P_n = r_SRP,n - r_Rcv,n (or midpoint relative to SRP).
    Returns tensor of shape (N_pulses, 3).
    """
    srp = torch.as_tensor(pvp["SRPPos"], dtype=torch.float64, device=device)

    if "RcvPos" in pvp:
        rcv = torch.as_tensor(pvp["RcvPos"], dtype=torch.float64, device=device)
        if "TxPos" in pvp:
            tx = torch.as_tensor(pvp["TxPos"], dtype=torch.float64, device=device)
            # Bistatic / APC midpoint
            phase_center = 0.5 * (tx + rcv)
        else:
            phase_center = rcv
    else:
        raise ValueError("PVP must contain 'RcvPos' or 'SRPPos'.")

    # Vector from phase center to SRP
    P_vecs = srp - phase_center
    return P_vecs

def compute_kspace(
    pvp: Dict[str, np.ndarray],
    uIAX_np: np.ndarray,
    uIAY_np: np.ndarray,
    num_samples: int,
    domain_type: str = "FX",
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes spatial frequency mappings (K_u, K_r) in cycles/meter for each sample (pulse, sample).
    Formula:
        K_u(n, k) = (2 * F(n, k) / c) * cos(theta_n)   [cycles/meter]
        K_r(n, k) = (2 * F(n, k) / c) * sin(theta_n)   [cycles/meter]
    """
    uIAX = torch.as_tensor(uIAX_np, dtype=torch.float64, device=device)
    uIAY = torch.as_tensor(uIAY_np, dtype=torch.float64, device=device)

    P_vecs = compute_look_vectors(pvp, device=device)
    cos_theta, sin_theta = compute_look_components(P_vecs, uIAX, uIAY)

    F_hz = compute_fasttime_frequencies(pvp, num_samples, domain_type=domain_type, device=device)
    F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT  # cycles/meter

    K_u = F_cpm * cos_theta.unsqueeze(1)  # (N_pulses, N_samples)
    K_r = F_cpm * sin_theta.unsqueeze(1)  # (N_pulses, N_samples)

    return K_u, K_r
    

def get_image_plane_vectors(cphd_meta: CPHDMetadata, image_plane: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
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

def deskew_rvp(signal: torch.Tensor, pvp: dict, N_samples: int, device: str) -> torch.Tensor:
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
