from typing import Dict, Optional, Tuple
import numpy as np
import torch

from diffpfa.constants import SPEED_OF_LIGHT


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


def compute_look_components(
    P_vecs: torch.Tensor,
    uIAX: torch.Tensor,
    uIAY: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects look vectors onto image plane cross-range (uIAX) and range (uIAY) unit vectors
    using pure linear algebra:
        P_U = P . uIAX
        P_R = P . uIAY
        cos(theta) = P_U / sqrt(P_U^2 + P_R^2)
        sin(theta) = P_R / sqrt(P_U^2 + P_R^2)
    Returns:
        cos_theta: tensor of shape (N_pulses,)
        sin_theta: tensor of shape (N_pulses,)
    """
    u_unit = uIAX / torch.linalg.norm(uIAX)
    r_unit = uIAY / torch.linalg.norm(uIAY)

    P_U = torch.matmul(P_vecs, u_unit)
    P_R = torch.matmul(P_vecs, r_unit)

    # Use the true 3D slant-range magnitude to compute direction cosines
    inv_mag = 1.0 / torch.linalg.norm(P_vecs, dim=-1)
    cos_theta = P_U * inv_mag
    sin_theta = P_R * inv_mag

    return cos_theta, sin_theta


def compute_fasttime_frequencies(
    pvp: Dict[str, np.ndarray],
    num_samples: int,
    domain_type: str = "FX",
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Calculates fast-time RF frequencies F(n, k) in Hz for each pulse n and sample k.
    Returns tensor of shape (N_pulses, N_samples).
    """
    num_pulses = len(pvp["SC0"]) if "SC0" in pvp else len(pvp["FX1"])

    if domain_type == "FX":
        sc0 = torch.as_tensor(pvp["SC0"], dtype=torch.float64, device=device)
        scss = torch.as_tensor(pvp["SCSS"], dtype=torch.float64, device=device)
        k_indices = torch.arange(num_samples, dtype=torch.float64, device=device)
        # F(n, k) = SC0[n] + k * SCSS[n]
        F_hz = sc0.unsqueeze(1) + scss.unsqueeze(1) * k_indices.unsqueeze(0)
        return F_hz
    else:
        raise NotImplementedError(f"DomainType '{domain_type}' not supported yet. Must be 'FX'.")


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
