from typing import Tuple
import numpy as np
import torch

from diffpfa.constants import SPEED_OF_LIGHT

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

def apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY):
    P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
    R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
    R_patch = torch.linalg.norm(P_patch, dim=-1)
    dR = R_patch - R_orig
    phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
    corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
    sig_patch = signal * corr_term.to(signal.dtype)
    return sig_patch, P_patch
