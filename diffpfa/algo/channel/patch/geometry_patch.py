from typing import Tuple
import numpy as np
import torch

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
