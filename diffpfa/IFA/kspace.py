import torch
import numpy as np
from typing import Dict, Tuple

from diffpfa.constants import SPEED_OF_LIGHT

def compute_kspace(
    u_axis: torch.Tensor,
    r_axis: torch.Tensor,
    SRPPos: torch.Tensor,
    RcvPos: torch.Tensor,
    TxPos: torch.Tensor,
    Fx0: torch.Tensor,
    FxSS: torch.Tensor,
    num_samples: int,
    device: torch.device = torch.device("cuda")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes spatial frequency mappings (K_u, K_r) in cycles/meter for each sample (pulse, sample).
    from cpu np to gpu torch
    """
    
    # 1.) normalize just-in-case

    u_unit = u_axis * (1 / torch.linalg.norm(u_axis))                          
    r_unit = r_axis * (1 / torch.linalg.norm(r_axis))
    
    # 2.) Compute look direction and the cos and sin look components

    phse_cntr = 0.5 * (TxPos + RcvPos)              # midpoint.  hmmm. not sure
    P_vecs    = SRPPos - phse_cntr                  # Vector from phase center to SRP
    P_U       = torch.matmul(P_vecs, u_unit)
    P_R       = torch.matmul(P_vecs, r_unit)
    mag       = torch.linalg.norm(P_vecs, dim=-1)   # Usins 3D slant-range magnitude instead of plane
    inv_mag   = 1.0 / mag                               
    cos_theta = P_U * inv_mag
    sin_theta = P_R * inv_mag

    # 3.) Calc fasttime freqs, spatial freqs, then k_space 
    k_ndcs = torch.arange(num_samples, dtype=torch.float64, device=device)  # something
    F_hz   = Fx0.unsqueeze(1) + FxSS.unsqueeze(1) * k_ndcs.unsqueeze(0)     # F(n, k) = SC0[n] + k * SCSS[n]
    F_cpm  = 2.0 * F_hz / SPEED_OF_LIGHT                                    # cycles/meter
    K_u    = F_cpm * cos_theta.unsqueeze(1)                                 # (N_pulses, N_samples)
    K_r    = F_cpm * sin_theta.unsqueeze(1)                                 # (N_pulses, N_samples)

    return K_u, K_r

