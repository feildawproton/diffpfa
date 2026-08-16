'''
stuff applying to the collection of the data that doesn't fit elsewhere (so far)
'''
from typing import Dict
import numpy as np
import torch

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

