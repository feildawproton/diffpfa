from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
from scipy.fft import next_fast_len
import math

from diffpfa.IFA.channel.czt_torch import czt_resample_kspace_1d
from diffpfa.IFA.channel.nufft_torch import nufft_grid_1d

from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.types import CPHDMetadata

def _deskew_rvp(signal: torch.Tensor, fxc: float, pvp: dict, N_samples: int, device: str) -> torch.Tensor:
    """
    remove residual video phase (RVP) caused by stretch processing (deramping)
    phase_rvp = pi*(F^2 / gamma)
    gamma is the chirp rate
    """
    if "TxFMRate" in pvp:
        gamma = torch.as_tensor(pvp["TxFMRate"], dtype=torch.float64, device=device)
        # only move forward if we truly are stretch processing gamma > 0
        if torch.any(torch.abs(gamma) > 1e-12) and "SC0" in pvp and "SCSS" in pvp:
            sc0 = torch.as_tensor(pvp["SC0"], dtype=torch.float64, device=device)
            scss = torch.as_tensor(pvp["SCSS"], dtype=torch.float64, device=device)
            k_idx = torch.arange(N_samples, dtype=torch.float64, device=device)
            F_hz = sc0.unsqueeze(1) + scss.unsqueeze(1) * k_idx.unsqueeze(0)
            F_v = F_hz - fxc
            rvp_phase = torch.pi * (F_v ** 2) / gamma.unsqueeze(1)
            rvp_term = torch.exp(torch.complex(torch.zeros_like(rvp_phase), rvp_phase))
            signal = signal * rvp_term.to(signal.dtype)
    return signal

def process_cztnufft(
    signal: torch.Tensor,
    fxc: float,
    pvp: dict,
    Ku: torch.Tensor,
    Kr: torch.Tensor,
    N_u: int,
    N_r: int,
    L_u: float,
    L_r: float,
    k_ctr_u: float,
    k_ctr_r: float,
    batch_size: int,
    device: str
) -> torch.Tensor:

    N_samples = signal.shape[-1]
    
    signal = _deskew_rvp(signal, fxc, pvp, N_samples, device)
    
    dK_u = 1 / L_u
    dK_r = 1 / L_r

    k_out_start_r = k_ctr_r - (N_r / 2.0) * dK_r
    k_out_step_r = dK_r
    
    k_start = Kr[:, 0].unsqueeze(1)
    k_step = ((Kr[:, -1] - Kr[:, 0]) / max(N_samples - 1, 1)).unsqueeze(1)

    fast_resampled = czt_resample_kspace_1d(
        signal,
        k_start=k_start,
        k_step=k_step,
        M_out=N_r,
        k_out_start=k_out_start_r,
        k_out_step=k_out_step_r,
        spatial_extent=L_r,
        batch_size=batch_size,
    )

    cot_theta = Ku[:, N_samples//2] / Kr[:, N_samples//2]
    m_idx = torch.arange(N_r, device=device, dtype=torch.float64)
    Kr_cart = k_out_start_r + m_idx * dK_r

    grid_2d = nufft_grid_1d(
        signal=fast_resampled,
        kx=(cot_theta, Kr_cart),
        grid_size=N_u,
        L_x=L_u,
        k_center=k_ctr_u,
        batch_size=batch_size,
    )
    return grid_2d

