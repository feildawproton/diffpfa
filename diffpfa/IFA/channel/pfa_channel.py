import torch
from diffpfa.IFA.channel.czt_torch import czt_resample_kspace_1d
from diffpfa.IFA.channel.nufft_torch import nufft_grid_1d

# -- NOTE (Audit C9 / F11 Remediated): RVP Deskew Removed --
# Previously, a residual video phase (RVP) deskew function was defined here:
#   if "TxFMRate" in pvp:
#       rvp_phase = pi * (F_v ** 2) / gamma
#       signal = signal * exp(j * rvp_phase)
# This was an error in understanding the CPHD data standard:
# 1. Per CPHD DIDD §1.4 and §4, CPHD phase history data in the FX domain is already
#    deskewed and SRP-referenced by the producer before writing.
# 2. 'TxFMRate' is not a defined CPHD 1.x Per-Vector Parameter (PVP).
# 3. If triggered with synthetic chirp rates, applying an additional quadratic phase
#    distorts range focus and induces target localization shifts (~0.8 m in audit a05).
# Therefore, RVP deskew has been completely removed.

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

