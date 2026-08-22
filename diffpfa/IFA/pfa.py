import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import math

from diffpfa.IFA.channel.pfa_channel import process_channel
from diffpfa.constants import SPEED_OF_LIGHT

def _compute_look_vectors(pvp: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")) -> torch.Tensor:
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

def _compute_look_components(
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

def _compute_fasttime_frequencies(
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

def _compute_kspace(
    pvp: Dict[str, np.ndarray],
    uIAX_np: np.ndarray,
    uIAY_np: np.ndarray,
    num_samples: int,
    domain_type: str = "FX",
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes spatial frequency mappings (K_u, K_r) in cycles/meter for each sample (pulse, sample).
    """
    uIAX = torch.as_tensor(uIAX_np, dtype=torch.float64, device=device)
    uIAY = torch.as_tensor(uIAY_np, dtype=torch.float64, device=device)

    P_vecs = _compute_look_vectors(pvp, device=device)
    cos_theta, sin_theta = _compute_look_components(P_vecs, uIAX, uIAY)

    F_hz = _compute_fasttime_frequencies(pvp, num_samples, domain_type=domain_type, device=device)
    F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT  # cycles/meter

    K_u = F_cpm * cos_theta.unsqueeze(1)  # (N_pulses, N_samples)
    K_r = F_cpm * sin_theta.unsqueeze(1)  # (N_pulses, N_samples)

    return K_u, K_r

def _apply_ifft_and_deconv(grid_2d: torch.Tensor, is_rotated: bool, M_u: int, M_r: int, N_u: int, N_r: int, device: str) -> torch.Tensor:
    # deconv to adjust for the bell shaped taper caused by kaisser_bessel (which gets rid of gridding artifacts/aliasinidfd
    grid_shifted = torch.fft.ifftshift(grid_2d)
    img_oversampled = torch.fft.ifft2(grid_shifted)
    img_oversampled.mul_(M_u * M_r)
    img_shifted = torch.fft.fftshift(img_oversampled)
    
    beta = 13.9086
    J = 6
    real_dtype = torch.float64
    
    if not is_rotated:
        grid_coords = (torch.arange(M_u, device=device, dtype=real_dtype) - M_u / 2.0) / M_u
        deconv = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=device)**2 - (math.pi * J * grid_coords)**2, min=1e-12)))
        img_deconv = img_shifted / (deconv.unsqueeze(1) + 1e-12)
    else:
        grid_coords = (torch.arange(M_r, device=device, dtype=real_dtype) - M_r / 2.0) / M_r
        deconv = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=device)**2 - (math.pi * J * grid_coords)**2, min=1e-12)))
        img_deconv = img_shifted / (deconv.unsqueeze(0) + 1e-12)
    
    start_u = (M_u - N_u) // 2
    start_r = (M_r - N_r) // 2
    
    patch_img = img_deconv[start_u : start_u + N_u, start_r : start_r + N_r]
    return patch_img

def pfa_per_polar(
    channel_signals: List[np.ndarray],
    channel_pvps: List[Dict[str, np.ndarray]],
    channel_fxcs: List[float],
    channel_domain_types: List[str],
    ref_rcv_time: np.ndarray,
    cphd_meta, 
    u_min: float, u_max: float, r_min: float, r_max: float,
    custom_pixel_spacing: Optional[Tuple[float, float]] = None,
    image_plane: str = "Ground",
    czt_batch_size: int = 1024,
    device: str = "cuda"
) -> Tuple[np.ndarray, float, float, int, int]:
    
    global_ku_min, global_ku_max = float('inf'), float('-inf')
    global_kr_min, global_kr_max = float('inf'), float('-inf')
    
    Ku_list = []
    Kr_list = []

    for i in range(len(channel_signals)):
        Ku_sample, Kr_sample = _compute_kspace(
            channel_pvps[i],
            cphd_meta.uIAX,
            cphd_meta.uIAY,
            channel_signals[i].shape[1],
            channel_domain_types[i],
            device=device
        )
        Ku_list.append(Ku_sample)
        Kr_list.append(Kr_sample)
        global_ku_min = min(global_ku_min, Ku_sample.min().item())
        global_ku_max = max(global_ku_max, Ku_sample.max().item())
        global_kr_min = min(global_kr_min, Kr_sample.min().item())
        global_kr_max = max(global_kr_max, Kr_sample.max().item())

    global_k_ctr_u = (global_ku_min + global_ku_max) / 2.0
    global_k_ctr_r = (global_kr_min + global_kr_max) / 2.0
    bw_u = global_ku_max - global_ku_min
    bw_r = global_kr_max - global_kr_min

    L_u = u_max - u_min
    L_r = r_max - r_min

    if custom_pixel_spacing is not None:
        du, dr = custom_pixel_spacing
    else:
        du = 1.0 / max(bw_u, 1e-6)
        dr = 1.0 / max(bw_r, 1e-6)
        
    N_u = max(16, int(np.round(L_u / du)))
    N_r = max(16, int(np.round(L_r / dr)))

    combined_grid = None
    grid_params = None

    for i in range(len(channel_signals)):
        sig = torch.from_numpy(channel_signals[i].astype(np.complex64)).cfloat().to(device) # don't move until needed
        pvp = channel_pvps[i]
        fxc = channel_fxcs[i]

        if "RcvTime" in pvp and ref_rcv_time is not None: #"RcvTime" in channel_pvps[0]:
            #tau = pvp["RcvTime"] - channel_pvps[0]["RcvTime"]
            tau = pvp["RcvTime"] - ref_rcv_time
            fc_global = (cphd_meta.global_fx_min + cphd_meta.global_fx_max) / 2.0
            tau_tensor = torch.as_tensor(tau, dtype=torch.float64, device=device)
            phase_corr = -2.0 * torch.pi * (fc_global - fxc) * tau_tensor
            corr_term = torch.exp(1j * phase_corr).unsqueeze(1)
            sig = sig * corr_term.to(sig.dtype)

        grid_2d, is_rotated, M_u, M_r = process_channel(
            signal=sig,
            pvp=pvp,
            domain_type=channel_domain_types[i],
            cphd_meta=cphd_meta,
            u_min=u_min, u_max=u_max, r_min=r_min, r_max=r_max,
            N_u=N_u, N_r=N_r,
            Ku = Ku_list[i],
            Kr = Kr_list[i],
            batch_size_pts=czt_batch_size,
            global_k_ctr_u=global_k_ctr_u,
            global_k_ctr_r=global_k_ctr_r,
            image_plane=image_plane,
            device=device
        )
        
        if combined_grid is None:
            combined_grid = grid_2d.clone()
        else:
            combined_grid.add_(grid_2d)
            
        grid_params = (is_rotated, M_u, M_r)
        
        del sig
        del grid_2d

    is_rotated, M_u, M_r = grid_params
    combined_img = _apply_ifft_and_deconv(combined_grid, is_rotated, M_u, M_r, N_u, N_r, device)
    
    img_cpu = combined_img.cpu().numpy().astype(np.complex64)
    del combined_grid
    del combined_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return img_cpu, bw_u, bw_r, N_u, N_r
