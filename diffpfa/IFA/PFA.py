import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
from scipy.fft import next_fast_len

from diffpfa.IFA.channel.pfa_channel import process_cztnufft
from diffpfa.IFA.kspace import compute_kspace

def _apply_ifft_and_deconv(grid: torch.Tensor, M_u: int, M_r, device: str) -> torch.Tensor:
    """
    deconv to adjust for the bell shaped taper caused by kaisser_bessel 
        - which gets rid of gridding artifacts/aliasinidfd
    """
    grid = torch.fft.ifftshift(grid)
    img  = torch.fft.ifft2(grid)
    
    # Eagerly delete the grid copy to free up VRAM before continuing!
    del grid
    
    # Do the scalar multiplication IN-PLACE using mul_
    img.mul_(M_u * M_r)            
    
    # fftshift creates a new tensor, so re-assign and delete the old one implicitly
    img = torch.fft.fftshift(img)
    
    beta = 13.9086
    J = 6
    real_dtype = torch.float64
    
    beta_tensor = torch.tensor(beta, dtype=real_dtype, device=device)
    grid_coords = (torch.arange(M_u, device=device, dtype=real_dtype) - M_u / 2.0) / M_u
    z2 = beta_tensor**2 - (math.pi * J * grid_coords)**2
    sqrt_pos = torch.sqrt(torch.clamp(z2, min=1e-12))
    sqrt_neg = torch.sqrt(torch.clamp(-z2, min=1e-12))
    deconv = torch.where(
        z2 >= 0,
        torch.sinh(sqrt_pos) / sqrt_pos,
        torch.sin(sqrt_neg) / sqrt_neg
    )
    deconv = deconv / (torch.sinh(beta_tensor) / beta_tensor)
    
    # Divide IN-PLACE to save allocating another full-size image tensor
    img.div_(deconv.unsqueeze(1) + 1e-12)  
    return img

def pfa_per_polar(
    channel_signals: List[np.ndarray],
    channel_pvps: List[Dict[str, np.ndarray]],
    channel_fxcs: List[float],
    channel_domain_types: List[str],
    ref_rcv_time: np.ndarray,
    cphd_meta, 
    u_min: float, u_max: float, r_min: float, r_max: float,
    custom_pixel_spacing: Optional[Tuple[float, float]] = None,
    image_oversample: float = 1.0,
    image_plane: str = "Ground",
    czt_batch_size: int = 1024,
    device: str = "cuda"
) -> Tuple[torch.Tensor, float, float, int, int]:
    """
    takes in data on cpu, including numpy arrays
    allocates gpu shared kspace and image space
    allocates and copies per channel; cleans up after each channel
    copies image to cpu, cleans up
    """

    # -- 1.) CALCULATE PER CHANNEL AND GLOBAL KSPACES --
    
    Ku_list = []                                    # per channel
    Kr_list = []
    for i in range(len(channel_signals)):
        Ku_chnnl, Kr_chnnl = compute_kspace(
            channel_pvps[i],
            cphd_meta.uIAX,
            cphd_meta.uIAY,
            channel_signals[i].shape[1],
            channel_domain_types[i],
            device=device
        ) 
        Ku_list.append(Ku_chnnl)
        Kr_list.append(Kr_chnnl)
        
    # -- 1.1) SOMETIMES THE GROUND AXES ARE FLIPPED FROM HOW WE'D EXPECT FOR ASSIGNING RANGE->k_R, AZM->k_U --
    
    N_s = Ku_list[0].shape[1]
    cos_t = Ku_list[0][:, N_s//2] / (torch.sqrt(Ku_list[0][:, N_s//2]**2 + Kr_list[0][:, N_s//2]**2) + 1e-12)
    sin_t = Kr_list[0][:, N_s//2] / (torch.sqrt(Ku_list[0][:, N_s//2]**2 + Kr_list[0][:, N_s//2]**2) + 1e-12)
    is_rotated_dataset = abs(cos_t.mean()) > abs(sin_t.mean())

    if is_rotated_dataset:
        print("Data is rotated compared to what PFA expects. Swapping internal axes for processing...")
        Ku_list, Kr_list = Kr_list, Ku_list
        u_min, r_min = r_min, u_min
        u_max, r_max = r_max, u_max
        if custom_pixel_spacing is not None:
            custom_pixel_spacing = (custom_pixel_spacing[1], custom_pixel_spacing[0])

    # -- 1.2) CALC GLOBALS --
    
    gku_min, gku_max = float('inf'), float('-inf')
    gkr_min, gkr_max = float('inf'), float('-inf')
    for Ku_chnnl, Kr_chnnl in zip(Ku_list, Kr_list):
        gku_min = min(gku_min, Ku_chnnl.min().item())
        gku_max = max(gku_max, Ku_chnnl.max().item())
        gkr_min = min(gkr_min, Kr_chnnl.min().item())
        gkr_max = max(gkr_max, Kr_chnnl.max().item())

    gku_ctr = (gku_min + gku_max) / 2.0
    gkr_ctr = (gkr_min + gkr_max) / 2.0
    bw_u = gku_max - gku_min
    bw_r = gkr_max - gkr_min

    # -- 2.) CALCULATE GRID DIMS FROM IMAGE EXTENTS AND SAMPLE SPACING --

    L_u = u_max - u_min
    L_r = r_max - r_min

    if custom_pixel_spacing is not None:
        du, dr = custom_pixel_spacing
    else:
        du = 1.0 / (max(bw_u, 1e-6) * image_oversample)
        dr = 1.0 / (max(bw_r, 1e-6) * image_oversample)
        
    N_u = next_fast_len(int(np.round(L_u / du))) # be kind to FFTs
    N_r = next_fast_len(int(np.round(L_r / dr)))

    # -- 3.) HOLD GLOBAL RESULTS AND PROCESS PER CHANNEL -- 
    
    combined_grid = None
    grid_params   = None
    
    for i in range(len(channel_signals)):
        
        # -- 3.1) ALLOCATE AND COPY CHANNEL -- 
         
        sig = torch.from_numpy(channel_signals[i].astype(np.complex64)).cfloat().to(device) # don't move until needed
        pvp = channel_pvps[i]
        fxc = channel_fxcs[i]
        
        # -- 3.2) CALC AND APPLY PHASE CORRECTION FOR STEP CENTER FREQUENCIES
        
        tau        = pvp["RcvTime"] - ref_rcv_time
        fc_global  = (cphd_meta.global_fx_min + cphd_meta.global_fx_max) / 2.0
        tau_tensor = torch.as_tensor(tau, dtype=torch.float64, device=device)
        phase_corr = -2.0 * torch.pi * (fc_global - fxc) * tau_tensor
        corr_term  = torch.exp(1j * phase_corr).unsqueeze(1)
        sig        = sig * corr_term.to(sig.dtype)

        # -- 3.3) CZT-NUFFT PFA EACH CHANNEL --
        
        grid_2d = process_cztnufft(
            signal=sig,
            fxc=fxc,
            pvp=pvp,
            Ku = Ku_list[i],
            Kr = Kr_list[i],
            N_u=N_u, 
            N_r=N_r,
            L_u=L_u,
            L_r=L_r,
            k_ctr_u=gku_ctr,
            k_ctr_r=gkr_ctr,
            czt_batch_size=czt_batch_size,
            device=device
        )
        
        # -- 3.4) ADD THIS CHANNEL'S KSPACE TO GLOBAL KSPACE --

        if combined_grid is None:
            combined_grid = grid_2d.clone() # allocate on first channel
        else:
            combined_grid.add_(grid_2d)     # add on subsequent channels
        
        # -- 3.4) CLEAN UP PER CHANNEL        
        
        del sig
        del grid_2d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # -- 4.) BACK TO IMAGE SPACE WITH KAISSER-BESSEL SPOTLIGHT INTENSITY CORRECTION --
    
    combined_img = _apply_ifft_and_deconv(combined_grid, N_u, N_r, device)
    img_cpu = combined_img.cpu().numpy().astype(np.complex64)
    
    # -- 5.) CLEAN UP FOR POLARIZATION --
    
    del combined_grid
    del combined_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- 6.) TRANSPOSE TO (RANGE, AZIMUTH) FOR SICD ROW/COL MAPPING --
    if is_rotated_dataset:
        bw_range, bw_azm = bw_u, bw_r
        N_range, N_azm = N_u, N_r
    else:
        bw_range, bw_azm = bw_r, bw_u
        N_range, N_azm = N_r, N_u

    return img_cpu, bw_range, bw_azm, N_range, N_azm

