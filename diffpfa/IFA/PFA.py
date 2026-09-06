import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math
from scipy.fft import next_fast_len

from diffpfa.IFA.channel.pfa_channel import process_cztnufft
from diffpfa.IFA.kspace import compute_kspace

class PFAResult(tuple):
    """
    Tuple containing (img_out, bw_range, bw_azm, N_range, N_azm, is_rotated_dataset)
    with auxiliary k-space metadata and spacing attributes for downstream SICD writers.
    """
    def __new__(
        cls,
        values: tuple,
        kspace_bounds: Optional[Tuple[float, float, float, float]] = None,
        kctr_dict: Optional[Dict[str, float]] = None,
        dr_range: Optional[float] = None,
        du_azm: Optional[float] = None
    ):
        inst = super().__new__(cls, values)
        inst.kspace_bounds = kspace_bounds
        inst.kctr_dict = kctr_dict
        inst.dr_range = dr_range
        inst.du_azm = du_azm
        return inst

def _apply_ifft_and_deconv(grid: torch.Tensor, M_u: int, M_r, device: str) -> torch.Tensor:
    """
    deconv to adjust for the bell shaped taper caused by kaisser_bessel 
        - which gets rid of gridding artifacts/aliasinidfd
    """
    grid = torch.fft.ifftshift(grid)
    img  = torch.fft.ifft2(grid)
    
    # Eagerly delete the grid copy to free up VRAM before continuing!
    del grid
    
    # Out-of-place scalar multiplication to preserve autograd graph
    img = img * (M_u * M_r)            
    
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
    
    # Cast deconv to the image real dtype (e.g., float32) to prevent implicit promotion to complex128
    deconv = deconv.to(img.real.dtype)

    # Out-of-place division to preserve autograd gradient graph
    img = img / (deconv.unsqueeze(1) + 1e-12)  
    return img

def pfa_per_polar(
    channel_signals: List[Union[np.ndarray, torch.Tensor]],
    channel_pvps: List[Dict[str, np.ndarray]],
    channel_fxcs: List[float],
    channel_domain_types: List[str],
    ref_rcv_time: Optional[np.ndarray],
    cphd_meta,
    u_min: float, u_max: float, r_min: float, r_max: float,
    custom_pixel_spacing: Optional[Tuple[float, float]] = None,
    image_oversample: float = 1.0,
    batch_size: int = 256,
    return_tensor: bool = False,
    device: str = "cuda"
) -> Tuple[Union[np.ndarray, torch.Tensor], float, float, int, int, bool]:
    """
    takes in data on cpu, including numpy arrays
    allocates gpu shared kspace and image space
    allocates and copies per channel; cleans up after each channel
    copies image to cpu, cleans up
    """

    # -- 0.9) MASK VECTORS WHERE SIGNAL != 1 (Audit C9 / F12) --
    # CPHD DIDD §5.2.1 defines PVP 'SIGNAL' (1 = valid signal, 0 = filler / transmit off).
    # Any vectors flagged with SIGNAL != 1 do not contain valid radar returns and are dropped.
    filtered_signals = []
    filtered_pvps = []
    for i in range(len(channel_signals)):
        sig_i = channel_signals[i]
        pvp_i = channel_pvps[i]
        if isinstance(pvp_i, dict) and "SIGNAL" in pvp_i:
            sig_flags = np.asarray(pvp_i["SIGNAL"])
            if not np.all(sig_flags == 1):
                valid = (sig_flags == 1)
                sig_i = sig_i[valid]
                pvp_i = {
                    k: v[valid] if (isinstance(v, (np.ndarray, torch.Tensor)) and len(v) == len(sig_flags)) else v
                    for k, v in pvp_i.items()
                }
        filtered_signals.append(sig_i)
        filtered_pvps.append(pvp_i)
    channel_signals = filtered_signals
    channel_pvps = filtered_pvps

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
    denom = torch.sqrt(Ku_list[0][:, N_s//2]**2 + Kr_list[0][:, N_s//2]**2) + 1e-12
    cos_t = Ku_list[0][:, N_s//2] / denom
    sin_t = Kr_list[0][:, N_s//2] / denom
    is_rotated_dataset = bool(abs(cos_t.mean()) > abs(sin_t.mean()))

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
        if isinstance(channel_signals[i], torch.Tensor):
            sig = channel_signals[i].cfloat().to(device)
        else:
            sig = torch.from_numpy(channel_signals[i].astype(np.complex64)).cfloat().to(device)
        
        # -- NOTE (Audit C6 Remediated): CPHD SGN convention --
        # CPHD DIDD §4.3 defines phase phi(fx) = SGN * 2 * pi * fx * Delta_TOA.
        # The standard SAR imaging processor assumes SGN = -1 (echo delayed -> negative phase).
        # When Global/SGN == +1, the phase history is conjugated relative to the processor's
        # forward model. Without conjugating, the reconstructed image is point-mirrored
        # across the origin. Conjugating when SGN == +1 brings it into the standard
        # SGN = -1 convention, matching SICD Grid/Row/Sgn = -1 and Grid/Col/Sgn = -1.
        if getattr(cphd_meta, "sgn", -1) == 1:
            sig = torch.conj(sig)
        
        pvp = channel_pvps[i]
        fxc = channel_fxcs[i]
        
        # -- NOTE (Audit C1 Remediated): Phase Rotation Removed --
        # Previously, an inter-channel carrier phase rotation was applied here:
        #   tau = pvp["RcvTime"] - ref_rcv_time
        #   phase_corr = -2.0 * torch.pi * (fc_global - fxc) * tau
        #   sig = sig * torch.exp(1j * phase_corr)
        #
        # Under the mistaken assumption that stepped-chirp burst channels required
        # carrier remodulation to a global center frequency due to inter-pulse timing
        # offsets (tau = RcvTime - ref_rcv_time).
        #
        # In reality, per CPHD DIDD §1.4 and §4, CPHD phase history data in the FX domain
        # is already fully SRP-referenced and RF-frequency labeled:
        #   phi(fx) = SGN * 2 * pi * fx * Delta_TOA
        # The CPHD producer's compensation zeroes the SRP echo phase in every vector of
        # every channel. No LO or carrier modulation terms survive in compliant CPHD data.
        #
        # When applied to multi-step data with realistic burst timing delays:
        #   cycles = (fc_global - fxc) * tau
        # unless 'cycles' happens to be an exact integer (as was artificially the case
        # in early simulations with delta_tau = 150 us), this injected an uncontrolled
        # phase offset into each subband. This destroyed subband phase alignment and
        # collapsed coherence (from 0.999 down to 0.35 - 0.61).
        # Real multi-step CPHD subbands are already coherent at the SRP; therefore,
        # this rotation was spurious and has been removed.

        # -- 3.3) CZT-NUFFT PFA EACH CHANNEL (WITH ON-DEMAND OOM RECOVERY) --
        
        try:
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
                batch_size=batch_size,
                device=device
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
                batch_size=batch_size,
                device=device
            )
        
        # -- 3.4) ADD THIS CHANNEL'S KSPACE TO GLOBAL KSPACE --
        # Use out-of-place addition to preserve autograd computation graph
        if combined_grid is None:
            combined_grid = grid_2d
        else:
            combined_grid = combined_grid + grid_2d
        
        # -- 3.5) CLEAN UP PER CHANNEL (NO UNCONDITIONAL SYNC, EMPTY CACHE CONDITIONAL IN STEP 3.3) --
        del sig
        del grid_2d

    # -- 3.6) SHIFT K-SPACE GRID FOR ASYMMETRIC IMAGE AREA CENTER (Audit F7) --
    u_c = (u_min + u_max) / 2.0
    r_c = (r_min + r_max) / 2.0
    if combined_grid is not None and (abs(u_c) > 1e-9 or abs(r_c) > 1e-9):
        m_u = torch.arange(N_u, device=device, dtype=torch.float64) - N_u / 2.0
        m_r = torch.arange(N_r, device=device, dtype=torch.float64) - N_r / 2.0
        dKu = 1.0 / max(L_u, 1e-12)
        dKr = 1.0 / max(L_r, 1e-12)
        K_prime_u = (m_u * dKu).unsqueeze(1)
        K_prime_r = (m_r * dKr).unsqueeze(0)
        phase = 2.0 * math.pi * (K_prime_u * u_c + K_prime_r * r_c)
        shift_term = torch.exp(torch.complex(torch.zeros_like(phase), phase)).to(combined_grid.dtype)
        # Out-of-place multiply for autograd
        combined_grid = combined_grid * shift_term

    # -- 4.) BACK TO IMAGE SPACE WITH KAISER-BESSEL SPOTLIGHT INTENSITY CORRECTION --
    try:
        combined_img = _apply_ifft_and_deconv(combined_grid, N_u, N_r, device)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        combined_img = _apply_ifft_and_deconv(combined_grid, N_u, N_r, device)

    if return_tensor:
        img_out = combined_img
    else:
        img_cpu = combined_img.cpu().numpy().astype(np.complex64)
        del combined_img
        img_out = img_cpu

    # -- 5.) CLEAN UP FOR POLARIZATION --
    del combined_grid

    # -- 6.) ASSIGN OUTPUT BANDWIDTH AND SIZES --
    # NOTE ON ROTATED AXES (Audit Defect C5):
    # Previously, this block swapped (bw_u, bw_r) into (bw_range, bw_azm) when
    # is_rotated_dataset was True:
    #   if is_rotated_dataset:
    #       bw_range, bw_azm = bw_u, bw_r
    #       N_range, N_azm = N_u, N_r
    #   else:
    #       bw_range, bw_azm = bw_r, bw_u
    #       N_range, N_azm = N_r, N_u
    # This was a double-swap error: Step 1.1 had ALREADY swapped Ku_list and Kr_list
    # (and the spatial bounds) when is_rotated_dataset was True.
    # Therefore, Kr is ALREADY the range axis (with bandwidth bw_r and size N_r), and
    # Ku is ALREADY the azimuth axis (with bandwidth bw_u and size N_u).
    # Swapping them a second time here inverted range and azimuth bandwidths and sizes.
    # Step 6 is therefore unconditional:
    bw_range, bw_azm = bw_r, bw_u
    N_range, N_azm = N_r, N_u

    # Exact pixel spacings (Audit A3)
    dr_range = float(L_r / max(N_range, 1))
    du_azm = float(L_u / max(N_azm, 1))

    # Combined k-space bounds across all gridded channels (Audit G1)
    krg1, krg2 = float(gkr_min), float(gkr_max)
    kaz1, kaz2 = float(gku_min), float(gku_max)
    kspace_bounds = (krg1, krg2, kaz1, kaz2)
    kctr_dict = {
        "Row": float(gkr_ctr),
        "Col": float(gku_ctr)
    }

    return PFAResult(
        (img_out, bw_range, bw_azm, N_range, N_azm, is_rotated_dataset),
        kspace_bounds=kspace_bounds,
        kctr_dict=kctr_dict,
        dr_range=dr_range,
        du_azm=du_azm
    )


def run_diffpfa_tensor(
    signal: torch.Tensor,
    pvp: Dict[str, np.ndarray],
    cphd_meta,
    fxc: float,
    L: float,
    oversample: float = 1.25,
    custom_pixel_spacing: Optional[Tuple[float, float]] = None,
    device: str = "cuda"
) -> Tuple[torch.Tensor, float, float, int, int, bool]:
    """
    Tensor-native entry point for differentiable PFA imaging.
    Accepts a PyTorch Tensor signal with requires_grad=True and returns the
    complex image as a PyTorch Tensor on device, preserving the autograd computation graph.
    """
    return pfa_per_polar(
        channel_signals=[signal],
        channel_pvps=[pvp],
        channel_fxcs=[fxc],
        channel_domain_types=["FX"],
        ref_rcv_time=pvp.get("RcvTime", None) if isinstance(pvp, dict) else None,
        cphd_meta=cphd_meta,
        u_min=-L / 2.0,
        u_max=L / 2.0,
        r_min=-L / 2.0,
        r_max=L / 2.0,
        custom_pixel_spacing=custom_pixel_spacing,
        image_oversample=oversample,
        return_tensor=True,
        device=device
    )

