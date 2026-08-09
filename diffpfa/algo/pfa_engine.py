from scipy.fft import next_fast_len
from dataclasses import dataclass
import os
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from diffpfa.algo.channel_combine import align_and_combine_channels
from diffpfa.algo.czt_torch import czt_1d_torch, czt_resample_kspace_1d
from diffpfa.algo.kspace import (
    compute_fasttime_frequencies,
    compute_kspace,
    compute_look_components,
    compute_look_vectors,
)
from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.algo.nufft_torch import nufft_1d_type1_torch, nufft_2d_type1_torch, nufft_grid_1d
from diffpfa.utils.timer import log_time, Timer
from diffpfa.io.base import (
    BaseCPHDReader,
    BaseSICDWriter,
    CPHDChannelData,
    CPHDMetadata,
    SICDImagePayload,
)


@dataclass
class PFAConfig:
    """Configuration settings for PFA processing."""
    mode: str = "cztnufft"  # "cztnufft", "nufft"
    image_area_mode: str = "ImageArea"  # "ImageArea", "ExtendedArea", "InscribedRectangle", "TargetPixelSpacing"
    image_plane: str = "Ground"  # "Ground" or "Slant"
    custom_pixel_spacing: Optional[Tuple[float, float]] = None  # (du, dr) in meters
    custom_image_area: Optional[Tuple[float, float, float, float]] = None  # (u_min, u_max, r_min, r_max)
    align_subchannels: bool = False
    debug_save_channels: bool = False
    output_dir: str = "output"
    device: str = "cpu"
    nufft_batch_size_pts: int = 1_000_000
    czt_batch_size: int = 500
    enable_compile: bool = False
    num_subpatches: int = 1  # 1 for global PFA (1x1 grid), > 1 for localized wavefront correction (e.g., 2 for 2x2 grid)

class PFAEngine:
    """Core PyTorch Polar Format Algorithm Processor Engine."""

    def __init__(
        self,
        reader: BaseCPHDReader,
        writer: BaseSICDWriter,
        config: Optional[PFAConfig] = None
    ):
        self.reader = reader
        self.writer = writer
        self.config = config or PFAConfig()
        self.device = torch.device(self.config.device)
        
        # JIT compile hot paths with torch.compile if requested
        if self.config.enable_compile:
            self._czt_fn = torch.compile(czt_resample_kspace_1d)
            self._nufft_1d_fn = torch.compile(nufft_1d_type1_torch)
            self._nufft_2d_fn = torch.compile(nufft_2d_type1_torch)
        else:
            self._czt_fn = czt_resample_kspace_1d
            self._nufft_1d_fn = nufft_1d_type1_torch
            self._nufft_2d_fn = nufft_2d_type1_torch

    def _determine_spatial_bounds(
        self,
        cphd_meta: CPHDMetadata,
        sample_ku: torch.Tensor,
        sample_kr: torch.Tensor
    ) -> Tuple[float, float, float, float, int, int]:
        """
        Determines spatial image bounds (u_min, u_max, r_min, r_max) and grid counts (N_u, N_r).
        """
        # Spatial bandwidth B_u = max(Ku) - min(Ku), B_r = max(Kr) - min(Kr)
        b_u = (sample_ku.max() - sample_ku.min()).item()
        b_r = (sample_kr.max() - sample_kr.min()).item()
        native_du = 1.0 / max(b_u, 1e-6)
        native_dr = 1.0 / max(b_r, 1e-6)

        mode = self.config.image_area_mode

        if mode == "ImageArea" and cphd_meta.image_area is not None:
            ia = cphd_meta.image_area
            u_min, u_max = min(ia.x1, ia.x2), max(ia.x1, ia.x2)
            r_min, r_max = min(ia.y1, ia.y2), max(ia.y1, ia.y2)
        elif mode == "ExtendedArea" and cphd_meta.extended_area is not None:
            ea = cphd_meta.extended_area
            u_min, u_max = min(ea.x1, ea.x2), max(ea.x1, ea.x2)
            r_min, r_max = min(ea.y1, ea.y2), max(ea.y1, ea.y2)
        else:
            # Fallback bounds derived from K-space support and full sample count
            max_extent_u = native_du * sample_ku.shape[0]
            max_extent_r = native_dr * sample_kr.shape[1]
            u_min, u_max = -max_extent_u / 2.0, max_extent_u / 2.0
            r_min, r_max = -max_extent_r / 2.0, max_extent_r / 2.0

        # Override with custom image area if provided
        if self.config.custom_image_area is not None:
            u_min, u_max, r_min, r_max = self.config.custom_image_area

        L_u = u_max - u_min
        L_r = r_max - r_min

        # Compute grid counts (N_u, N_r)
        if self.config.custom_pixel_spacing is not None:
            du, dr = self.config.custom_pixel_spacing
            N_u = int(np.round(L_u / du))
            N_r = int(np.round(L_r / dr))
        elif cphd_meta.line_spacing is not None and cphd_meta.sample_spacing is not None:
            N_u = int(np.round(L_u / cphd_meta.line_spacing))
            N_r = int(np.round(L_r / cphd_meta.sample_spacing))
        else:
            # Match native bandwidth pixel spacing
            N_u = int(np.round(L_u / native_du))
            N_r = int(np.round(L_r / native_dr))

        N_u = max(N_u, 16)
        N_r = max(N_r, 16)

        return u_min, u_max, r_min, r_max, N_u, N_r



    def _deskew_rvp(self, signal: torch.Tensor, pvp: dict, N_samples: int) -> torch.Tensor:
        if "TxFMRate" in pvp:
            gamma = torch.as_tensor(pvp["TxFMRate"], dtype=torch.float64, device=self.device)
            if "SC0" in pvp and "SCSS" in pvp:
                sc0 = torch.as_tensor(pvp["SC0"], dtype=torch.float64, device=self.device)
                scss = torch.as_tensor(pvp["SCSS"], dtype=torch.float64, device=self.device)
                k_idx = torch.arange(N_samples, dtype=torch.float64, device=self.device)
                F_hz = sc0.unsqueeze(1) + scss.unsqueeze(1) * k_idx.unsqueeze(0)
                rvp_phase = torch.pi * (F_hz ** 2) / gamma.unsqueeze(1)
                rvp_term = torch.exp(torch.complex(torch.zeros_like(rvp_phase), rvp_phase))
                signal = signal * rvp_term.to(signal.dtype)
        return signal

    def _get_image_plane_vectors(self, cphd_meta: CPHDMetadata) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.image_plane == "Slant":
            srp = cphd_meta.srp_ecf
            arp = cphd_meta.arp_pos_coa
            arp_v = cphd_meta.arp_vel_coa
            p_vec = srp - arp
            u_row = p_vec / np.linalg.norm(p_vec)
            u_v = arp_v / np.linalg.norm(arp_v)
            u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
            u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
            uIAX = torch.as_tensor(u_col, dtype=torch.float64, device=self.device)
            uIAY = torch.as_tensor(u_row, dtype=torch.float64, device=self.device)
        else:
            uIAX = torch.as_tensor(cphd_meta.uIAX, dtype=torch.float64, device=self.device)
            uIAY = torch.as_tensor(cphd_meta.uIAY, dtype=torch.float64, device=self.device)
        return uIAX, uIAY
        
    def _apply_scp_shift(self, signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY):
        P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
        R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
        R_patch = torch.linalg.norm(P_patch, dim=-1)
        dR = R_patch - R_orig
        phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
        corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
        sig_patch = signal * corr_term.to(signal.dtype)
        return sig_patch, P_patch


    @log_time("process_channel_czt_nufft")
    def process_channel_czt_nufft(
        self,
        ch_data: CPHDChannelData,
        cphd_meta: CPHDMetadata,
        u_min: float,
        u_max: float,
        r_min: float,
        r_max: float,
        N_u: int,
        N_r: int,
        global_k_ctr_u: float = None,
        global_k_ctr_r: float = None,
        return_kspace: bool = False
    ) -> torch.Tensor:
        """Processes a single CPHD channel using a CZTNUFFT approach: 1D CZT in Range, 1D NUFFT in Cross-Range."""
        # Pin memory for async H2D copy
        if self.device.type == "cuda" and ch_data.signal.device.type == "cpu":
            ch_data.signal = ch_data.signal.pin_memory()
            
        pvp = ch_data.pvp
        N_pulses, N_samples = ch_data.signal.shape

        num_patches = max(1, self.config.num_subpatches)
        subpatch_size_u = int(np.ceil(N_u / num_patches))
        subpatch_size_r = int(np.ceil(N_r / num_patches))

        image_2d = torch.zeros((N_u, N_r), dtype=torch.complex64, device=self.device)
        
        P_vecs_orig = compute_look_vectors(pvp, device=self.device)
        F_hz_full = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=self.device)
        
        # Precompute RVP params if necessary
        has_rvp = "TxFMRate" in pvp and "SC0" in pvp and "SCSS" in pvp
        if has_rvp:
            gamma_full = torch.as_tensor(pvp["TxFMRate"], dtype=torch.float64, device=self.device)
        
        uIAX, uIAY = self._get_image_plane_vectors(cphd_meta)

        u_edges = np.linspace(u_min, u_max, N_u + 1)
        r_edges = np.linspace(r_min, r_max, N_r + 1)
        
        czt_batch_size = self.config.czt_batch_size
        
        stream_compute = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        stream_copy = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

        for i_u in range(0, N_u, subpatch_size_u):
            for i_r in range(0, N_r, subpatch_size_r):
                end_u = min(i_u + subpatch_size_u, N_u)
                end_r = min(i_r + subpatch_size_r, N_r)
                
                p_u_min, p_u_max = u_edges[i_u], u_edges[end_u]
                p_r_min, p_r_max = r_edges[i_r], r_edges[end_r]
                
                u_c = (p_u_min + p_u_max) / 2.0
                r_c = (p_r_min + p_r_max) / 2.0
                
                P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
                R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
                R_patch = torch.linalg.norm(P_patch, dim=-1)
                dR_full = R_patch - R_orig
                
                cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
                F_cpm = 2.0 * F_hz_full / SPEED_OF_LIGHT
                Ku = F_cpm * cos_theta.unsqueeze(1)
                Kr = F_cpm * sin_theta.unsqueeze(1)
                
                local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
                local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
                
                p_N_u, p_N_r = end_u - i_u, end_r - i_r
                
                oversample = 1.5
                M_u = next_fast_len(int(math.ceil(p_N_u * oversample)))
                M_r = next_fast_len(int(math.ceil(p_N_r * oversample)))
                L_u = local_u_max - local_u_min
                L_r = local_r_max - local_r_min
                dK_u = p_N_u / (M_u * max(L_u, 1e-12))
                dK_r = p_N_r / (M_r * max(L_r, 1e-12))
                k_ctr_u = global_k_ctr_u if global_k_ctr_u is not None else (Ku.min() + Ku.max()).item() / 2.0
                k_ctr_r = global_k_ctr_r if global_k_ctr_r is not None else (Kr.min() + Kr.max()).item() / 2.0
                
                # 1. Range CZT Resampling (K-space -> Cartesian K_r)
                k_out_start_r = k_ctr_r - (M_r / 2.0) * dK_r
                k_out_step_r = dK_r
                
                range_resampled = torch.zeros((N_pulses, M_r), dtype=torch.complex64, device=self.device)
                
                def get_batch(b_idx):
                    if b_idx >= N_pulses: return None
                    b_e = min(b_idx + czt_batch_size, N_pulses)
                    kwargs = {"non_blocking": True} if self.device.type == "cuda" else {}
                    return ch_data.signal[b_idx:b_e].to(self.device, **kwargs), b_idx, b_e

                if stream_copy:
                    with torch.cuda.stream(stream_copy):
                        next_batch_info = get_batch(0)
                else:
                    next_batch_info = get_batch(0)
                
                for b in range(0, N_pulses, czt_batch_size):
                    if stream_compute and stream_copy:
                        stream_compute.wait_stream(stream_copy)
                        
                    sig_gpu, b_start, b_end = next_batch_info
                    
                    if stream_copy and b + czt_batch_size < N_pulses:
                        with torch.cuda.stream(stream_copy):
                            next_batch_info = get_batch(b + czt_batch_size)
                    elif not stream_copy and b + czt_batch_size < N_pulses:
                        next_batch_info = get_batch(b + czt_batch_size)
                            
                    with torch.cuda.stream(stream_compute) if stream_compute else torch.autograd.profiler.record_function("cpu_compute"):
                        F_hz_b = F_hz_full[b_start:b_end]
                        
                        # Deskew RVP for batch
                        if has_rvp:
                            gamma_b = gamma_full[b_start:b_end]
                            rvp_phase = torch.pi * (F_hz_b ** 2) / gamma_b.unsqueeze(1)
                            rvp_term = torch.exp(torch.complex(torch.zeros_like(rvp_phase), rvp_phase))
                            sig_gpu = sig_gpu * rvp_term.to(sig_gpu.dtype)
                            
                        # Apply SCP Shift for batch
                        dR_b = dR_full[b_start:b_end]
                        phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz_b * dR_b.unsqueeze(1)
                        corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
                        sig_patch_b = sig_gpu * corr_term.to(sig_gpu.dtype)
                        
                        # Apply CZT for batch
                        kr_b = Kr[b_start:b_end]
                        k_start = kr_b[:, 0].unsqueeze(1)
                        k_step = ((kr_b[:, -1] - kr_b[:, 0]) / max(N_samples - 1, 1)).unsqueeze(1)
                        
                        batch_out = self._czt_fn(
                            sig_patch_b, 
                            k_start=k_start, 
                            k_step=k_step,
                            M_out=M_r,
                            k_out_start=k_out_start_r,
                            k_out_step=k_out_step_r,
                            spatial_extent=L_r,
                            oversample=oversample
                        )
                        range_resampled[b_start:b_end, :] = batch_out

                if stream_compute:
                    torch.cuda.current_stream(device=self.device).wait_stream(stream_compute)
                
                # 2. Cross-Range NUFFT Gridding (Cartesian K_r -> Cartesian K_u)
                cot_theta = Ku[:, N_samples//2] / Kr[:, N_samples//2]
                m_idx = torch.arange(M_r, device=self.device, dtype=torch.float64)
                Kr_cart = k_out_start_r + m_idx * dK_r
                
                del Ku, Kr, F_hz_full, dR_full
                
                nufft_batch_size = self.config.czt_batch_size
                grid_2d = torch.zeros((M_u, M_r), dtype=range_resampled.dtype, device=self.device)
                
                for b in range(0, M_r, nufft_batch_size):
                    chunk_signal = range_resampled[:, b:b + nufft_batch_size]
                    chunk_Kr_cart = Kr_cart[b:b + nufft_batch_size]
                    chunk_Ku_cart = cot_theta.unsqueeze(1) * chunk_Kr_cart.unsqueeze(0)
                    
                    chunk_grid = nufft_grid_1d(
                        signal=chunk_signal,
                        kx=chunk_Ku_cart,
                        grid_size=p_N_u,
                        L_x=L_u,
                        k_center=k_ctr_u,
                        oversample=oversample
                    )
                    grid_2d[:, b:b+nufft_batch_size] = chunk_grid
                
                del range_resampled
                
                # 3. 2D IFFT and Deconvolution
                if return_kspace:
                    if num_patches > 1:
                        raise ValueError("Cannot return kspace when num_subpatches > 1")
                    return grid_2d
                    
                grid_shifted = torch.fft.ifftshift(grid_2d)
                del grid_2d
                img_oversampled = torch.fft.ifft2(grid_shifted)
                del grid_shifted
                img_oversampled.mul_(M_u * M_r)
                img_shifted = torch.fft.fftshift(img_oversampled)
                del img_oversampled
                
                beta = 13.9086
                J = 6
                real_dtype = torch.float64
                grid_u_coords = (torch.arange(M_u, device=self.device, dtype=real_dtype) - M_u / 2.0) / M_u
                deconv_u = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=self.device)**2 - (math.pi * J * grid_u_coords)**2, min=1e-12)))
                
                img_deconv = img_shifted / (deconv_u.unsqueeze(1) + 1e-12)
                
                start_u = (M_u - p_N_u) // 2
                start_r = (M_r - p_N_r) // 2
                
                patch_img = img_deconv[start_u : start_u + p_N_u, start_r : start_r + p_N_r]
                image_2d[i_u:end_u, i_r:end_r] = patch_img

        return image_2d



    @log_time("process_channel_nufft")
    def process_channel_nufft(
        self,
        ch_data: CPHDChannelData,
        cphd_meta: CPHDMetadata,
        u_min: float,
        u_max: float,
        r_min: float,
        r_max: float,
        N_u: int,
        N_r: int,
        global_k_ctr_u: float = None,
        global_k_ctr_r: float = None
    ) -> torch.Tensor:
        """Processes a single CPHD channel using 2D PyTorch NUFFT gridding."""
        signal = ch_data.signal.to(self.device)
        pvp = ch_data.pvp
        N_pulses, N_samples = signal.shape

        signal = self._deskew_rvp(signal, pvp, N_samples)

        num_patches = max(1, self.config.num_subpatches)
        subpatch_size_u = int(np.ceil(N_u / num_patches))
        subpatch_size_r = int(np.ceil(N_r / num_patches))

        P_vecs_orig = compute_look_vectors(pvp, device=self.device)
        F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=self.device)
        
        uIAX, uIAY = self._get_image_plane_vectors(cphd_meta)

        u_edges = np.linspace(u_min, u_max, N_u + 1)
        r_edges = np.linspace(r_min, r_max, N_r + 1)

        image_rows = []
        for i_u in range(0, N_u, subpatch_size_u):
            row_patches = []
            for i_r in range(0, N_r, subpatch_size_r):
                end_u = min(i_u + subpatch_size_u, N_u)
                end_r = min(i_r + subpatch_size_r, N_r)
                
                p_u_min, p_u_max = u_edges[i_u], u_edges[end_u]
                p_r_min, p_r_max = r_edges[i_r], r_edges[end_r]
                
                u_c = (p_u_min + p_u_max) / 2.0
                r_c = (p_r_min + p_r_max) / 2.0
                
                sig_patch, P_patch = self._apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY)
                
                cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
                F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
                Ku = F_cpm * cos_theta.unsqueeze(1)
                Kr = F_cpm * sin_theta.unsqueeze(1)
                
                if global_k_ctr_u is not None:
                    Ku = Ku - global_k_ctr_u
                if global_k_ctr_r is not None:
                    Kr = Kr - global_k_ctr_r
                
                local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
                local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
                p_N_u, p_N_r = end_u - i_u, end_r - i_r
                
                patch_img = self._nufft_2d_fn(
                    signal=sig_patch,
                    ku=Ku,
                    kr=Kr,
                    grid_size_u=p_N_u,
                    grid_size_r=p_N_r,
                    u_min=local_u_min,
                    u_max=local_u_max,
                    r_min=local_r_min,
                    r_max=local_r_max,
                    batch_size_pts=self.config.nufft_batch_size_pts
                )
                row_patches.append(patch_img)
            image_rows.append(torch.cat(row_patches, dim=1))

        image_2d = torch.cat(image_rows, dim=0)

        return image_2d


    def run(self) -> List[str]:
        """Runs the complete PFA processing pipeline."""
        cphd_meta = self.reader.get_metadata()
        channel_names = self.reader.get_channel_names()

        # Group channels by polarization pair: (tx_pol, rcv_pol) -> List[CPHDChannelData]
        pol_groups: Dict[Tuple[str, str], List[CPHDChannelData]] = {}

        with Timer("CPHD Read & Channel Grouping"):
            from concurrent.futures import ThreadPoolExecutor
            
            def _load_ch(name):
                return self.reader.read_channel(name)
                
            max_workers = max(1, len(channel_names))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                loaded_channels = list(executor.map(_load_ch, channel_names))
                
            for ch_data in loaded_channels:
                pol_key = (ch_data.tx_pol, ch_data.rcv_pol)
                if pol_key not in pol_groups:
                    pol_groups[pol_key] = []
                pol_groups[pol_key].append(ch_data)

        output_files = []
        os.makedirs(self.config.output_dir, exist_ok=True)

        for pol_key, ch_list in pol_groups.items():
            tx_pol, rcv_pol = pol_key

            # Determine global spatial bounds and center frequencies from all channels
            global_ku_min, global_ku_max = float('inf'), float('-inf')
            global_kr_min, global_kr_max = float('inf'), float('-inf')
            
            for ch_data in ch_list:
                Ku_sample, Kr_sample = compute_kspace(
                    ch_data.pvp,
                    cphd_meta.uIAX,
                    cphd_meta.uIAY,
                    ch_data.signal.shape[1],
                    ch_data.domain_type,
                    device=self.device
                )
                global_ku_min = min(global_ku_min, Ku_sample.min().item())
                global_ku_max = max(global_ku_max, Ku_sample.max().item())
                global_kr_min = min(global_kr_min, Kr_sample.min().item())
                global_kr_max = max(global_kr_max, Kr_sample.max().item())
            
            first_ch = ch_list[0]
            N_pulses = first_ch.signal.shape[0]
            N_samples = first_ch.signal.shape[1]
            dummy_ku = torch.full((N_pulses, N_samples), global_ku_min, device=self.device, dtype=torch.float64)
            dummy_kr = torch.full((N_pulses, N_samples), global_kr_min, device=self.device, dtype=torch.float64)
            dummy_ku[0, 0] = global_ku_max
            dummy_kr[0, 0] = global_kr_max
            
            u_min, u_max, r_min, r_max, N_u, N_r = self._determine_spatial_bounds(
                cphd_meta, dummy_ku, dummy_kr
            )
            
            global_k_ctr_u = (global_ku_min + global_ku_max) / 2.0
            global_k_ctr_r = (global_kr_min + global_kr_max) / 2.0

            # Process each channel in group
            channel_images = []

            for ch_data in ch_list:
                # Strict mode selection
                mode = self.config.mode
                combine_in_kspace = (self.config.num_subpatches == 1) and (mode == "cztnufft")
                
                if mode == "nufft":
                    img = self.process_channel_nufft(ch_data, cphd_meta, u_min, u_max, r_min, r_max, N_u, N_r, global_k_ctr_u, global_k_ctr_r)
                elif mode == "cztnufft":
                    img = self.process_channel_czt_nufft(ch_data, cphd_meta, u_min, u_max, r_min, r_max, N_u, N_r, global_k_ctr_u, global_k_ctr_r, return_kspace=combine_in_kspace)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                channel_images.append(img)

                # Export debug channel image if requested
                if self.config.debug_save_channels:
                    # If returning kspace, we'd need to IFFT it to save it. For simplicity, we skip debug save when combine_in_kspace is True
                    if combine_in_kspace:
                        pass # Skipping debug channels when combining in K-space
                    else:
                        dbg_name = f"SICD_U_{tx_pol}_{rcv_pol}_ch_{ch_data.identifier}.nitf"
                        dbg_path = os.path.join(self.config.output_dir, dbg_name)
    
                        du = (u_max - u_min) / max(N_u, 1)
                        dr = (r_max - r_min) / max(N_r, 1)
                        local_Ku, local_Kr = compute_kspace(
                            ch_data.pvp, cphd_meta.uIAX, cphd_meta.uIAY, ch_data.signal.shape[1], ch_data.domain_type, device=self.device
                        )
                        bw_u = (local_Ku.max() - local_Ku.min()).item()
                        bw_r = (local_Kr.max() - local_Kr.min()).item()
    
                        payload = SICDImagePayload(
                            complex_image=img,
                            tx_pol=tx_pol,
                            rcv_pol=rcv_pol,
                            uIAX=cphd_meta.uIAX,
                            uIAY=cphd_meta.uIAY,
                            iarp_ecf=cphd_meta.iarp_ecf,
                            line_spacing=du,
                            sample_spacing=dr,
                            first_line=u_min,
                            first_sample=r_min,
                            center_freq=(cphd_meta.global_fx_min + cphd_meta.global_fx_max) / 2.0,
                            bandwidth_u=bw_u,
                            bandwidth_r=bw_r,
                            channel_id=ch_data.identifier,
                        )
    
                        saved_path = self.writer.write_sicd(dbg_path, payload, cphd_meta)
                        output_files.append(saved_path)

            # Combine sub-channels
            with Timer("Combine Sub-channels"):
                combined_img, _ = align_and_combine_channels(
                    channel_images,
                    align_phase=self.config.align_subchannels
                )
                
                if combine_in_kspace:
                    # combined_img is currently in K-space. We must 2D IFFT it.
                    with torch.autograd.profiler.record_function("combined_ifft2"):
                        combined_img = combined_img.contiguous()
                        grid_shifted = torch.fft.ifftshift(combined_img)
                        del combined_img
                        img_oversampled = torch.fft.ifft2(grid_shifted)
                        del grid_shifted
                        
                        M_u, M_r = img_oversampled.shape
                        img_oversampled.mul_(M_u * M_r)
                        img_shifted = torch.fft.fftshift(img_oversampled)
                        del img_oversampled
                        
                        beta = 13.9086
                        J = 6
                        real_dtype = torch.float64
                        grid_u_coords = (torch.arange(M_u, device=self.device, dtype=real_dtype) - M_u / 2.0) / M_u
                        deconv_u = torch.i0(torch.sqrt(torch.clamp(torch.tensor(beta, dtype=real_dtype, device=self.device)**2 - (math.pi * J * grid_u_coords)**2, min=1e-12)))
                        
                        img_deconv = img_shifted / (deconv_u.unsqueeze(1) + 1e-12)
                        
                        start_u = (M_u - N_u) // 2
                        start_r = (M_r - N_r) // 2
                        
                        combined_img = img_deconv[start_u : start_u + N_u, start_r : start_r + N_r]

            # Write primary combined SICD-U file
            out_name = f"SICD_U_{tx_pol}_{rcv_pol}.nitf"
            out_path = os.path.join(self.config.output_dir, out_name)

            du = (u_max - u_min) / max(N_u, 1)
            dr = (r_max - r_min) / max(N_r, 1)
            bw_u = (global_ku_max - global_ku_min)
            bw_r = (global_kr_max - global_kr_min)

            payload = SICDImagePayload(
                complex_image=combined_img,
                tx_pol=tx_pol,
                rcv_pol=rcv_pol,
                uIAX=cphd_meta.uIAX,
                uIAY=cphd_meta.uIAY,
                iarp_ecf=cphd_meta.iarp_ecf,
                line_spacing=du,
                sample_spacing=dr,
                first_line=u_min,
                first_sample=r_min,
                center_freq=(cphd_meta.global_fx_min + cphd_meta.global_fx_max) / 2.0,
                bandwidth_u=bw_u,
                bandwidth_r=bw_r,
            )

            with Timer("Write Combined SICD"):
                saved_path = self.writer.write_sicd(out_path, payload, cphd_meta)
                output_files.append(saved_path)

        return output_files
