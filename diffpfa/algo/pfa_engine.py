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
    SPEED_OF_LIGHT
)
from diffpfa.algo.nufft_torch import nufft_1d_type1_torch, nufft_2d_type1_torch, nufft_grid_1d
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
    mode: str = "hybrid"  # "hybrid", "czt", "nufft"
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
        
        # JIT compile hot paths for massive GPU acceleration if requested
        if self.config.enable_compile:
            global czt_1d_torch, nufft_1d_type1_torch, nufft_2d_type1_torch
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
            czt_1d_torch = torch.compile(czt_1d_torch)
            nufft_1d_type1_torch = torch.compile(nufft_1d_type1_torch)
            nufft_2d_type1_torch = torch.compile(nufft_2d_type1_torch)

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
        else:
            # Match native bandwidth pixel spacing
            N_u = int(np.round(L_u / native_du))
            N_r = int(np.round(L_r / native_dr))

        N_u = max(N_u, 16)
        N_r = max(N_r, 16)

        return u_min, u_max, r_min, r_max, N_u, N_r


    def process_channel_czt(
        self,
        ch_data: CPHDChannelData,
        cphd_meta: CPHDMetadata,
        u_min: float,
        u_max: float,
        r_min: float,
        r_max: float,
        N_u: int,
        N_r: int
    ) -> torch.Tensor:
        """Processes a single CPHD channel using 1D PyTorch CZT separable resampling."""
        signal = ch_data.signal.to(self.device)
        pvp = ch_data.pvp
        N_pulses, N_samples = signal.shape

        # Step 0: RVP Deskew (if deramped data)
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

        num_patches = max(1, self.config.num_subpatches)
        subpatch_size_u = int(np.ceil(N_u / num_patches))
        subpatch_size_r = int(np.ceil(N_r / num_patches))

        image_2d = torch.zeros((N_u, N_r), dtype=signal.dtype, device=self.device)
        
        P_vecs_orig = compute_look_vectors(pvp, device=self.device)
        F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=self.device)
        
        if self.config.image_plane == "Slant":
            # Extract ReferenceGeometry via the original Sarpy object
            # Note: We re-open or access the underlying raw metadata from the reader if needed.
            # But wait, our cphd_meta is just a dataclass. Let's get the original metadata from self.reader!
            raw_meta = self.reader.reader.cphd_meta
            rg = raw_meta.ReferenceGeometry
            srp = rg.SRP.get_array()
            arp = rg.Monostatic.ARPPos.get_array()
            arp_v = rg.Monostatic.ARPVel.get_array()
            
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

        u_edges = np.linspace(u_min, u_max, N_u + 1)
        r_edges = np.linspace(r_min, r_max, N_r + 1)

        for i_u in range(0, N_u, subpatch_size_u):
            for i_r in range(0, N_r, subpatch_size_r):
                end_u = min(i_u + subpatch_size_u, N_u)
                end_r = min(i_r + subpatch_size_r, N_r)
                
                p_u_min, p_u_max = u_edges[i_u], u_edges[end_u]
                p_r_min, p_r_max = r_edges[i_r], r_edges[end_r]
                
                u_c = (p_u_min + p_u_max) / 2.0
                r_c = (p_r_min + p_r_max) / 2.0
                
                # SCP Shift
                P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
                R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
                R_patch = torch.linalg.norm(P_patch, dim=-1)
                dR = R_patch - R_orig
                
                phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
                corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
                sig_patch = signal * corr_term.to(signal.dtype)
                
                cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
                F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
                Ku = F_cpm * cos_theta.unsqueeze(1)
                Kr = F_cpm * sin_theta.unsqueeze(1)
                
                local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
                local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
                
                p_N_u, p_N_r = end_u - i_u, end_r - i_r
                
                # Shared Grid Definitions
                oversample = 1.5
                from scipy.fft import next_fast_len
                M_u = next_fast_len(int(math.ceil(p_N_u * oversample)))
                M_r = next_fast_len(int(math.ceil(p_N_r * oversample)))
                L_u = local_u_max - local_u_min
                L_r = local_r_max - local_r_min
                dK_u = p_N_u / (M_u * max(L_u, 1e-12))
                dK_r = p_N_r / (M_r * max(L_r, 1e-12))
                k_ctr_u = (Ku.min() + Ku.max()).item() / 2.0
                k_ctr_r = (Kr.min() + Kr.max()).item() / 2.0
                
                # 1. Range CZT Resampling (K-space -> Cartesian K_r)
                k_out_start_r = k_ctr_r - (M_r / 2.0) * dK_r
                k_out_step_r = dK_r
                
                czt_batch_size = self.config.czt_batch_size
                range_resampled = torch.zeros((N_pulses, M_r), dtype=sig_patch.dtype, device=self.device)
                
                for b in range(0, N_pulses, czt_batch_size):
                    sig_b = sig_patch[b:b + czt_batch_size]
                    kr_b = Kr[b:b + czt_batch_size]
                    k_start = kr_b[:, 0].unsqueeze(1)
                    k_step = ((kr_b[:, -1] - kr_b[:, 0]) / max(N_samples - 1, 1)).unsqueeze(1)
                    
                    batch_out = czt_resample_kspace_1d(
                        sig_b, 
                        k_start=k_start, 
                        k_step=k_step,
                        M_out=M_r,
                        k_out_start=k_out_start_r,
                        k_out_step=k_out_step_r,
                        L_r=L_r,
                        oversample=oversample
                    )
                    range_resampled[b:b+czt_batch_size, :] = batch_out
                
                del sig_patch
                del Kr
                torch.cuda.empty_cache()
                
                # 2. Cross-Range CZT Resampling (Cartesian K_r -> Cartesian K_u)
                Ku_center_bin = Ku[:, N_samples // 2]
                del Ku
                torch.cuda.empty_cache()
                
                k_out_start_u = k_ctr_u - (M_u / 2.0) * dK_u
                k_out_step_u = dK_u
                
                ku_start = Ku_center_bin[0].unsqueeze(0)
                ku_step = ((Ku_center_bin[-1] - Ku_center_bin[0]) / max(N_pulses - 1, 1)).unsqueeze(0)
                
                grid_2d = torch.zeros((M_u, M_r), dtype=range_resampled.dtype, device=self.device)
                
                for b in range(0, M_r, czt_batch_size):
                    range_batch = range_resampled[:, b:b + czt_batch_size]
                    
                    # Transpose to (batch, N_pulses) so CZT operates on the N_pulses dimension
                    range_batch_t = range_batch.T
                    
                    batch_out_t = czt_resample_kspace_1d(
                        range_batch_t, 
                        k_start=ku_start, 
                        k_step=ku_step,
                        M_out=M_u,
                        k_out_start=k_out_start_u,
                        k_out_step=k_out_step_u,
                        L_r=L_u,
                        oversample=oversample
                    )
                    grid_2d[:, b:b+range_batch.shape[1]] = batch_out_t.T
                
                del range_resampled
                torch.cuda.empty_cache()
                
                # 3. 2D IFFT (No deconvolution needed since CZT is exact interpolation)
                grid_shifted = torch.fft.ifftshift(grid_2d)
                del grid_2d
                img_oversampled = torch.fft.ifft2(grid_shifted)
                del grid_shifted
                img_oversampled.mul_(M_u * M_r)
                img_shifted = torch.fft.fftshift(img_oversampled)
                del img_oversampled
                
                start_u = (M_u - p_N_u) // 2
                start_r = (M_r - p_N_r) // 2
                
                patch_img = img_shifted[start_u : start_u + p_N_u, start_r : start_r + p_N_r]
                image_2d[i_u:end_u, i_r:end_r] = patch_img

        return image_2d

    def process_channel_hybrid(
        self,
        ch_data: CPHDChannelData,
        cphd_meta: CPHDMetadata,
        u_min: float,
        u_max: float,
        r_min: float,
        r_max: float,
        N_u: int,
        N_r: int
    ) -> torch.Tensor:
        """Processes a single CPHD channel using a Hybrid approach: 1D CZT in Range, 1D NUFFT in Cross-Range."""
        signal = ch_data.signal.to(self.device)
        pvp = ch_data.pvp
        N_pulses, N_samples = signal.shape

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

        num_patches = max(1, self.config.num_subpatches)
        subpatch_size_u = int(np.ceil(N_u / num_patches))
        subpatch_size_r = int(np.ceil(N_r / num_patches))

        image_2d = torch.zeros((N_u, N_r), dtype=signal.dtype, device=self.device)
        
        P_vecs_orig = compute_look_vectors(pvp, device=self.device)
        F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=self.device)
        
        if self.config.image_plane == "Slant":
            raw_meta = self.reader.reader.cphd_meta
            rg = raw_meta.ReferenceGeometry
            srp = rg.SRP.get_array()
            arp = rg.Monostatic.ARPPos.get_array()
            arp_v = rg.Monostatic.ARPVel.get_array()
            
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

        u_edges = np.linspace(u_min, u_max, N_u + 1)
        r_edges = np.linspace(r_min, r_max, N_r + 1)

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
                dR = R_patch - R_orig
                
                phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
                corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
                sig_patch = signal * corr_term.to(signal.dtype)
                
                cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
                F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
                Ku = F_cpm * cos_theta.unsqueeze(1)
                Kr = F_cpm * sin_theta.unsqueeze(1)
                
                local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
                local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
                
                p_N_u, p_N_r = end_u - i_u, end_r - i_r
                
                # Shared Grid Definitions
                oversample = 1.5
                from scipy.fft import next_fast_len
                M_u = next_fast_len(int(math.ceil(p_N_u * oversample)))
                M_r = next_fast_len(int(math.ceil(p_N_r * oversample)))
                L_u = local_u_max - local_u_min
                L_r = local_r_max - local_r_min
                dK_u = p_N_u / (M_u * max(L_u, 1e-12))
                dK_r = p_N_r / (M_r * max(L_r, 1e-12))
                k_ctr_u = (Ku.min() + Ku.max()).item() / 2.0
                k_ctr_r = (Kr.min() + Kr.max()).item() / 2.0
                
                # 1. Range CZT Resampling (K-space -> Cartesian K_r)
                k_out_start_r = k_ctr_r - (M_r / 2.0) * dK_r
                k_out_step_r = dK_r
                
                czt_batch_size = self.config.czt_batch_size
                range_resampled = torch.zeros((N_pulses, M_r), dtype=sig_patch.dtype, device=self.device)
                
                for b in range(0, N_pulses, czt_batch_size):
                    sig_b = sig_patch[b:b + czt_batch_size]
                    kr_b = Kr[b:b + czt_batch_size]
                    k_start = kr_b[:, 0].unsqueeze(1)
                    k_step = ((kr_b[:, -1] - kr_b[:, 0]) / max(N_samples - 1, 1)).unsqueeze(1)
                    
                    batch_out = czt_resample_kspace_1d(
                        sig_b, 
                        k_start=k_start, 
                        k_step=k_step,
                        M_out=M_r,
                        k_out_start=k_out_start_r,
                        k_out_step=k_out_step_r,
                        L_r=L_r,
                        oversample=oversample
                    )
                    range_resampled[b:b+czt_batch_size, :] = batch_out
                
                del sig_patch
                torch.cuda.empty_cache()
                
                # 2. Cross-Range NUFFT Gridding (Cartesian K_r -> Cartesian K_u)
                cot_theta = Ku[:, N_samples//2] / Kr[:, N_samples//2]
                m_idx = torch.arange(M_r, device=self.device, dtype=torch.float64)
                Kr_cart = k_out_start_r + m_idx * dK_r
                
                del Ku
                del Kr
                torch.cuda.empty_cache()
                
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
                torch.cuda.empty_cache()
                
                # 3. 2D IFFT and Deconvolution
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



    def process_channel_nufft(
        self,
        ch_data: CPHDChannelData,
        cphd_meta: CPHDMetadata,
        u_min: float,
        u_max: float,
        r_min: float,
        r_max: float,
        N_u: int,
        N_r: int
    ) -> torch.Tensor:
        """Processes a single CPHD channel using 2D PyTorch NUFFT gridding."""
        signal = ch_data.signal.to(self.device)
        pvp = ch_data.pvp
        N_pulses, N_samples = signal.shape

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

        num_patches = max(1, self.config.num_subpatches)
        subpatch_size_u = int(np.ceil(N_u / num_patches))
        subpatch_size_r = int(np.ceil(N_r / num_patches))

        image_2d = torch.zeros((N_u, N_r), dtype=signal.dtype, device=self.device)
        
        P_vecs_orig = compute_look_vectors(pvp, device=self.device)
        F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device=self.device)
        
        if self.config.image_plane == "Slant":
            raw_meta = self.reader.reader.cphd_meta
            rg = raw_meta.ReferenceGeometry
            srp = rg.SRP.get_array()
            arp = rg.Monostatic.ARPPos.get_array()
            arp_v = rg.Monostatic.ARPVel.get_array()
            
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

        u_edges = np.linspace(u_min, u_max, N_u + 1)
        r_edges = np.linspace(r_min, r_max, N_r + 1)

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
                dR = R_patch - R_orig
                
                phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
                corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
                sig_patch = signal * corr_term.to(signal.dtype)
                
                cos_theta, sin_theta = compute_look_components(P_patch, uIAX, uIAY)
                F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
                Ku = F_cpm * cos_theta.unsqueeze(1)
                Kr = F_cpm * sin_theta.unsqueeze(1)
                
                local_u_min, local_u_max = p_u_min - u_c, p_u_max - u_c
                local_r_min, local_r_max = p_r_min - r_c, p_r_max - r_c
                p_N_u, p_N_r = end_u - i_u, end_r - i_r
                
                patch_img = nufft_2d_type1_torch(
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
                image_2d[i_u:end_u, i_r:end_r] = patch_img

        return image_2d


    def run(self) -> List[str]:
        """Runs the complete PFA processing pipeline."""
        cphd_meta = self.reader.get_metadata()
        channel_names = self.reader.get_channel_names()

        # Group channels by polarization pair: (tx_pol, rcv_pol) -> List[CPHDChannelData]
        pol_groups: Dict[Tuple[str, str], List[CPHDChannelData]] = {}

        for ch_name in channel_names:
            ch_data = self.reader.read_channel(ch_name)
            pol_key = (ch_data.tx_pol, ch_data.rcv_pol)
            if pol_key not in pol_groups:
                pol_groups[pol_key] = []
            pol_groups[pol_key].append(ch_data)

        output_files = []
        os.makedirs(self.config.output_dir, exist_ok=True)

        for pol_key, ch_list in pol_groups.items():
            tx_pol, rcv_pol = pol_key

            # Determine spatial bounds from first channel
            first_ch = ch_list[0]
            Ku_sample, Kr_sample = compute_kspace(
                first_ch.pvp,
                cphd_meta.uIAX,
                cphd_meta.uIAY,
                first_ch.signal.shape[1],
                first_ch.domain_type,
                device=self.device
            )

            u_min, u_max, r_min, r_max, N_u, N_r = self._determine_spatial_bounds(
                cphd_meta, Ku_sample, Kr_sample
            )

            # Process each channel in group
            channel_images = []

            for ch_data in ch_list:
                # Strict mode selection
                mode = self.config.mode
                if mode == "czt":
                    img = self.process_channel_czt(ch_data, cphd_meta, u_min, u_max, r_min, r_max, N_u, N_r)
                elif mode == "nufft":
                    img = self.process_channel_nufft(ch_data, cphd_meta, u_min, u_max, r_min, r_max, N_u, N_r)
                elif mode == "hybrid":
                    img = self.process_channel_hybrid(ch_data, cphd_meta, u_min, u_max, r_min, r_max, N_u, N_r)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                channel_images.append(img)

                # Export debug channel image if requested
                if self.config.debug_save_channels:
                    dbg_name = f"SICD_U_{tx_pol}_{rcv_pol}_ch_{ch_data.identifier}.nitf"
                    dbg_path = os.path.join(self.config.output_dir, dbg_name)

                    du = (u_max - u_min) / max(N_u, 1)
                    dr = (r_max - r_min) / max(N_r, 1)
                    bw_u = (Ku_sample.max() - Ku_sample.min()).item()
                    bw_r = (Kr_sample.max() - Kr_sample.min()).item()

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
            combined_img, _ = align_and_combine_channels(
                channel_images,
                align_phase=self.config.align_subchannels
            )

            # Write primary combined SICD-U file
            out_name = f"SICD_U_{tx_pol}_{rcv_pol}.nitf"
            out_path = os.path.join(self.config.output_dir, out_name)

            du = (u_max - u_min) / max(N_u, 1)
            dr = (r_max - r_min) / max(N_r, 1)
            bw_u = (Ku_sample.max() - Ku_sample.min()).item()
            bw_r = (Kr_sample.max() - Kr_sample.min()).item()

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

            saved_path = self.writer.write_sicd(out_path, payload, cphd_meta)
            output_files.append(saved_path)

        return output_files
