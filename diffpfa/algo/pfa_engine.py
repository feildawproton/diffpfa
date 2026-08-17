from scipy.fft import next_fast_len
from dataclasses import dataclass
import os
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from diffpfa.algo.channel.nufft_channel import process_channel_nufft
from diffpfa.algo.channel.cztnufft_channel import process_channel_czt_nufft
from diffpfa.algo.channel.geometry_channel import compute_kspace
from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.io.base import (
    BaseCPHDReader,
    BaseSICDWriter,
    CPHDChannelData,
    CPHDMetadata,
    SICDImagePayload,
)

# does nothing?!
def align_and_combine_channels(
    channel_images: List[torch.Tensor],
    align_phase: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Aligns relative phases between multiple sub-channel images of the same polarization pair
    and coherently combines them.

    Args:
        channel_images: List of 2D complex image tensors [I_0, I_1, ..., I_{C-1}].
        align_phase: If True, estimates phase offset relative to reference channel I_0 and aligns.

    Returns:
        combined_image: 2D complex image tensor of the combined response.
        aligned_images: List of aligned individual channel image tensors.
    """
    if len(channel_images) == 0:
        raise ValueError("channel_images list cannot be empty.")

    if len(channel_images) == 1:
        return channel_images[0], channel_images

    ref_img = channel_images[0]
    aligned_images = [ref_img]

    for c in range(1, len(channel_images)):
        curr_img = channel_images[c]

        # Note: Data-driven cross-correlation phase alignment is mathematically invalid for orthogonal 
        # frequency sub-bands. Phase alignment is now performed analytically using PVP RcvTime metadata 
        # in the PFA engine before gridding.
        curr_aligned = curr_img

        aligned_images.append(curr_aligned)

    # Coherent summation across channels without large stack allocations
    combined_image = aligned_images[0].clone()
    for i in range(1, len(aligned_images)):
        combined_image.add_(aligned_images[i])
        
    return combined_image, aligned_images

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
    czt_batch_size: int = 512
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

    def run(self) -> List[str]:
        """Runs the complete PFA processing pipeline."""
        cphd_meta = self.reader.get_metadata()
        channel_names = self.reader.get_channel_names()

        # Group channels by polarization pair: (tx_pol, rcv_pol) -> List[CPHDChannelData]
        pol_groups: Dict[Tuple[str, str], List[CPHDChannelData]] = {}
             
        reader_cls = self.reader.__class__
        file_path = self.reader.file_path
        
        def _load_ch(name):
            # Instantiate a new reader per-worker to ensure file operations are thread/process safe
            with reader_cls(file_path) as local_reader:
                return local_reader.read_channel(name)
            
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
                                
                # Analytical Phase Alignment (Center Frequency & Pulse Delay)
                if "RcvTime" in ch_data.pvp and "RcvTime" in first_ch.pvp:
                    tau = ch_data.pvp["RcvTime"] - first_ch.pvp["RcvTime"]
                    fc_global = (cphd_meta.global_fx_min + cphd_meta.global_fx_max) / 2.0
                    fc_channel = ch_data.fxc
                    tau_tensor = torch.as_tensor(tau, dtype=torch.float64, device=ch_data.signal.device)
                    phase_corr = -2.0 * torch.pi * (fc_global - fc_channel) * tau_tensor
                    corr_term = torch.exp(1j * phase_corr).unsqueeze(1)
                    ch_data.signal = ch_data.signal * corr_term.to(ch_data.signal.dtype)
                
                if mode == "nufft":
                    #img = self.process_channel_nufft(ch_data, cphd_meta, u_min, u_max, r_min, r_max, N_u, N_r, global_k_ctr_u, global_k_ctr_r)
                    img = process_channel_nufft(ch_data = ch_data,
                                                cphd_meta = cphd_meta,
                                                u_min = u_min,
                                                u_max = u_max,
                                                r_min = r_min,
                                                r_max = r_max,
                                                N_u = N_u,
                                                N_r = N_r,
                                                nufft_batch_size_pts = self.config.nufft_batch_size_pts,
                                                num_subpatches = self.config.num_subpatches,
                                                global_k_ctr_u = global_k_ctr_u,
                                                global_k_ctr_r = global_k_ctr_r,
                                                image_plane = self.config.image_plane,
                                                device = self.device) 
                elif mode == "cztnufft":
                    #img = self.process_channel_czt_nufft(ch_data, cphd_meta, u_min, u_max, r_min, r_max, N_u, N_r, global_k_ctr_u, global_k_ctr_r, return_kspace=combine_in_kspace)
                    img = process_channel_czt_nufft(ch_data = ch_data,
                                                    cphd_meta = cphd_meta,
                                                    u_min = u_min,
                                                    u_max = u_max,
                                                    r_min = r_min,
                                                    r_max = r_max,
                                                    N_u = N_u,
                                                    N_r = N_r,
                                                    czt_batch_size = self.config.czt_batch_size,
                                                    num_subpatches = self.config.num_subpatches,
                                                    global_k_ctr_u = global_k_ctr_u,
                                                    global_k_ctr_r = global_k_ctr_r,    
                                                    image_plane = self.config.image_plane,
                                                    device = self.device) 
  
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                channel_images.append(img)
 
                # Export debug channel image if requested
                if self.config.debug_save_channels:
                    save_img = img

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
                        complex_image=save_img,
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

            saved_path = self.writer.write_sicd(out_path, payload, cphd_meta)
            output_files.append(saved_path)

        return output_files
