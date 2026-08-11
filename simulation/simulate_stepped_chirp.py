import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffpfa.algo.pfa_engine import PFAEngine, PFAConfig
from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.io.base import BaseCPHDReader, CPHDMetadata, CPHDChannelData, BaseSICDWriter, SICDImagePayload

class MockSteppedChirpCPHDReader(BaseCPHDReader):
    def __init__(self, file_path="mock_stepped.cphd"):
        super().__init__(file_path)
        
        # 256 pulse pairs (512 total pulses, but 256 per subband) for a square output
        self.N_pulse_pairs = 256
        self.N_samples = 256
        
        self.srp_ecf = np.array([0.0, 0.0, 0.0])
        vel = 100.0  # m/s
        PRF = 1000.0 # Hz 
        PRI = 1.0 / PRF
        
        # Subband 1 timing
        t_1 = np.arange(self.N_pulse_pairs) * (2 * PRI)
        # Subband 2 timing (offset by 1 PRI)
        t_2 = t_1 + PRI
        
        # Subband 1 Platform Positions
        self.pos_1 = np.zeros((self.N_pulse_pairs, 3))
        self.pos_1[:, 0] = 10000.0
        self.pos_1[:, 1] = vel * (t_1 - np.mean(t_1))
        self.pos_1[:, 2] = 1000.0
        
        # Subband 2 Platform Positions
        self.pos_2 = np.zeros((self.N_pulse_pairs, 3))
        self.pos_2[:, 0] = 10000.0
        self.pos_2[:, 1] = vel * (t_2 - np.mean(t_1))
        self.pos_2[:, 2] = 1000.0
        
        self.arp_pos_coa = self.pos_1[self.N_pulse_pairs // 2]
        self.arp_vel_coa = np.array([0.0, vel, 0.0])
        
        self.fxbw = 250e6
        self.fxc_1 = 10.0e9
        self.f_start_1 = self.fxc_1 - self.fxbw / 2
        
        self.fxc_2 = self.fxc_1 + self.fxbw
        self.f_start_2 = self.fxc_2 - self.fxbw / 2
        
        self.f_step = self.fxbw / self.N_samples
        
        self.channels = [
            {"id": "Ch1_VV_Low", "tx": "V", "rcv": "V", "pos": self.pos_1, "fxc": self.fxc_1, "f_start": self.f_start_1, "t": t_1},
            {"id": "Ch2_VH_Low", "tx": "V", "rcv": "H", "pos": self.pos_1, "fxc": self.fxc_1, "f_start": self.f_start_1, "t": t_1},
            {"id": "Ch3_VV_High", "tx": "V", "rcv": "V", "pos": self.pos_2, "fxc": self.fxc_2, "f_start": self.f_start_2, "t": t_2},
            {"id": "Ch4_VH_High", "tx": "V", "rcv": "H", "pos": self.pos_2, "fxc": self.fxc_2, "f_start": self.f_start_2, "t": t_2},
        ]
        
    def get_metadata(self) -> CPHDMetadata:
        return CPHDMetadata(
            domain_type="FX",
            sgn=-1,
            global_fx_min=self.f_start_1,
            global_fx_max=self.f_start_2 + self.fxbw,
            iarp_ecf=np.array([0.0, 0.0, 0.0]),
            uIAX=np.array([0.0, 1.0, 0.0]), 
            uIAY=np.array([1.0, 0.0, 0.0]), 
            srp_ecf=self.srp_ecf,
            arp_pos_coa=self.arp_pos_coa,
            arp_vel_coa=self.arp_vel_coa,
            line_spacing=0.1,
            sample_spacing=0.1,
            classification="UNCLASSIFIED"
        )
        
    def get_channel_names(self):
        return [ch["id"] for ch in self.channels]
        
    def read_channel(self, channel_name: str) -> CPHDChannelData:
        ch_info = next(ch for ch in self.channels if ch["id"] == channel_name)
        pos = ch_info["pos"]
        
        dist = np.linalg.norm(pos, axis=1)
        freqs = ch_info["f_start"] + np.arange(self.N_samples) * self.f_step
        
        # Target at SRP has 0 phase after motion compensation.
        # We model the actual physics: the deterministic phase drift between sub-bands
        # over the difference in receive times needs to be injected so the engine can correct it.
        phase = np.zeros((self.N_pulse_pairs, self.N_samples))
        
        # Inject the deterministic offset that the engine's new analytical correction expects to fix!
        # The engine will subtract this. We ADD it so that when the engine subtracts it, phase becomes 0.
        t_arr = ch_info["t"]
        t_ref = self.channels[0]["t"]
        tau = t_arr - t_ref
        fc_global = (self.f_start_1 + (self.f_start_2 + self.fxbw)) / 2.0
        phase += 2.0 * np.pi * (fc_global - ch_info["fxc"]) * tau[:, None]
        
        if ch_info["rcv"] == "H":
            phase += np.pi / 4.0  # Synthetic polarization phase offset (should remain in VH)
            
        signal = torch.tensor(np.exp(1j * phase), dtype=torch.complex64)
        
        tau_roundtrip = 2.0 * dist / SPEED_OF_LIGHT
        
        pvp = {
            "TxPos": pos,
            "RcvPos": pos,
            "SRPPos": np.tile(self.srp_ecf, (self.N_pulse_pairs, 1)),
            "SC0": np.full(self.N_pulse_pairs, ch_info["f_start"]),
            "SCSS": np.full(self.N_pulse_pairs, self.f_step),
            "TxTime": t_arr,
            "RcvTime": t_arr + tau_roundtrip
        }
        
        return CPHDChannelData(
            identifier=channel_name,
            tx_pol=ch_info["tx"],
            rcv_pol=ch_info["rcv"],
            fxc=ch_info["fxc"],
            fxbw=self.fxbw,
            signal=signal,
            pvp=pvp
        )
        
    def close(self): pass


class TrackingSICDWriter(BaseSICDWriter):
    def __init__(self, mode):
        from diffpfa.io import SICDWriter
        self.real_writer = SICDWriter(backend="sarkit")
        self.payloads = []
        self.mode = mode
        
    def write_sicd(self, output_path: str, payload: SICDImagePayload, cphd_meta: CPHDMetadata) -> str:
        self.payloads.append(payload)
        out_path = output_path.replace('.nitf', f'_{self.mode}.nitf')
        print(f"  [TrackingWriter] Captured and saving output for {payload.tx_pol}/{payload.rcv_pol} to {out_path}")
        return self.real_writer.write_sicd(out_path, payload, cphd_meta)
        
    def finalize(self): pass
    def close(self): pass


def run_simulation(mode="cztnufft"):
    reader = MockSteppedChirpCPHDReader()
    writer = TrackingSICDWriter(mode)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PFAConfig(
        mode=mode,
        device=device,
        num_subpatches=1,
        align_subchannels=True,
        debug_save_channels=True,
        output_dir="simulation/output"
    )
    
    engine = PFAEngine(reader, writer, config)
    engine.run()
    
    vv_combined = next(p for p in writer.payloads if p.tx_pol == "V" and p.rcv_pol == "V" and p.channel_id is None)
    vv_ch1 = next(p for p in writer.payloads if p.channel_id == "Ch1_VV_Low")
    vv_ch3 = next(p for p in writer.payloads if p.channel_id == "Ch3_VV_High")
    vh_combined = next(p for p in writer.payloads if p.tx_pol == "V" and p.rcv_pol == "H" and p.channel_id is None)
    vh_ch2 = next(p for p in writer.payloads if p.channel_id == "Ch2_VH_Low")
    
    img_combined_vv = torch.abs(vv_combined.complex_image).cpu().numpy()
    img_single1_vv = torch.abs(vv_ch1.complex_image).cpu().numpy()
    img_single2_vv = torch.abs(vv_ch3.complex_image).cpu().numpy()
    img_combined_vh = torch.abs(vh_combined.complex_image).cpu().numpy()
    img_single_vh = torch.abs(vh_ch2.complex_image).cpu().numpy()
    
    center_u = img_combined_vv.shape[0] // 2
    center_r = img_combined_vv.shape[1] // 2
    
    # Range slices (fix cross-range u)
    slice_r_combined_vv = img_combined_vv[center_u, :]
    slice_r_single1_vv = img_single1_vv[center_u, :]
    slice_r_single2_vv = img_single2_vv[center_u, :]
    slice_r_combined_vh = img_combined_vh[center_u, :]
    slice_r_single_vh = img_single_vh[center_u, :]
    
    # Cross-Range slices (fix range r)
    slice_u_combined_vv = img_combined_vv[:, center_r]
    slice_u_single1_vv = img_single1_vv[:, center_r]
    slice_u_single2_vv = img_single2_vv[:, center_r]
    
    slice_r_combined_vv /= slice_r_combined_vv.max()
    slice_r_single1_vv /= slice_r_single1_vv.max()
    slice_r_single2_vv /= slice_r_single2_vv.max()
    slice_r_combined_vh /= slice_r_combined_vh.max()
    slice_r_single_vh /= slice_r_single_vh.max()
    
    slice_u_combined_vv /= slice_u_combined_vv.max()
    slice_u_single1_vv /= slice_u_single1_vv.max()
    slice_u_single2_vv /= slice_u_single2_vv.max()
    
    # Zoom in around the peak for metrics
    center_r_s_vv = np.argmax(slice_r_single1_vv)
    center_r_c_vv = np.argmax(slice_r_combined_vv)
    center_r_s_vh = np.argmax(slice_r_single_vh)
    center_r_c_vh = np.argmax(slice_r_combined_vh)
    window = 10
    
    zoom_single_vv = slice_r_single1_vv[center_r_s_vv - window : center_r_s_vv + window + 1]
    zoom_combined_vv = slice_r_combined_vv[center_r_c_vv - window : center_r_c_vv + window + 1]
    zoom_single_vh = slice_r_single_vh[center_r_s_vh - window : center_r_s_vh + window + 1]
    zoom_combined_vh = slice_r_combined_vh[center_r_c_vh - window : center_r_c_vh + window + 1]
    
    energy_s_vv = np.sum(zoom_single_vv**2)
    energy_c_vv = np.sum(zoom_combined_vv**2)
    core_s_vv = np.sum(zoom_single_vv[window-1:window+2]**2) / energy_s_vv
    core_c_vv = np.sum(zoom_combined_vv[window-1:window+2]**2) / energy_c_vv

    energy_s_vh = np.sum(zoom_single_vh**2)
    energy_c_vh = np.sum(zoom_combined_vh**2)
    core_s_vh = np.sum(zoom_single_vh[window-1:window+2]**2) / energy_s_vh
    core_c_vh = np.sum(zoom_combined_vh[window-1:window+2]**2) / energy_c_vh

    print("\nQuantitative Comparison:")
    print(f"VV Energy concentration in core 3 pixels - Single: {core_s_vv:.2%}")
    print(f"VV Energy concentration in core 3 pixels - Combined: {core_c_vv:.2%}")
    print(f"VH Energy concentration in core 3 pixels - Single: {core_s_vh:.2%}")
    print(f"VH Energy concentration in core 3 pixels - Combined: {core_c_vh:.2%}")
    
    def get_fwhm(slice_arr):
        above_half = np.where(slice_arr > 0.5)[0]
        if len(above_half) < 2: return 1.0
        left = above_half[0]
        right = above_half[-1]
        
        if left > 0:
            y1, y2 = slice_arr[left-1], slice_arr[left]
            frac_l = (0.5 - y1) / (y2 - y1)
            true_left = left - 1 + frac_l
        else:
            true_left = left
            
        if right < len(slice_arr) - 1:
            y1, y2 = slice_arr[right], slice_arr[right+1]
            frac_r = (0.5 - y1) / (y2 - y1)
            true_right = right + frac_r
        else:
            true_right = right
            
        return true_right - true_left

    print("Max VV Single:", np.max(img_single1_vv))
    print("Max VV Combined:", np.max(img_combined_vv))

    fwhm_s_vv = get_fwhm(slice_r_single1_vv)
    fwhm_c_vv = get_fwhm(slice_r_combined_vv)
    fwhm_s_vh = get_fwhm(slice_r_single_vh)
    fwhm_c_vh = get_fwhm(slice_r_combined_vh)
    
    fwhm_u_s_vv = get_fwhm(slice_u_single1_vv)
    fwhm_u_c_vv = get_fwhm(slice_u_combined_vv)
    
    print(f"\nVV FWHM (pixels) - Single Channel [Range]: {fwhm_s_vv}")
    print(f"VV FWHM (pixels) - Combined [Range]: {fwhm_c_vv}")
    print(f"VH FWHM (pixels) - Single Channel [Range]: {fwhm_s_vh}")
    print(f"VH FWHM (pixels) - Combined [Range]: {fwhm_c_vh}")
    
    print(f"\nVV FWHM (pixels) - Single Channel [Cross-Range]: {fwhm_u_s_vv}")
    print(f"VV FWHM (pixels) - Combined [Cross-Range]: {fwhm_u_c_vv}")
    
    if fwhm_c_vv < fwhm_s_vv * 0.6 and fwhm_c_vh < fwhm_s_vh * 0.6:
        print("\nSUCCESS: Combined resolution is sharper than single channel for both polarizations!")
    else:
        print("\nWARNING: Coherent combination did not achieve expected resolution improvement.")

    # 2-Panel Plot: Range and Cross-Range Slices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Range slice
    peak_r = np.argmax(slice_r_combined_vv)
    ax1.plot(slice_r_single1_vv, label="Subband 1 (VV_Low)", linewidth=2, color='blue', alpha=0.7)
    ax1.plot(slice_r_single2_vv, label="Subband 2 (VV_High)", linewidth=2, color='green', alpha=0.7)
    ax1.plot(slice_r_combined_vv, label="Coherent Combined", linewidth=2, color='red', linestyle='--')
    ax1.set_title(f"VV IPR Range Slice - {mode}")
    ax1.set_xlabel("Pixels")
    ax1.set_ylabel("Normalized Magnitude")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(peak_r - 20, peak_r + 20)
    
    # Panel 2: Cross-Range slice
    peak_u = np.argmax(slice_u_combined_vv)
    ax2.plot(slice_u_single1_vv, label="Subband 1 (VV_Low)", linewidth=2, color='blue', alpha=0.7)
    ax2.plot(slice_u_single2_vv, label="Subband 2 (VV_High)", linewidth=2, color='green', alpha=0.7)
    ax2.plot(slice_u_combined_vv, label="Coherent Combined", linewidth=2, color='red', linestyle='--')
    ax2.set_title(f"VV IPR Cross-Range Slice - {mode}")
    ax2.set_xlabel("Pixels")
    ax2.set_ylabel("Normalized Magnitude")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(peak_u - 20, peak_u + 20)
    
    plt.tight_layout()
    plt.savefig(f"simulation/output/ipr_slice_{mode}.png")
    plt.close()

if __name__ == '__main__':
    for m in ["cztnufft", "nufft"]:
        print(f"\\n{'='*50}\\nRunning simulation for mode: {m}\\n{'='*50}")
        run_simulation(m)
