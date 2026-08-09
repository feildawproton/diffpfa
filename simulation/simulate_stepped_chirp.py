import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffpfa.algo.pfa_engine import PFAEngine, PFAConfig
from diffpfa.io.base import BaseCPHDReader, CPHDMetadata, CPHDChannelData, BaseSICDWriter, SICDImagePayload

SPEED_OF_LIGHT = 299792458.0

class MockSteppedChirpCPHDReader(BaseCPHDReader):
    def __init__(self):
        super().__init__("mock_stepped.cphd")
        
        # 64 pulse pairs (128 total pulses, but 64 per subband)
        self.N_pulse_pairs = 64
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
            {"id": "Ch1_VV_Low", "tx": "V", "rcv": "V", "pos": self.pos_1, "fxc": self.fxc_1, "f_start": self.f_start_1},
            {"id": "Ch2_VH_Low", "tx": "V", "rcv": "H", "pos": self.pos_1, "fxc": self.fxc_1, "f_start": self.f_start_1},
            {"id": "Ch3_VV_High", "tx": "V", "rcv": "V", "pos": self.pos_2, "fxc": self.fxc_2, "f_start": self.f_start_2},
            {"id": "Ch4_VH_High", "tx": "V", "rcv": "H", "pos": self.pos_2, "fxc": self.fxc_2, "f_start": self.f_start_2},
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
            sample_spacing=0.1
        )
        
    def get_channel_names(self):
        return [ch["id"] for ch in self.channels]
        
    def read_channel(self, channel_name: str) -> CPHDChannelData:
        ch_info = next(ch for ch in self.channels if ch["id"] == channel_name)
        pos = ch_info["pos"]
        
        dist = np.linalg.norm(pos, axis=1)
        freqs = ch_info["f_start"] + np.arange(self.N_samples) * self.f_step
        
        # Target at SRP has 0 phase after motion compensation.
        # We inject a synthetic phase shift to test the channel alignment logic.
        phase = np.zeros((self.N_pulse_pairs, self.N_samples))
        
        if "High" in ch_info["id"]:
            phase += np.pi / 3.0  # Synthetic constant phase error on high band
            
        if ch_info["rcv"] == "H":
            phase += np.pi / 4.0  # Synthetic polarization phase offset
            
        signal = torch.tensor(np.exp(1j * phase), dtype=torch.complex64)
        
        pvp = {
            "TxPos": pos,
            "RcvPos": pos,
            "SRPPos": np.tile(self.srp_ecf, (self.N_pulse_pairs, 1)),
            "SC0": np.full(self.N_pulse_pairs, ch_info["f_start"]),
            "SCSS": np.full(self.N_pulse_pairs, self.f_step)
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
    def __init__(self):
        from diffpfa.io import SICDWriter
        self.real_writer = SICDWriter(backend="sarpy")
        self.payloads = []
        
    def write_sicd(self, output_path: str, payload: SICDImagePayload, cphd_meta: CPHDMetadata) -> str:
        self.payloads.append(payload)
        print(f"  [TrackingWriter] Captured and saving output for {payload.tx_pol}/{payload.rcv_pol} to {output_path}")
        return self.real_writer.write_sicd(output_path, payload, cphd_meta)
        
    def finalize(self): pass
    def close(self): pass


def run_simulation():
    reader = MockSteppedChirpCPHDReader()
    writer = TrackingSICDWriter()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PFAConfig(
        mode="czt",
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
    
    img_combined = torch.abs(vv_combined.complex_image).cpu().numpy()
    img_single = torch.abs(vv_ch1.complex_image).cpu().numpy()
    
    center_u = img_combined.shape[0] // 2
    slice_combined = img_combined[center_u, :]
    slice_single = img_single[center_u, :]
    
    slice_combined /= slice_combined.max()
    slice_single /= slice_single.max()
    
    # Zoom in around the peak
    center_r_s = np.argmax(slice_single)
    center_r_c = np.argmax(slice_combined)
    window = 10
    
    zoom_single = slice_single[center_r_s - window : center_r_s + window + 1]
    zoom_combined = slice_combined[center_r_c - window : center_r_c + window + 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    ax1.plot(range(-window, window + 1), zoom_single, color='blue', marker='o')
    ax1.set_title('Single Channel (250 MHz) - Zoomed')
    ax1.set_xlabel('Relative Range Bin')
    ax1.set_ylabel('Normalized Magnitude')
    ax1.grid(True)
    
    ax2.plot(range(-window, window + 1), zoom_combined, color='red', marker='o')
    ax2.set_title('Combined (500 MHz) - Zoomed')
    ax2.set_xlabel('Relative Range Bin')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation/ipr_comparison_zoomed.png')
    
    # Calculate energy concentration (fraction of energy in center 3 pixels vs total)
    energy_s = np.sum(zoom_single**2)
    energy_c = np.sum(zoom_combined**2)
    
    core_s = np.sum(zoom_single[window-1:window+2]**2) / energy_s
    core_c = np.sum(zoom_combined[window-1:window+2]**2) / energy_c
    
    print(f"\\nQuantitative Comparison:")
    print(f"Energy concentration in core 3 pixels - Single: {core_s:.2%}")
    print(f"Energy concentration in core 3 pixels - Combined: {core_c:.2%}")
    
    # Interpolated FWHM calculation
    def get_fwhm(slice_arr):
        above_half = np.where(slice_arr > 0.5)[0]
        if len(above_half) < 2: return 1.0
        # Simple linear interpolation for the edges
        left = above_half[0]
        right = above_half[-1]
        
        # interpolate left edge
        if left > 0:
            y1, y2 = slice_arr[left-1], slice_arr[left]
            frac_l = (0.5 - y1) / (y2 - y1)
            true_left = left - 1 + frac_l
        else:
            true_left = left
            
        # interpolate right edge
        if right < len(slice_arr) - 1:
            y1, y2 = slice_arr[right], slice_arr[right+1]
            frac_r = (0.5 - y1) / (y2 - y1)
            true_right = right + frac_r
        else:
            true_right = right
            
        return true_right - true_left

    fwhm_single = get_fwhm(slice_single)
    fwhm_combined = get_fwhm(slice_combined)
    
    print(f"\\nFWHM (pixels) - Single Channel: {fwhm_single}")
    print(f"FWHM (pixels) - Combined: {fwhm_combined}")
    
    if fwhm_combined < fwhm_single:
        print("SUCCESS: Combined resolution is sharper than single channel!")
    else:
        print("WARNING: Combined resolution did not improve.")

if __name__ == '__main__':
    run_simulation()
