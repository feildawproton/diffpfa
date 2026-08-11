import sys
import os
import pytest
import numpy as np
import torch
from diffpfa.io.base import BaseCPHDReader, CPHDMetadata, CPHDChannelData

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

@pytest.fixture
def synthetic_cphd_reader():
    """Returns a MockCPHDReader that generates an ideal synthetic point target."""
    class MockCPHDReader(BaseCPHDReader):
        def __init__(self, *args, **kwargs):
            super().__init__("mock_file.cphd")
            self.N_pulses = 64
            self.N_samples = 128
            self.fxc = 10e9
            self.fxbw = 500e6
            
            self.srp_ecf = np.array([0.0, 0.0, 0.0])
            self.arp_pos = np.zeros((self.N_pulses, 3))
            self.arp_pos[:, 0] = 10000.0  # X position (range direction)
            self.arp_pos[:, 1] = np.linspace(-100.0, 100.0, self.N_pulses)  # Y position (cross-range)
            self.arp_pos[:, 2] = 1000.0   # Z position (altitude)
            
            self.arp_pos_coa = self.arp_pos[self.N_pulses // 2]
            self.arp_vel_coa = np.array([0.0, 100.0, 0.0])
            
        def get_metadata(self) -> CPHDMetadata:
            return CPHDMetadata(
                domain_type="FX",
                sgn=-1,
                global_fx_min=self.fxc - self.fxbw/2,
                global_fx_max=self.fxc + self.fxbw/2,
                iarp_ecf=np.array([0.0, 0.0, 0.0]),
                uIAX=np.array([0.0, 1.0, 0.0]),
                uIAY=np.array([1.0, 0.0, 0.0]),
                srp_ecf=self.srp_ecf,
                arp_pos_coa=self.arp_pos_coa,
                arp_vel_coa=self.arp_vel_coa,
                line_spacing=0.3,
                sample_spacing=0.3
            )
            
        def get_channel_names(self):
            return ["Primary"]
            
        def read_channel(self, channel_name: str) -> CPHDChannelData:
            dist = np.linalg.norm(self.arp_pos, axis=1) # (N_pulses,)
            f_start = self.fxc - self.fxbw / 2
            f_step = self.fxbw / self.N_samples
            freqs = f_start + np.arange(self.N_samples) * f_step # (N_samples,)
            
            phase = -4 * np.pi * np.outer(dist, freqs) / 299792458.0
            signal = torch.tensor(np.exp(1j * phase), dtype=torch.complex64)
            
            pvp = {
                "TxPos": self.arp_pos,
                "RcvPos": self.arp_pos,
                "SRPPos": np.tile(self.srp_ecf, (self.N_pulses, 1)),
                "SC0": np.full(self.N_pulses, f_start),
                "SCSS": np.full(self.N_pulses, f_step)
            }
            
            return CPHDChannelData(
                identifier=channel_name,
                tx_pol="V",
                rcv_pol="V",
                fxc=self.fxc,
                fxbw=self.fxbw,
                signal=signal,
                pvp=pvp
            )
            
        def close(self):
            pass
            
    return MockCPHDReader()

@pytest.fixture(params=["cpu", "cuda"])
def pfa_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param
