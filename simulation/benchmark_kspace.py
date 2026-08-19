import os
import sys
import numpy as np
import torch
import time
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffpfa.algo.pfa_engine import PFAEngine, PFAConfig
from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.io.base import BaseCPHDReader, CPHDMetadata, CPHDChannelData, BaseSICDWriter, SICDImagePayload

N_SAMPLES_G = 4096
N_CHANNELS_G = 8

class BenchmarkCPHDReader(BaseCPHDReader):
    def __init__(self, file_path="benchmark.cphd"):
        super().__init__(file_path)
        
        self.N_samples = N_SAMPLES_G
        self.N_pulses = N_SAMPLES_G
        
        self.srp_ecf = np.array([0.0, 0.0, 0.0])
        vel = 100.0  # m/s
        PRF = 1000.0 # Hz 
        PRI = 1.0 / PRF
        
        self.fxbw = 250e6
        self.f_step = self.fxbw / self.N_samples
        
        self.channels = []
        for i in range(N_CHANNELS_G):
            t_i = np.arange(self.N_pulses) * (2 * PRI) + (i * PRI)
            pos_i = np.zeros((self.N_pulses, 3))
            pos_i[:, 0] = 10000.0
            pos_i[:, 1] = vel * (t_i - np.mean(t_i))
            pos_i[:, 2] = 1000.0
            
            fxc_i = 10.0e9 + i * self.fxbw
            f_start_i = fxc_i - self.fxbw / 2
            
            self.channels.append({
                "id": f"Ch{i}_VV", "tx": "V", "rcv": "V", 
                "pos": pos_i, "fxc": fxc_i, "f_start": f_start_i, "t": t_i
            })
            
        self.arp_pos_coa = self.channels[0]["pos"][self.N_pulses // 2]
        self.arp_vel_coa = np.array([0.0, vel, 0.0])
        self.global_fx_min = self.channels[0]["f_start"]
        self.global_fx_max = self.channels[-1]["f_start"] + self.fxbw
        
    def get_metadata(self) -> CPHDMetadata:
        return CPHDMetadata(
            domain_type="FX",
            sgn=-1,
            global_fx_min=self.global_fx_min,
            global_fx_max=self.global_fx_max,
            iarp_ecf=np.array([0.0, 0.0, 0.0]),
            uIAX=np.array([0.0, 1.0, 0.0]), 
            uIAY=np.array([1.0, 0.0, 0.0]), 
            srp_ecf=self.srp_ecf,
            arp_pos_coa=self.arp_pos_coa,
            arp_vel_coa=self.arp_vel_coa,
            classification="UNCLASSIFIED"
        )
        
    def get_channel_names(self):
        return [ch["id"] for ch in self.channels]
        
    def read_channel(self, channel_name: str) -> CPHDChannelData:
        ch_info = next(ch for ch in self.channels if ch["id"] == channel_name)
        pos = ch_info["pos"]
        
        # We don't need a real signal for speed benchmarking, just a complex tensor of ones
        signal = torch.ones((self.N_pulses, self.N_samples), dtype=torch.complex64)
        
        t_arr = ch_info["t"]
        dist = np.linalg.norm(pos, axis=1)
        tau_roundtrip = 2.0 * dist / SPEED_OF_LIGHT
        
        pvp = {
            "TxPos": pos,
            "RcvPos": pos,
            "SRPPos": np.tile(self.srp_ecf, (self.N_pulses, 1)),
            "SC0": np.full(self.N_pulses, ch_info["f_start"]),
            "SCSS": np.full(self.N_pulses, self.f_step),
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


class NullSICDWriter(BaseSICDWriter):
    # Dummy writer that does nothing to avoid I/O bottlenecks in the benchmark
    def write_sicd(self, output_path: str, payload: SICDImagePayload, cphd_meta: CPHDMetadata) -> str:
        return output_path
    def finalize(self): pass
    def close(self): pass


def run_benchmark(combine_in_kspace: bool):
    reader = BenchmarkCPHDReader()
    writer = NullSICDWriter()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PFAConfig(
        device=device,
        combine_in_kspace=combine_in_kspace,
        debug_save_channels=False,
        output_dir="workspace/simulation/output"
    )
    
    engine = PFAEngine(reader, writer, config)
    
    # Run a tiny warmup if CUDA
    if device == "cuda":
        torch.cuda.empty_cache()
        dummy = torch.randn(10, 10, device=device)
        _ = torch.fft.ifft2(dummy)
        torch.cuda.synchronize()
        
    gc.collect()
    
    print(f"Running PFAEngine (K-Space Combine: {combine_in_kspace}) - {N_CHANNELS_G} Channels of size {N_SAMPLES_G}x{N_SAMPLES_G}...")
    start_time = time.time()
    
    engine.run()
    
    if device == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"  -> Execution Time: {duration:.2f} seconds\n")
    return duration

if __name__ == '__main__':
    print(f"\\n{'='*60}\\nStarting Benchmark: K-Space vs Image-Space Subband Summation\\n{'='*60}")
    
    # Run once to warm up PyTorch entirely
    print("Warming up PyTorch JIT and Allocator (Tiny Run)...")
    N_SAMPLES_G = 256
    N_CHANNELS_G = 2
    run_benchmark(combine_in_kspace=False)
    
    N_SAMPLES_G = 8192
    N_CHANNELS_G = 8
    
    # Benchmark Image Combine
    t_image = run_benchmark(combine_in_kspace=False)
    
    # Benchmark K-Space Combine
    t_kspace = run_benchmark(combine_in_kspace=True)
    
    print(f"{'='*60}\\nRESULTS:\\n{'='*60}")
    print(f"Image-Space Combine : {t_image:.2f} s")
    print(f"K-Space Combine     : {t_kspace:.2f} s")
    
    if t_kspace < t_image:
        speedup = t_image / t_kspace
        print(f"\\nConclusion: K-Space Combine is {speedup:.2f}x FASTER.")
    else:
        slowdown = t_kspace / t_image
        print(f"\\nConclusion: K-Space Combine is {slowdown:.2f}x SLOWER.")
