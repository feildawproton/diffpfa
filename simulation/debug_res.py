import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffpfa.algo.pfa_engine import PFAEngine, PFAConfig
from diffpfa.io.base import BaseCPHDReader, CPHDMetadata, CPHDChannelData, BaseSICDWriter, SICDImagePayload
from simulation.simulate_stepped_chirp import MockSteppedChirpCPHDReader, SimWriter

def debug_res():
    reader = MockSteppedChirpCPHDReader()
    writer = SimWriter()
    
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
    slice_c = img_combined[center_u, :]
    slice_s = img_single[center_u, :]
    
    slice_c /= slice_c.max()
    slice_s /= slice_s.max()
    
    center_r = np.argmax(slice_s)
    
    print("\\nSingle Channel slice:")
    print(slice_s[center_r-10:center_r+11])
    
    print("\\nCombined Channel slice:")
    print(slice_c[center_r-10:center_r+11])
    
    fwhm_s = np.sum(slice_s > 0.5)
    fwhm_c = np.sum(slice_c > 0.5)
    print(f"FWHM Single: {fwhm_s}")
    print(f"FWHM Combined: {fwhm_c}")

if __name__ == '__main__':
    debug_res()
