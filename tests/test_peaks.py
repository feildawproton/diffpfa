import sys, os
sys.path.insert(0, os.path.abspath('.'))
from diffpfa.algo.pfa_engine import PFAEngine, PFAConfig
from simulation.simulate_stepped_chirp import MockSteppedChirpCPHDReader, TrackingSICDWriter
import torch, numpy as np

for mode in ["cztnufft", "nufft"]:
    reader = MockSteppedChirpCPHDReader()
    writer = TrackingSICDWriter(mode)
    config = PFAConfig(mode=mode, device="cuda", num_subpatches=1, align_subchannels=True, output_dir=f"workspace/simulation/output/{mode}")
    engine = PFAEngine(reader, writer, config)
    engine.run()
    vv = next(p for p in writer.payloads if p.tx_pol == "V" and p.rcv_pol == "V" and p.channel_id is None)
    img = torch.abs(vv.complex_image).cpu().numpy()
    c_u = img.shape[0]//2
    c_r = img.shape[1]//2
    slice_r = img[c_u, :]
    slice_u = img[:, c_r]
    print(f"Mode {mode} peak_r: {np.argmax(slice_r)}, peak_u: {np.argmax(slice_u)}")
