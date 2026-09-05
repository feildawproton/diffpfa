import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from simulation.stepped_chirp_simulation import run_stepped_chirp_simulation
import diffpfa.IFA.PFA as pfa_mod

print("Testing bite on stepped-chirp coherence with phase_corr removed...")

orig_pfa = pfa_mod.pfa_per_polar

# We patch pfa_per_polar or simulate with phase_corr corrupted
# Let's see what happens inside stepped_chirp_simulation
# In PFA.py line 145:
# phase_corr = -2.0 * torch.pi * (fc_global - fxc) * tau_tensor
# If we set phase_corr = 0:
def broken_pfa(*args, **kwargs):
    # Call original but monkeypatch torch.exp in that scope, or patch tau
    # Better: just set ref_rcv_time to None or corrupt channel_pvps
    channel_pvps = kwargs.get("channel_pvps", args[1] if len(args) > 1 else None)
    # If we corrupt RcvTime to equal ref_rcv_time, tau becomes 0 (no motion compensation)
    saved_rcv_times = [p["RcvTime"].copy() for p in channel_pvps]
    for p in channel_pvps:
        p["RcvTime"] = kwargs["ref_rcv_time"].copy()
    try:
        return orig_pfa(*args, **kwargs)
    finally:
        for p, t in zip(channel_pvps, saved_rcv_times):
            p["RcvTime"] = t

pfa_mod.pfa_per_polar = broken_pfa

try:
    print("\nRunning simulation with UNCOMPENSATED platform motion (tau=0):")
    # capture output
    run_stepped_chirp_simulation()
finally:
    pfa_mod.pfa_per_polar = orig_pfa
