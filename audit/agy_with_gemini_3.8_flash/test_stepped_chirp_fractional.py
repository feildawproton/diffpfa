import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from simulation.stepped_chirp_simulation import run_stepped_chirp_simulation
import diffpfa.IFA.PFA as pfa_mod

delta_tau_frac = 150e-6 + 1.25e-9

print("=== 1. Running stepped-chirp simulation WITH phase_corr (delta_tau = 150.00125 us) ===")
run_stepped_chirp_simulation(delta_tau=delta_tau_frac)

print("\n=== 2. Running stepped-chirp simulation WITHOUT phase_corr (forced tau=0) ===")
orig_pfa = pfa_mod.pfa_per_polar
def broken_pfa(*args, **kwargs):
    channel_pvps = kwargs.get("channel_pvps", args[1] if len(args) > 1 else None)
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
    run_stepped_chirp_simulation(delta_tau=delta_tau_frac)
finally:
    pfa_mod.pfa_per_polar = orig_pfa
