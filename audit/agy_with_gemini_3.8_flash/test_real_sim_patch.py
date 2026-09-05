import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import simulation.stepped_chirp_simulation as sim_mod

print("Testing with correctly patched sim_mod.pfa_per_polar...")

orig_pfa = sim_mod.pfa_per_polar

delta_tau_frac = 150e-6 + 1.25e-9

def broken_pfa(*args, **kwargs):
    channel_pvps = kwargs.get("channel_pvps", args[1] if len(args) > 1 else None)
    for p in channel_pvps:
        p["RcvTime"] = kwargs["ref_rcv_time"].copy()
    return orig_pfa(*args, **kwargs)

print("--- 1. WITH phase_corr ---")
sim_mod.run_stepped_chirp_simulation(delta_tau=delta_tau_frac)

print("\n--- 2. WITHOUT phase_corr (tau forced to 0) ---")
sim_mod.pfa_per_polar = broken_pfa
try:
    sim_mod.run_stepped_chirp_simulation(delta_tau=delta_tau_frac)
finally:
    sim_mod.pfa_per_polar = orig_pfa
