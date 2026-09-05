import numpy as np

fc_global = 9.6e9
bw_total = 600e6
num_subbands = 3
bw_sub = bw_total / num_subbands

for delta_tau in [150e-6, 155e-6, 143.2e-6]:
    print(f"\n--- Testing delta_tau = {delta_tau * 1e6:.1f} us ---")
    for m in range(num_subbands):
        f_sub_center = (fc_global - bw_total / 2.0) + (m + 0.5) * bw_sub
        tau = m * delta_tau
        df = fc_global - f_sub_center
        cycles = df * tau
        phase_rad = -2.0 * np.pi * cycles
        phase_mod_2pi = phase_rad % (2 * np.pi)
        print(f"  Subband {m}: df={df/1e6:+.1f} MHz, tau={tau*1e6:.1f} us -> cycles = {cycles:.4f}, phase mod 2pi = {phase_mod_2pi:.4f} rad")
