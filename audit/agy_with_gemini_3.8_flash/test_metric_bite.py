import torch
import numpy as np
from simulation.stepped_chirp_simulation import run_stepped_chirp_simulation
from unittest.mock import patch

print("Testing stepped-chirp coherence sensitivity to broken physics...")

# Let's test what happens when the phase correction phi_corr is disabled (set to 0)
import diffpfa.IFA.PFA as pfa_mod

orig_pfa_per_polar = pfa_mod.pfa_per_polar

# We will run a minimal version of stepped chirp simulation with and without phase correction
from diffpfa.constants import SPEED_OF_LIGHT

fc_global = 9.6e9
bw_total = 600e6
num_subbands = 3
samples_per_sub = 128
num_pulses = 128
delta_tau = 150e-6
sat_vel = 7500.0

# 1. Correct physics
print("\n1. Running with CORRECT physics:")
# We can import and run with smaller dimensions to be fast
from diffpfa.IFA.channel.czt_torch import czt_resample_kspace_1d
from diffpfa.IFA.channel.nufft_torch import nufft_grid_1d

# Let's inspect the coherence difference directly
# We compute subband 0, 1, 2 phase differences at delta_tau:
# f_c0 = 9.5e9, f_c1 = 9.6e9, f_c2 = 9.7e9
# delta_f = 100 MHz
# phase shift = 2 * pi * 100e6 * 150e-6 = 2 * pi * 15000 radians = 30,000 pi radians!
# If this phase is uncompensated, subbands add completely incoherently.
phase_shift_ch0 = -2.0 * np.pi * (fc_global - 9.5e9) * delta_tau
print(f"Phase correction on subband 0: {phase_shift_ch0:.2f} rad (mod 2pi: {phase_shift_ch0 % (2*np.pi):.4f})")
print(f"Platform motion during delta_tau: {sat_vel * delta_tau:.3f} m")

# Let's measure coherence if phase_corr = 0
with patch.object(pfa_mod, 'pfa_per_polar') as mock_pfa:
    pass
