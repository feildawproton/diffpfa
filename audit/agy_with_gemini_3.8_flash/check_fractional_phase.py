import numpy as np
import torch
from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.IFA.channel.pfa_channel import process_cztnufft
from diffpfa.IFA.PFA import _apply_ifft_and_deconv
from diffpfa.IFA.kspace import compute_kspace

print("Testing stepped-chirp coherence with fractional cycle inter-pulse delays...")

fc_global = 9.6e9
bw_total = 600e6
num_subbands = 3
bw_sub = bw_total / num_subbands
samples_per_sub = 128
num_pulses = 128

# Choose delta_tau so that delta_f * delta_tau has a fractional cycle of 0.25 (90 degrees phase shift)
# delta_f = 200 MHz. delta_f * delta_tau = 200e6 * delta_tau.
# If delta_tau = 150e-6 + 0.25 / 200e6 = 150e-6 + 1.25e-9 = 150.00125 microseconds:
delta_tau = 150e-6 + 1.25e-9
df = 200e6
cycles = df * delta_tau
print(f"delta_tau = {delta_tau*1e6:.6f} us")
print(f"cycles = {cycles:.4f}, fraction = {cycles % 1.0:.4f}")
print(f"Phase shift = {(-2.0 * np.pi * cycles) % (2 * np.pi):.4f} rad ({np.degrees((-2.0 * np.pi * cycles) % (2 * np.pi)):.1f} deg)")
