import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.simulate_stepped_chirp import MockSteppedChirpCPHDReader
from diffpfa.IFA.kspace import compute_kspace

reader = MockSteppedChirpCPHDReader()
ch_data = reader.read_channel("Ch1_VV_Low")
meta = reader.get_metadata()

pvp = ch_data.pvp
Ku, Kr = compute_kspace(pvp, meta.uIAX, meta.uIAY, ch_data.signal.shape[1], domain_type="FX")

Ku = Ku.numpy()
Kr = Kr.numpy()

# Check the bounds
print("Ku min:", Ku.min(), "Ku max:", Ku.max())
print("Kr min:", Kr.min(), "Kr max:", Kr.max())

# Check if center of Kr shifts with Ku (skew)
mid_r = ch_data.signal.shape[1] // 2
mid_u = Ku.shape[0] // 2

Ku_mid = Ku[:, mid_r]
Kr_mid = Kr[:, mid_r]

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Ku.flatten(), Kr.flatten(), s=1)
plt.title("K-space Support on Ground Plane")
plt.xlabel("Ku")
plt.ylabel("Kr")
plt.savefig("/home/feildaw/diffpfa/audit/kspace_support.png")
print("Saved K-space plot to /home/feildaw/diffpfa/audit/kspace_support.png")

# Calculate the slope of the center line (Kr vs Ku)
slope, intercept = np.polyfit(Ku_mid, Kr_mid, 1)
print(f"Slope of Kr vs Ku (skew): {slope}")
