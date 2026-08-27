import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.types import CPHDMetadata

# Simulate a simple dataset
N_pulses = 512
N_samples = 512

# Geometry: Broadside, radar flying along Y axis, looking along X axis
# So scene is at X = 10000, Y = 0, Z = 0
# Radar is at X = 0, Y = vt, Z = 10000
vel = 100.0
t = np.linspace(-2.5, 2.5, N_pulses)
pos = np.zeros((N_pulses, 3))
pos[:, 1] = vel * t
pos[:, 2] = 10000.0

srp_ecf = np.array([10000.0, 0.0, 0.0])

# Frequencies
fc = 10e9
fxbw = 500e6
f_step = fxbw / N_samples
f_start = fc - fxbw/2

dist = np.linalg.norm(srp_ecf - pos, axis=1)

# Point target at SRP
phase = np.zeros((N_pulses, N_samples))
signal = np.exp(1j * phase).astype(np.complex64)

pvp = {
    "TxPos": pos,
    "RcvPos": pos,
    "SRPPos": np.tile(srp_ecf, (N_pulses, 1)),
    "SC0": np.full(N_pulses, f_start),
    "SCSS": np.full(N_pulses, f_step),
    "TxTime": t,
    "RcvTime": t + 2*dist/3e8,
    "TxFMRate": np.full(N_pulses, 0.0) # Set to 0 to bypass buggy RVP for this first test
}

meta = CPHDMetadata(
    domain_type="FX",
    sgn=-1,
    global_fx_min=f_start,
    global_fx_max=f_start + fxbw,
    iarp_ecf=srp_ecf,
    uIAX=np.array([0.0, 1.0, 0.0]), # Azimuth (Y)
    uIAY=np.array([1.0, 0.0, 0.0]), # Range (X)
    ref_ch_id="Ch1",
    image_area=None,
    extended_area=None,
    collection_start=None,
    radar_mode="SPOTLIGHT",
    classification="U",
    srp_ecf=srp_ecf,
    arp_pos_coa=pos[N_pulses//2],
    arp_vel_coa=np.array([0.0, vel, 0.0]),
    line_spacing=None,
    sample_spacing=None,
    raw_meta=None
)

ref_rcv_time = pvp["RcvTime"]

u_min, u_max = -10, 10
r_min, r_max = -10, 10

print("Running PFA...")
img_cpu, bw_u, bw_r, N_u, N_r = pfa_per_polar(
    channel_signals=[signal],
    channel_pvps=[pvp],
    channel_fxcs=[fc],
    channel_domain_types=["FX"],
    ref_rcv_time=ref_rcv_time,
    cphd_meta=meta,
    u_min=u_min, u_max=u_max, r_min=r_min, r_max=r_max,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

img_mag = np.abs(img_cpu)
plt.imshow(img_mag, extent=[u_min, u_max, r_min, r_max], cmap='gray', origin='lower')
plt.title("IPR on Ground Plane (No RVP bug)")
plt.savefig("/home/feildaw/diffpfa/audit/ipr_ground.png")
print("Saved /home/feildaw/diffpfa/audit/ipr_ground.png")
