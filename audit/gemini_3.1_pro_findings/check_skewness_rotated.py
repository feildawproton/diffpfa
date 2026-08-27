import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.types import CPHDMetadata

N_pulses, N_samples = 512, 512
vel = 100.0
t = np.linspace(-2.5, 2.5, N_pulses)
pos = np.zeros((N_pulses, 3))
# Flight path rotated by 30 degrees from Y axis
angle = np.deg2rad(30)
pos[:, 0] = vel * t * np.sin(angle)
pos[:, 1] = vel * t * np.cos(angle)
pos[:, 2] = 10000.0

srp_ecf = np.array([10000.0, 0.0, 0.0])
fc = 10e9
fxbw = 500e6
f_step = fxbw / N_samples
f_start = fc - fxbw/2

dist = np.linalg.norm(srp_ecf - pos, axis=1)
phase = np.zeros((N_pulses, N_samples))
signal = np.exp(1j * phase).astype(np.complex64)

pvp = {
    "TxPos": pos, "RcvPos": pos, "SRPPos": np.tile(srp_ecf, (N_pulses, 1)),
    "SC0": np.full(N_pulses, f_start), "SCSS": np.full(N_pulses, f_step),
    "TxTime": t, "RcvTime": t + 2*dist/3e8, "TxFMRate": np.full(N_pulses, 0.0)
}

meta = CPHDMetadata(
    domain_type="FX", sgn=-1, global_fx_min=f_start, global_fx_max=f_start + fxbw,
    iarp_ecf=srp_ecf, uIAX=np.array([0.0, 1.0, 0.0]), uIAY=np.array([1.0, 0.0, 0.0]),
    ref_ch_id="Ch1", image_area=None, extended_area=None, collection_start=None,
    radar_mode="SPOTLIGHT", classification="U", srp_ecf=srp_ecf,
    arp_pos_coa=pos[N_pulses//2], arp_vel_coa=np.array([0.0, vel, 0.0]),
    line_spacing=None, sample_spacing=None, raw_meta=None
)

img_cpu, bw_u, bw_r, N_u, N_r = pfa_per_polar(
    channel_signals=[signal], channel_pvps=[pvp], channel_fxcs=[fc],
    channel_domain_types=["FX"], ref_rcv_time=pvp["RcvTime"], cphd_meta=meta,
    u_min=-10, u_max=10, r_min=-10, r_max=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

img_mag = np.abs(img_cpu)
max_idx = np.unravel_index(np.argmax(img_mag), img_mag.shape)

window = 16
crop = img_mag[max_idx[0]-window:max_idx[0]+window, max_idx[1]-window:max_idx[1]+window]
y, x = np.mgrid[0:2*window, 0:2*window]
total = crop.sum()
x_c = (x * crop).sum() / total
y_c = (y * crop).sum() / total
mxx = ((x - x_c)**2 * crop).sum() / total
myy = ((y - y_c)**2 * crop).sum() / total
mxy = ((x - x_c) * (y - y_c) * crop).sum() / total

print(f"Moments: mxx={mxx:.3f}, myy={myy:.3f}, mxy={mxy:.3f}")
eigvals = np.linalg.eigvals([[mxx, mxy], [mxy, myy]])
print(f"Eigenvalues: {eigvals}")
