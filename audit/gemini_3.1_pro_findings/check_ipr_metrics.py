import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.types import CPHDMetadata

def run_sim(use_rvp=False):
    N_pulses, N_samples = 512, 512
    vel = 100.0
    t = np.linspace(-2.5, 2.5, N_pulses)
    pos = np.zeros((N_pulses, 3))
    pos[:, 1] = vel * t
    pos[:, 2] = 10000.0

    srp_ecf = np.array([10000.0, 0.0, 0.0])
    fc = 10e9
    fxbw = 500e6
    f_step = fxbw / N_samples
    f_start = fc - fxbw/2
    gamma = fxbw / 1e-4 if use_rvp else 0.0

    dist = np.linalg.norm(srp_ecf - pos, axis=1)
    phase = np.zeros((N_pulses, N_samples))
    signal = np.exp(1j * phase).astype(np.complex64)

    pvp = {
        "TxPos": pos, "RcvPos": pos, "SRPPos": np.tile(srp_ecf, (N_pulses, 1)),
        "SC0": np.full(N_pulses, f_start), "SCSS": np.full(N_pulses, f_step),
        "TxTime": t, "RcvTime": t + 2*dist/3e8, "TxFMRate": np.full(N_pulses, gamma)
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
    u_vals = np.linspace(-10, 10, img_mag.shape[0])
    r_vals = np.linspace(-10, 10, img_mag.shape[1])
    return np.max(img_mag), u_vals[max_idx[0]], r_vals[max_idx[1]]

m1, u1, r1 = run_sim(use_rvp=False)
m2, u2, r2 = run_sim(use_rvp=True)

print(f"NO RVP: Max={m1:.2f} at U={u1:.2f}, R={r1:.2f}")
print(f"W/ RVP: Max={m2:.2f} at U={u2:.2f}, R={r2:.2f}")

