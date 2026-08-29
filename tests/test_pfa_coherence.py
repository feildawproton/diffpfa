import torch
import numpy as np
import pytest
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.types import CPHDMetadata, ImageAreaBounds

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_point_target_localization():
    fc = 9.6e9
    bw_rf = 600e6
    num_samples = 256
    num_pulses = 256
    squint_span_deg = 2.0
    slant_range = 10000.0

    f_start = fc - bw_rf / 2.0
    f_step = bw_rf / num_samples
    f_vec = f_start + np.arange(num_samples) * f_step
    theta_vec = np.linspace(-np.radians(squint_span_deg / 2), np.radians(squint_span_deg / 2), num_pulses)

    uIAX = np.array([1.0, 0.0, 0.0])
    uIAY = np.array([0.0, 1.0, 0.0])
    srp_ecf = np.array([0.0, 0.0, 0.0])

    tx_pos = np.zeros((num_pulses, 3))
    for i, th in enumerate(theta_vec):
        tx_pos[i] = np.array([slant_range * np.sin(th), -slant_range * np.cos(th), 0.0])

    pvp = {
        "SRPPos": np.tile(srp_ecf, (num_pulses, 1)),
        "TxPos": tx_pos,
        "TxVel": np.tile(np.array([100.0, 0.0, 0.0]), (num_pulses, 1)),
        "RcvPos": tx_pos,
        "RcvVel": np.tile(np.array([100.0, 0.0, 0.0]), (num_pulses, 1)),
        "SC0": np.full(num_pulses, f_start),
        "SCSS": np.full(num_pulses, f_step),
        "RcvTime": np.linspace(0.0, 1.0, num_pulses)
    }

    # Point target at (+5.0m, -5.0m)
    u_t, r_t = 5.0, -5.0
    signal = np.zeros((num_pulses, num_samples), dtype=np.complex64)
    for n in range(num_pulses):
        th = theta_vec[n]
        cos_pfa = -np.sin(th)
        sin_pfa = np.cos(th)
        phase = -2.0 * np.pi * (2.0 * f_vec / SPEED_OF_LIGHT) * (u_t * cos_pfa + r_t * sin_pfa)
        signal[n] = np.exp(1j * phase).astype(np.complex64)

    meta = CPHDMetadata(
        domain_type="FX",
        sgn=-1,
        global_fx_min=f_start,
        global_fx_max=f_start + num_samples * f_step,
        iarp_ecf=srp_ecf,
        uIAX=uIAX,
        uIAY=uIAY,
        ref_ch_id="Primary",
        image_area=ImageAreaBounds(x1=-20.0, y1=-20.0, x2=20.0, y2=20.0, polygon=None),
        extended_area=None,
        collection_start=None,
        radar_mode="SPOTLIGHT",
        classification="UNCLASSIFIED",
        srp_ecf=srp_ecf,
        arp_pos_coa=tx_pos[num_pulses // 2],
        arp_vel_coa=np.array([100.0, 0.0, 0.0]),
        side_of_track="R",
        line_spacing=None,
        sample_spacing=None,
        raw_meta=None
    )

    img_cpu, bw_range, bw_azm, N_range, N_azm, _ = pfa_per_polar(
        channel_signals=[signal],
        channel_pvps=[pvp],
        channel_fxcs=[fc],
        channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"],
        cphd_meta=meta,
        u_min=-20.0, u_max=20.0,
        r_min=-20.0, r_max=20.0,
        image_oversample=1.25,
        device=device
    )

    du = 40.0 / N_azm
    dr = 40.0 / N_range
    mag = np.abs(img_cpu)
    peak_idx = np.unravel_index(np.argmax(mag), mag.shape)
    found_u = -20.0 + peak_idx[0] * du
    found_r = -20.0 + peak_idx[1] * dr

    loc_err = np.sqrt((found_u - u_t)**2 + (found_r - r_t)**2)
    assert loc_err < 0.35, f"Target location error {loc_err:.3f}m exceeds sub-pixel tolerance"
