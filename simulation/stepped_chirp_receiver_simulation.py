"""
Stepped-Chirp Receiver & CPHD Compensation Simulation for diffpfa
================================================================
Models the radar receiver's local oscillator (LO), per-pulse LO phase reset,
range compression, and the CPHD producer's SRP compensation explicitly (CPHD DIDD §1.4).

Demonstrates:
1. Compliant CPHD channels have exactly zero phase at the SRP across all pulses.
2. Multi-channel stepped-chirp coherence is independent of inter-pulse burst delay (delta_tau).
3. Inter-band phase diagnostic for assessing subband alignment on real CPHD data.
"""

import os
import sys
import numpy as np
import torch

from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT as C
from diffpfa.types import CPHDMetadata, ImageAreaBounds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FC, BW_TOTAL, NSUB = 9.6e9, 600e6, 3
NS_SUB, NPULSE, PRF = 256, 512, 1000.0
SLANT_RANGE, SPAN_DEG, SAT_VEL = 15000.0, 2.0, 7500.0
TARGETS = [(0.0, 0.0, 1.0), (5.0, 8.0, 1.0), (-7.0, -12.0, 1.0), (10.0, -5.0, 1.0)]
L = 40.0
uIAX, uIAY, SRP = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3)


def platform(t):
    t_dur = (NPULSE - 1) / PRF
    th = -np.radians(SPAN_DEG / 2.0) + (t / t_dur) * np.radians(SPAN_DEG)
    return np.stack([SLANT_RANGE * np.sin(th), -SLANT_RANGE * np.cos(th), np.zeros_like(th)], -1)


def look_components(pos):
    P = SRP[None, :] - pos
    Pn = np.linalg.norm(P, axis=1)
    return P @ uIAX / Pn, P @ uIAY / Pn, Pn


def upsample_cut(cut, factor=16):
    n = len(cut)
    F = np.fft.fftshift(np.fft.fft(cut))
    Fp = np.zeros(n * factor, complex)
    Fp[(n * factor - n) // 2 : (n * factor + n) // 2] = F
    return np.fft.ifft(np.fft.ifftshift(Fp)) * factor


def irw_pslr(cut, dx, factor=16):
    up = np.abs(upsample_cut(cut, factor))
    dxu = dx / factor
    i = int(np.argmax(up))
    hp = up[i] / np.sqrt(2.0)
    l = i
    while l > 0 and up[l] > hp:
        l -= 1
    r = i
    while r < len(up) - 1 and up[r] > hp:
        r += 1
    width = (r - l) * dxu

    l2 = i
    while l2 > 0 and up[l2 - 1] < up[l2]:
        l2 -= 1
    r2 = i
    while r2 < len(up) - 1 and up[r2 + 1] < up[r2]:
        r2 += 1
    side = max(up[:l2].max() if l2 > 0 else 0.0, up[r2 + 1 :].max() if r2 + 1 < len(up) else 0.0)
    return width, 20.0 * np.log10(side / max(up[i], 1e-12))


def make_channels(delta_tau, jitter_us=0.0, compliant=True, seed=0):
    rng = np.random.default_rng(seed)
    bw_sub = BW_TOTAL / NSUB
    t_burst = np.arange(NPULSE) / PRF
    chans = []
    for m in range(NSUB):
        f_m = (FC - BW_TOTAL / 2.0) + (m + 0.5) * bw_sub
        step = bw_sub / NS_SUB
        f_b = -bw_sub / 2.0 + np.arange(NS_SUB) * step
        fx = f_m + f_b
        t_mn = t_burst + m * delta_tau + rng.uniform(-jitter_us, jitter_us, NPULSE) * 1e-6
        pos = platform(t_mn)
        cos_t, sin_t, R_srp = look_components(pos)
        theta = np.mod(2.0 * np.pi * f_m * t_mn, 2.0 * np.pi)

        def raw_for(targets):
            r = np.zeros((NPULSE, NS_SUB), np.complex128)
            for (u_t, r_t, a) in targets:
                R_i = R_srp + (u_t * cos_t + r_t * sin_t)
                r += a * np.exp(-1j * 2.0 * np.pi * fx[None, :] * (2.0 * R_i[:, None] / C) - 1j * theta[:, None])
            return r

        raw = raw_for(TARGETS)
        raw_srp_only = raw_for([(0.0, 0.0, 1.0)])
        comp = np.exp(+1j * 2.0 * np.pi * fx[None, :] * (2.0 * R_srp[:, None] / C) + 1j * theta[:, None])
        theta_resid = 0.0
        if not compliant:
            theta_c = np.mod(2.0 * np.pi * f_m * (m * delta_tau), 2.0 * np.pi)
            comp = comp * np.exp(-1j * theta_c)
            theta_resid = theta_c
        sig = raw * comp
        pvp = {
            "SRPPos": np.tile(SRP, (NPULSE, 1)), "TxPos": pos, "RcvPos": pos,
            "TxVel": np.tile([SAT_VEL, 0.0, 0.0], (NPULSE, 1)), "RcvVel": np.tile([SAT_VEL, 0.0, 0.0], (NPULSE, 1)),
            "SC0": np.full(NPULSE, fx[0]), "SCSS": np.full(NPULSE, step),
            "TxTime": t_mn, "RcvTime": t_mn
        }
        chans.append({
            "sig": sig.astype(np.complex64), "pvp": pvp, "fxc": f_m, "theta_resid": theta_resid,
            "srp_phase_raw": np.angle(raw_srp_only[:, NS_SUB // 2]),
            "srp_phase_cphd": np.angle((raw_srp_only * comp)[:, NS_SUB // 2])
        })
    return chans, t_burst


def make_monolithic(t_burst):
    ns = NSUB * NS_SUB
    step = BW_TOTAL / ns
    fx = (FC - BW_TOTAL / 2.0) + np.arange(ns) * step
    pos = platform(t_burst)
    cos_t, sin_t, R_srp = look_components(pos)
    sig = np.zeros((NPULSE, ns), np.complex128)
    for (u_t, r_t, a) in TARGETS:
        sig += a * np.exp(-1j * 2.0 * np.pi * fx[None, :] * (2.0 * (u_t * cos_t + r_t * sin_t)[:, None] / C))
    pvp = {
        "SRPPos": np.tile(SRP, (NPULSE, 1)), "TxPos": pos, "RcvPos": pos,
        "TxVel": np.tile([SAT_VEL, 0.0, 0.0], (NPULSE, 1)), "RcvVel": np.tile([SAT_VEL, 0.0, 0.0], (NPULSE, 1)),
        "SC0": np.full(NPULSE, fx[0]), "SCSS": np.full(NPULSE, step),
        "TxTime": t_burst, "RcvTime": t_burst
    }
    return sig.astype(np.complex64), pvp


def create_meta(fmin, fmax, pos0):
    return CPHDMetadata(
        domain_type="FX", sgn=-1, global_fx_min=fmin, global_fx_max=fmax,
        iarp_ecf=SRP, uIAX=uIAX, uIAY=uIAY, ref_ch_id="0",
        image_area=ImageAreaBounds(-L / 2.0, -L / 2.0, L / 2.0, L / 2.0, None),
        extended_area=None, collection_start=None, radar_mode="SPOTLIGHT",
        classification="U", srp_ecf=SRP, arp_pos_coa=pos0,
        arp_vel_coa=np.array([SAT_VEL, 0.0, 0.0]), side_of_track="R",
        line_spacing=None, sample_spacing=None, raw_meta=None
    )


def form_image(chans, ref_rcv_time, spacing=None, phase_fix=None):
    sigs = [c["sig"] for c in chans]
    if phase_fix is not None:
        sigs = [s * np.exp(-1j * p).astype(np.complex64) for s, p in zip(sigs, phase_fix)]
    fmin = min(c["pvp"]["SC0"][0] for c in chans)
    fmax = max(c["pvp"]["SC0"][0] + NS_SUB * c["pvp"]["SCSS"][0] for c in chans)
    fxcs = [c["fxc"] for c in chans]
    cphd_meta = create_meta(fmin, fmax, chans[0]["pvp"]["TxPos"][NPULSE // 2])
    res = pfa_per_polar(
        channel_signals=sigs,
        channel_pvps=[c["pvp"] for c in chans],
        channel_fxcs=fxcs,
        channel_domain_types=["FX"] * len(chans),
        ref_rcv_time=ref_rcv_time,
        cphd_meta=cphd_meta,
        u_min=-L / 2.0, u_max=L / 2.0, r_min=-L / 2.0, r_max=L / 2.0,
        custom_pixel_spacing=spacing,
        image_oversample=1.25,
        device=DEV
    )
    img, bw_r, bw_u, N_r, N_u, _ = res
    return img, (L / N_u, L / N_r), bw_r


def main():
    print("=" * 80)
    print("STEPPED-CHIRP RECEIVER & CPHD COMPENSATION SIMULATION")
    print("=" * 80)

    print("\n[1] Compliant CPHD channels: SRP phase per channel (radians)")
    chans, t_burst = make_channels(137.3217e-6)
    for m, c in enumerate(chans):
        max_raw = np.abs(c["srp_phase_raw"]).max()
        max_cphd = np.abs(c["srp_phase_cphd"]).max()
        print(f"    Channel {m}: Raw LO-demodulated max|SRP phase| = {max_raw:.4f} rad -> After CPHD compensation = {max_cphd:.2e} rad")

    print("\n[2] Multi-channel stepped chirp coherence across non-integer cycle delays")
    print(f"    {'Delay (delta_tau)':<25} | {'Coherence':<10} | {'Range IRW (m)':<14} | {'PSLR (dB)':<10}")
    print("-" * 80)

    sweep_delays = [
        (150e-6, "150.0 us (integer cycles)"),
        (137e-6, "137.0 us"),
        (137.3217e-6, "137.3217 us (fractional)"),
        (100.00025e-6, "100.00025 us")
    ]

    for dtau, label in sweep_delays:
        chans_sweep, t_b = make_channels(dtau)
        ref_sig, ref_pvp = make_monolithic(t_b)
        img_multi, spacing, _ = form_image(chans_sweep, chans_sweep[0]["pvp"]["RcvTime"])
        img_ref, _, _ = form_image([{"sig": ref_sig, "pvp": ref_pvp, "fxc": FC}], t_b, spacing=spacing)

        coh = float(abs(np.vdot(img_multi, img_ref)) / (np.linalg.norm(img_multi) * np.linalg.norm(img_ref)))
        cut = img_multi[img_multi.shape[0] // 2, :]
        irw, pslr = irw_pslr(cut, spacing[1])
        print(f"    {label:<25} | {coh:<10.6f} | {irw:<14.4f} | {pslr:<10.2f}")

    print("\n[3] Inter-band phase diagnostic (Single subband imaging of SRP scatterer)")
    ph = []
    _, spacing_diag, _ = form_image(chans, t_burst)
    for m, c in enumerate(chans):
        im, _, _ = form_image([c], t_burst, spacing=spacing_diag)
        ph.append(float(np.angle(im[im.shape[0] // 2, im.shape[1] // 2])))
    print("    SRP phase per subband image (deg):", ", ".join(f"{np.degrees(p):+.2f}°" for p in ph))
    print("    -> Inter-band phase differences are ~0°: subbands are naturally coherent at SRP.")
    print("=" * 80)


if __name__ == "__main__":
    main()
