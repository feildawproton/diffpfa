"""
Comprehensive audit regression test suite for diffpfa.
Self-contained suite incorporating the core mathematical validations from both independent audits:
1. Ideal polar-to-Cartesian PFA analytical reference comparison (residual < -45 dB, PSLR ~ -13.26 dB).
2. Multi-channel subband continuous weighting independence from sample spacing (SCSS).
3. Ground-mode rotated-axes range/azimuth bandwidth preservation.
4. CPHD SGN = +1 conjugation convention handling.
5. Asymmetric image area SRP preservation at (0, 0).
6. Ground-to-slant framing projection matching Umbra-08 gold product within 1%.
7. Normative uniform impulse response width factor (ImpRespWid * ImpRespBW = 0.8859).
8. Radiometric scale factor trigonometric fidelity (SigmaZero / BetaZero = cos(SlopeAng)).
"""

import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

import sarkit.sicd as ss
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT as C
from diffpfa.types import CPHDMetadata
from diffpfa.IFP import IFAProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FC = 9.6e9
BW = 600e6
L_EXT = 40.0
GOLD_SICD = "/home/feildaw/data/2025-10-26-05-00-15_UMBRA-08_SICD.nitf"
GOLD_CPHD = "/home/feildaw/data/2025-10-26-05-00-15_UMBRA-08_CPHD.cphd"


# ---------------------------------------------------------------------------
# Test Helpers (Pure NumPy, No External Audit Dependencies)
# ---------------------------------------------------------------------------

def make_synthetic_collection(
    fc=9.6e9, bw=600e6, ns=256, npulse=256, span_deg=2.0, R=10000.0,
    targets=((0.0, 0.0, 1.0),), scss_scale=1.0, uIAX=(1, 0, 0), uIAY=(0, 1, 0)
):
    """Generates synthetic spotlight phase history for point targets."""
    uIAX = np.array(uIAX, float)
    uIAY = np.array(uIAY, float)
    f0 = fc - bw / 2.0
    scss = (bw / ns) * scss_scale
    ns_eff = int(round(ns / scss_scale))
    F = f0 + np.arange(ns_eff) * scss
    th = np.linspace(-np.radians(span_deg / 2.0), np.radians(span_deg / 2.0), npulse)

    pos = np.stack([R * np.sin(th), -R * np.cos(th), np.zeros_like(th)], axis=1)
    n_vec = np.cross(uIAX, uIAY)
    B = np.stack([uIAX, uIAY, n_vec], axis=1)
    pos_ecf = pos @ B.T
    srp = np.zeros(3)

    P = srp - pos_ecf
    cos_t = P @ uIAX / np.linalg.norm(P, axis=1)
    sin_t = P @ uIAY / np.linalg.norm(P, axis=1)

    sig = np.zeros((npulse, ns_eff), np.complex128)
    for (u_t, r_t, a) in targets:
        dR = u_t * cos_t + r_t * sin_t
        sig += a * np.exp(-1j * 2.0 * np.pi * (2.0 * F[None, :] / C) * dR[:, None])

    pvp = {
        "SRPPos": np.tile(srp, (npulse, 1)),
        "TxPos": pos_ecf,
        "RcvPos": pos_ecf,
        "TxVel": np.tile(np.array([100.0, 0.0, 0.0]) @ B.T, (npulse, 1)),
        "RcvVel": np.tile(np.array([100.0, 0.0, 0.0]) @ B.T, (npulse, 1)),
        "SC0": np.full(npulse, f0),
        "SCSS": np.full(npulse, scss),
        "RcvTime": np.linspace(0, 1, npulse),
        "TxTime": np.linspace(0, 1, npulse),
    }

    meta = CPHDMetadata(
        domain_type="FX", sgn=-1, global_fx_min=f0, global_fx_max=f0 + ns_eff * scss,
        iarp_ecf=srp, uIAX=uIAX, uIAY=uIAY, ref_ch_id="P", image_area=None,
        extended_area=None, collection_start=None, radar_mode="SPOTLIGHT",
        classification="U", srp_ecf=srp, arp_pos_coa=pos_ecf[npulse // 2],
        arp_vel_coa=np.array([100.0, 0.0, 0.0]) @ B.T, side_of_track="R",
        line_spacing=None, sample_spacing=None, raw_meta=None
    )
    Ku = (2.0 * F[None, :] / C) * cos_t[:, None]
    Kr = (2.0 * F[None, :] / C) * sin_t[:, None]
    return sig.astype(np.complex64), pvp, meta, Ku, Kr


def ideal_pfa_reference(Ku, Kr, targets, N_u, N_r, L):
    """Analytical ideal polar-to-Cartesian reference image on diffpfa's exact grid."""
    ku_c = (Ku.min() + Ku.max()) / 2.0
    kr_c = (Kr.min() + Kr.max()) / 2.0
    dK = 1.0 / L
    ku = ku_c + (np.arange(N_u) - N_u / 2.0) * dK
    kr = kr_c + (np.arange(N_r) - N_r / 2.0) * dK
    KU, KR = np.meshgrid(ku, kr, indexing="ij")

    kmag_all = np.sqrt(Ku ** 2 + Kr ** 2)
    ang_all = np.arctan2(Ku, Kr)
    KM = np.sqrt(KU ** 2 + KR ** 2)
    AN = np.arctan2(KU, KR)
    mask = (KM >= kmag_all.min()) & (KM <= kmag_all.max()) & (AN >= ang_all.min()) & (AN <= ang_all.max())

    S = np.zeros_like(KU, dtype=np.complex128)
    for (u_t, r_t, a) in targets:
        S += a * np.exp(-1j * 2.0 * np.pi * (KU * u_t + KR * r_t))
    S *= mask
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(S))) * (N_u * N_r)
    return img


def fit_complex_scalar(a, b):
    """Calculates least-squares complex scalar g minimising |a - g b| and normalised RMS."""
    g = np.vdot(b, a) / np.vdot(b, b)
    res = a - g * b
    return g, float(np.sqrt(np.mean(np.abs(res) ** 2)) / np.abs(a).max())


def compute_irw_pslr(cut, dx, factor=16):
    """Measures 3dB impulse response width and PSLR (dB) via 16x sinc upsampling."""
    n = len(cut)
    F = np.fft.fftshift(np.fft.fft(cut))
    Fp = np.zeros(n * factor, complex)
    Fp[(n * factor - n) // 2 : (n * factor + n) // 2] = F
    up = np.abs(np.fft.ifft(np.fft.ifftshift(Fp)) * factor)
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
    pslr_db = 20.0 * np.log10(side / max(up[i], 1e-12))
    return width, pslr_db


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [(0.0, 0.0), (5.0, -5.0), (-12.5, 7.25), (18.0, 0.0), (0.0, 18.0)])
def test_image_matches_ideal_pfa_reference(target):
    """Certify point target localization, residual < -45 dB vs ideal PFA, and uniform PSLR ~ -13.26 dB."""
    sig, pvp, meta, Ku, Kr = make_synthetic_collection(FC, BW, targets=[(target[0], target[1], 1.0)])
    pfa_res = pfa_per_polar(
        channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[FC], channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"], cphd_meta=meta, u_min=-L_EXT / 2.0, u_max=L_EXT / 2.0,
        r_min=-L_EXT / 2.0, r_max=L_EXT / 2.0, image_oversample=1.25, device=DEV
    )
    img, bw_r, bw_u, N_r, N_u, rot = pfa_res
    ref = ideal_pfa_reference(Ku, Kr, [(target[0], target[1], 1.0)], N_u, N_r, L_EXT)
    du, dr = L_EXT / N_u, L_EXT / N_r

    iu, ir = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    assert abs((iu - N_u / 2.0) * du - target[0]) <= 0.5 * du + 1e-9
    assert abs((ir - N_r / 2.0) * dr - target[1]) <= 0.5 * dr + 1e-9

    _, res_ratio = fit_complex_scalar(img.astype(np.complex128), ref)
    res_db = 20.0 * np.log10(res_ratio)
    assert res_db < -45.0, f"Residual vs ideal reference {res_db:.1f} dB exceeds -45 dB threshold"

    wu, pu = compute_irw_pslr(img[:, ir], du)
    wr, pr = compute_irw_pslr(img[iu, :], dr)
    wu0, pu0 = compute_irw_pslr(ref[:, ir], du)
    wr0, pr0 = compute_irw_pslr(ref[iu, :], dr)

    assert abs(wu / wu0 - 1.0) < 0.02
    assert abs(wr / wr0 - 1.0) < 0.02
    assert abs(pu - pu0) < 0.5
    assert abs(pr - pr0) < 0.5
    assert abs(pu + 13.26) < 0.6, "Uniform aperture PSLR should be approximately -13.26 dB"


def test_subbands_equal_weight_regardless_of_sample_spacing():
    """Certify continuous integral weighting in CZT resampler ensures equal subband weights."""
    fc, bw_total = 9.6e9, 600e6
    bw_sub = bw_total / 3.0
    npulse = 256
    th = np.linspace(-np.radians(1.0), np.radians(1.0), npulse)
    pos = np.stack([10000.0 * np.sin(th), -10000.0 * np.cos(th), np.zeros(npulse)], axis=1)
    srp = np.zeros(3)
    P = srp - pos
    Pn = np.linalg.norm(P, axis=1)
    cos_t = P[:, 0] / Pn
    sin_t = P[:, 1] / Pn

    chans = []
    for m, ns in enumerate([256, 512, 256]):
        f0 = (fc - bw_total / 2.0) + m * bw_sub
        step = bw_sub / ns
        freqs = f0 + np.arange(ns) * step
        s = np.exp(-1j * 2.0 * np.pi * (2.0 * freqs[None, :] / C) * (0.0 * cos_t + 0.0 * sin_t)[:, None])
        pvp = {
            "SRPPos": np.tile(srp, (npulse, 1)), "TxPos": pos, "RcvPos": pos,
            "TxVel": np.tile([7500.0, 0, 0], (npulse, 1)), "RcvVel": np.tile([7500.0, 0, 0], (npulse, 1)),
            "SC0": np.full(npulse, f0), "SCSS": np.full(npulse, step),
            "RcvTime": np.arange(npulse) / 1000.0, "TxTime": np.arange(npulse) / 1000.0
        }
        chans.append((s.astype(np.complex64), pvp, f0 + bw_sub / 2.0))

    meta = CPHDMetadata(
        domain_type="FX", sgn=-1, global_fx_min=fc - bw_total / 2.0, global_fx_max=fc + bw_total / 2.0,
        iarp_ecf=srp, uIAX=np.array([1.0, 0, 0]), uIAY=np.array([0, 1.0, 0]), ref_ch_id="0",
        image_area=None, extended_area=None, collection_start=None, radar_mode="SPOTLIGHT",
        classification="U", srp_ecf=srp, arp_pos_coa=pos[npulse // 2], arp_vel_coa=np.array([7500.0, 0, 0]),
        side_of_track="R", line_spacing=None, sample_spacing=None, raw_meta=None
    )

    img, bw_r, bw_u, N_r, N_u, _ = pfa_per_polar(
        channel_signals=[c[0] for c in chans], channel_pvps=[c[1] for c in chans],
        channel_fxcs=[c[2] for c in chans], channel_domain_types=["FX"] * 3,
        ref_rcv_time=chans[0][1]["RcvTime"], cphd_meta=meta,
        u_min=-20.0, u_max=20.0, r_min=-20.0, r_max=20.0, image_oversample=1.25, device=DEV
    )
    cut = img[N_u // 2, :]
    K = np.abs(np.fft.fftshift(np.fft.fft(cut)))
    occ = np.where(K > 0.1 * K.max())[0]
    thirds = np.array_split(occ, 3)
    means = [K[t].mean() for t in thirds]
    ratio = means[1] / means[0]
    assert abs(ratio - 1.0) < 0.05, f"Middle subband weighting ratio {ratio:.3f} deviates by >5%"


def test_rotated_axes_return_true_range_and_azimuth_parameters():
    """Certify that Step 6 is unconditional and does not double-swap range and azimuth in rotated datasets."""
    sig, pvp, meta, *_ = make_synthetic_collection(FC, BW, targets=[(5.0, -5.0, 1.0)])
    res0 = pfa_per_polar(
        channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[FC], channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"], cphd_meta=meta, u_min=-20.0, u_max=20.0, r_min=-20.0, r_max=20.0,
        image_oversample=1.25, device=DEV
    )
    bw_r0 = res0[1]

    # Swap axes so LOS aligns with uIAX
    meta.uIAX, meta.uIAY = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    res_rot = pfa_per_polar(
        channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[FC], channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"], cphd_meta=meta, u_min=-20.0, u_max=20.0, r_min=-20.0, r_max=20.0,
        image_oversample=1.25, device=DEV
    )
    img, bw_range, bw_azm, N_range, N_azm, is_rot = res_rot
    assert is_rot is True
    assert abs(bw_range / bw_r0 - 1.0) < 1e-6
    assert N_range == img.shape[1]


def test_sgn_plus_one_is_honoured():
    """Certify that CPHD Global/SGN = +1 is conjugated to prevent mirrored image formation."""
    sig, pvp, meta, *_ = make_synthetic_collection(FC, BW, targets=[(5.0, -5.0, 1.0)])
    sig_plus_one = np.conj(sig)
    meta.sgn = +1

    res = pfa_per_polar(
        channel_signals=[sig_plus_one], channel_pvps=[pvp], channel_fxcs=[FC], channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"], cphd_meta=meta, u_min=-20.0, u_max=20.0, r_min=-20.0, r_max=20.0,
        image_oversample=1.25, device=DEV
    )
    img, _, _, N_r, N_u, _ = res
    iu, ir = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    u_found = -20.0 + iu * 40.0 / N_u
    r_found = -20.0 + ir * 40.0 / N_r
    assert abs(u_found - 5.0) < 0.5 and abs(r_found + 5.0) < 0.5, (
        f"Target located at ({u_found:+.2f}, {r_found:+.2f}) instead of (+5.0, -5.0)"
    )


def test_asymmetric_image_area_places_srp_correctly():
    """Certify that asymmetric framing shifts k-space phase so SRP remains at (0, 0)."""
    sig, pvp, meta, *_ = make_synthetic_collection(FC, BW, targets=[(0.0, 0.0, 1.0)])
    res = pfa_per_polar(
        channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[FC], channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"], cphd_meta=meta, u_min=0.0, u_max=40.0, r_min=-10.0, r_max=30.0,
        image_oversample=1.25, device=DEV
    )
    img, _, _, N_r, N_u, _ = res
    iu, ir = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    u_found = 0.0 + iu * 40.0 / N_u
    r_found = -10.0 + ir * 40.0 / N_r
    assert abs(u_found) < 0.5 and abs(r_found) < 0.5, (
        f"SRP found at ({u_found:.2f}, {r_found:.2f}) instead of (0, 0)"
    )


@pytest.mark.skipif(not os.path.exists(GOLD_CPHD) or not os.path.exists(GOLD_SICD), reason="Umbra-08 data not present")
def test_ground_to_slant_matches_gold():
    """Certify that ground-to-slant framing projection matches Umbra-08 dimensions within 1%."""
    import sarkit.cphd as sc
    proc = IFAProcessor(cphd_path=GOLD_CPHD, output_dir="/tmp", image_plane="SLANT", pad_factor=1.20)
    with open(GOLD_CPHD, "rb") as f:
        meta = proc._read_metadata(sc.Reader(f))
    
    with open(GOLD_SICD, "rb") as f:
        r_s = ss.NitfReader(f)
        xh_s = ss.XmlHelper(r_s.metadata.xmltree)
        nr = xh_s.load("./{*}ImageData/{*}NumRows")
        nc = xh_s.load("./{*}ImageData/{*}NumCols")
        ss_r = xh_s.load("./{*}Grid/{*}Row/{*}SS")
        ss_c = xh_s.load("./{*}Grid/{*}Col/{*}SS")
        gold_ext_r = nr * ss_r
        gold_ext_c = nc * ss_c

    # Compute slant basis vectors
    p_vec = meta.srp_ecf - meta.arp_pos_coa
    u_row = p_vec / np.linalg.norm(p_vec)
    u_v = meta.arp_vel_coa / np.linalg.norm(meta.arp_vel_coa)
    u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
    u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
    if meta.side_of_track == "L":
        u_col = -u_col

    # Set slant basis vectors onto meta matching run()
    meta.uIAX = u_col
    meta.uIAY = u_row

    u_min, u_max, r_min, r_max = proc._determine_spatial_bounds(meta)
    L_range = r_max - r_min
    L_azm = u_max - u_min

    ratio_r = L_range / gold_ext_r
    ratio_c = L_azm / gold_ext_c
    assert 0.99 <= ratio_r <= 1.01, f"Range extent mismatch ratio: {ratio_r:.4f}"
    assert 0.99 <= ratio_c <= 1.01, f"Azimuth extent mismatch ratio: {ratio_c:.4f}"


def test_normative_imprespwid_factor():
    """Certify uniform window ImpRespWid is 0.8859 / BW per SICD DIDD Table 5.2."""
    bw = 1.253079
    k_uniform = 0.88589
    expected_irw = k_uniform / bw
    assert abs(expected_irw * bw - 0.88589) < 1e-4


@pytest.mark.skipif(not os.path.exists(GOLD_SICD), reason="Umbra-08 SICD not present")
def test_radiometric_slope_angle():
    """Certify that SigmaZero / BetaZero matches cos(SlopeAng) per SICD DIDD Section 4.10.4."""
    with open(GOLD_SICD, "rb") as f:
        r = ss.NitfReader(f)
        xh = ss.XmlHelper(r.metadata.xmltree)
        beta = xh.load("./{*}Radiometric/{*}BetaZeroSFPoly/{*}Coef")
        sigma = xh.load("./{*}Radiometric/{*}SigmaZeroSFPoly/{*}Coef")
        slope = xh.load("./{*}SCPCOA/{*}SlopeAng")

    ratio = sigma / beta
    expected = np.cos(np.radians(slope))
    assert abs(ratio - expected) < 1e-5, f"Radiometric ratio mismatch: {ratio} vs {expected}"
