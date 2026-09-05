"""
Proposed regression tests for diffpfa, written by the audit (Claude Code, Fable 5.1).

Run from the repo root:
    /home/feildaw/mypyenv/bin/python -m pytest audit/claude_code_fable_5_1/proposed_tests -v

Tests marked EXPECTED-FAIL-TODAY document defects found by the audit; they should turn green once
the corresponding fix lands.  The others lock in behaviour the audit verified as correct.
None of them modify or depend on files under tests/.
"""
import os, sys, glob, subprocess
import numpy as np
import pytest
import torch

ROOT = os.getcwd()
AUD = os.path.join(ROOT, "audit", "claude_code_fable_5_1")
sys.path.insert(0, ROOT); sys.path.insert(0, AUD)

from a02_forward_model_vs_exact import make_collection, run_diffpfa, ideal_pfa_reference, fit_scalar
from a03_ipr_vs_position import irw_pslr
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT as C
from diffpfa.types import CPHDMetadata

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FC, BW, L = 9.6e9, 600e6, 40.0
SMALL_CPHD = "/home/feildaw/data/2023-09-11-10-37-05_UMBRA-05_CPHD.cphd"


# ----------------------------------------------------------------------------- verified-correct behaviour
@pytest.mark.parametrize("target", [(0.0, 0.0), (5.0, -5.0), (-12.5, 7.25), (18.0, 0.0), (0.0, 18.0)])
def test_image_matches_ideal_pfa_reference(target):
    """Point target: exact localisation, IPR width / PSLR equal to the ideal polar->Cartesian reference."""
    sig, pvp, meta, Ku, Kr, F, th = make_collection(FC, BW, targets=[(target[0], target[1], 1.0)])
    img, bw_r, bw_u, N_r, N_u, rot = run_diffpfa(sig, pvp, meta, FC, L)
    ref, _ = ideal_pfa_reference(Ku, Kr, [(target[0], target[1], 1.0)], N_u, N_r, L)
    du, dr = L / N_u, L / N_r
    iu, ir = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    # a target exactly half-way between pixel centres may legitimately peak on either neighbour
    assert abs((iu - N_u / 2) * du - target[0]) <= 0.5 * du + 1e-9 and abs((ir - N_r / 2) * dr - target[1]) <= 0.5 * dr + 1e-9
    g, res = fit_scalar(img.astype(np.complex128), ref)
    assert 20 * np.log10(res) < -45.0, f"residual vs ideal reference {20*np.log10(res):.1f} dB"
    wu, pu = irw_pslr(img[:, ir], du); wr, pr = irw_pslr(img[iu, :], dr)
    wu0, pu0 = irw_pslr(ref[:, ir], du); wr0, pr0 = irw_pslr(ref[iu, :], dr)
    assert abs(wu / wu0 - 1) < 0.02 and abs(wr / wr0 - 1) < 0.02
    assert abs(pu - pu0) < 0.5 and abs(pr - pr0) < 0.5
    assert abs(pu + 13.26) < 0.6, "uniform-aperture PSLR should be -13.26 dB"


def test_torch_chain_is_differentiable_and_adjoint_consistent():
    """Gradient w.r.t. the signal exists, passes gradcheck, and the linear operator's adjoint is exact."""
    from a04_gradient_check import torch_chain
    sig_np, pvp, meta, *_ = make_collection(FC, BW, ns=24, npulse=16, targets=[(1.0, -2.0, 1.0)])
    s = torch.tensor(sig_np, dtype=torch.complex128, device=DEV, requires_grad=True)
    torch.manual_seed(0)
    with torch.no_grad():
        w = torch.randn(torch_chain(s, pvp, meta, FC, 12.0, dtype=torch.complex128).shape, dtype=torch.complex128, device=DEV)
    f = lambda x: (torch_chain(x, pvp, meta, FC, 12.0, dtype=torch.complex128) * w).real.sum()
    assert torch.autograd.gradcheck(f, (s,), eps=1e-6, atol=1e-4, rtol=1e-3, fast_mode=True)
    a = torch.randn(sig_np.shape, dtype=torch.complex128, device=DEV); y = w
    x = a.clone().requires_grad_(True)
    (torch.conj(y) * torch_chain(x, pvp, meta, FC, 12.0, dtype=torch.complex128)).real.sum().backward()
    with torch.no_grad():
        lhs = (torch.conj(y) * torch_chain(a, pvp, meta, FC, 12.0, dtype=torch.complex128)).sum(); rhs = (torch.conj(x.grad) * a).sum()
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


# ----------------------------------------------------------------------------- EXPECTED-FAIL-TODAY
def _three_subbands(delta_tau, ns_list=(256, 256, 256)):
    fc, bw_total = 9.6e9, 600e6; bw_sub = bw_total / 3
    npulse, R = 512, 15000.0
    th = np.linspace(-np.radians(1.0), np.radians(1.0), npulse)
    pos = np.stack([R * np.sin(th), -R * np.cos(th), np.zeros_like(th)], 1)
    uIAX, uIAY, srp = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3)
    P = srp - pos; cos_t = P @ uIAX / np.linalg.norm(P, axis=1); sin_t = P @ uIAY / np.linalg.norm(P, axis=1)
    t0 = np.arange(npulse) / 1000.0
    chans = []
    for m, ns in enumerate(ns_list):
        f0 = (fc - bw_total / 2) + m * bw_sub; step = bw_sub / ns; F = f0 + np.arange(ns) * step
        s = np.exp(-1j * 2 * np.pi * (2 * F[None, :] / C) * (0.0 * cos_t + 0.0 * sin_t)[:, None]) + \
            np.exp(-1j * 2 * np.pi * (2 * F[None, :] / C) * (5.0 * cos_t - 8.0 * sin_t)[:, None])
        pvp = dict(SRPPos=np.tile(srp, (npulse, 1)), TxPos=pos, RcvPos=pos, TxVel=np.tile([7500.0, 0, 0], (npulse, 1)), RcvVel=np.tile([7500.0, 0, 0], (npulse, 1)),
                   SC0=np.full(npulse, f0), SCSS=np.full(npulse, step), RcvTime=t0 + m * delta_tau, TxTime=t0 + m * delta_tau)
        chans.append((s.astype(np.complex64), pvp, f0 + bw_sub / 2))
    meta = CPHDMetadata(domain_type="FX", sgn=-1, global_fx_min=fc - bw_total / 2, global_fx_max=fc + bw_total / 2, iarp_ecf=srp, uIAX=uIAX, uIAY=uIAY, ref_ch_id="0",
                        image_area=None, extended_area=None, collection_start=None, radar_mode="SPOTLIGHT", classification="U", srp_ecf=srp,
                        arp_pos_coa=pos[npulse // 2], arp_vel_coa=np.array([7500.0, 0, 0]), side_of_track="R", line_spacing=None, sample_spacing=None, raw_meta=None)
    return chans, meta, t0


def _form(chans, meta, t0):
    return pfa_per_polar(channel_signals=[c[0] for c in chans], channel_pvps=[c[1] for c in chans], channel_fxcs=[c[2] for c in chans],
                         channel_domain_types=["FX"] * len(chans), ref_rcv_time=t0, cphd_meta=meta, u_min=-20, u_max=20, r_min=-20, r_max=20,
                         image_oversample=1.25, device=DEV)


def test_subband_coherence_independent_of_burst_timing():
    """EXPECTED-FAIL-TODAY: CPHD sub-bands are SRP-referenced; image must not depend on inter-burst delay."""
    img_a, *_ = _form(*_three_subbands(150e-6))
    img_b, *_ = _form(*_three_subbands(137.3217e-6))
    coh = abs(np.vdot(img_a, img_b)) / (np.linalg.norm(img_a) * np.linalg.norm(img_b))
    assert coh > 0.999, f"complex coherence between delta_tau=150us and 137.3217us images is {coh:.4f}"


def test_subbands_equal_weight_regardless_of_sample_spacing():
    """EXPECTED-FAIL-TODAY: k-space amplitude must not depend on a channel's SCSS (resampler gain 1/(L*dk))."""
    img, bw_r, bw_u, N_r, N_u, _ = _form(*_three_subbands(150e-6, ns_list=(256, 512, 256)))
    cut = img[N_u // 2, :]; K = np.abs(np.fft.fftshift(np.fft.fft(cut)))
    occ = np.where(K > 0.1 * K.max())[0]; thirds = np.array_split(occ, 3)
    means = [K[t].mean() for t in thirds]
    assert abs(means[1] / means[0] - 1) < 0.05, f"middle sub-band weight {means[1]/means[0]:.3f} x low band"


def test_rotated_axes_return_true_range_and_azimuth_parameters():
    """EXPECTED-FAIL-TODAY: with uIAX along the LOS, pfa_per_polar must still return (bw_range, N_range) for the range axis."""
    sig, pvp, meta, *_ = make_collection(FC, BW, targets=[(5.0, -5.0, 1.0)])
    img0, bw_r0, bw_u0, N_r0, N_u0, rot0 = run_diffpfa(sig, pvp, meta, FC, L)
    assert not rot0
    meta.uIAX, meta.uIAY = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    img, bw_range, bw_azm, N_range, N_azm, rot = pfa_per_polar(channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[FC], channel_domain_types=["FX"],
                                                               ref_rcv_time=pvp["RcvTime"], cphd_meta=meta, u_min=-L / 2, u_max=L / 2, r_min=-L / 2, r_max=L / 2,
                                                               image_oversample=1.25, device=DEV)
    assert rot
    # after IFP's transpose, rows run along the range (uIAX here) axis and have img.shape[1] samples
    assert abs(bw_range / bw_r0 - 1) < 1e-6 and N_range == img.shape[1], f"bw_range={bw_range} (true {bw_r0}), N_range={N_range} (rows {img.shape[1]})"


def test_sgn_plus_one_is_honoured():
    """EXPECTED-FAIL-TODAY: CPHD Global/SGN = +1 data must not form a mirrored image."""
    sig, pvp, meta, *_ = make_collection(FC, BW, targets=[(5.0, -5.0, 1.0)])
    sig = np.conj(sig)              # same scene written with SGN = +1
    meta.sgn = +1
    img, bw_r, bw_u, N_r, N_u, _ = run_diffpfa(sig, pvp, meta, FC, L)
    iu, ir = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    u_found, r_found = (iu - N_u / 2) * L / N_u, (ir - N_r / 2) * L / N_r
    assert abs(u_found - 5.0) < 0.5 and abs(r_found + 5.0) < 0.5, f"target found at ({u_found:+.2f},{r_found:+.2f}) instead of (+5,-5)"


def test_asymmetric_image_area_places_srp_correctly():
    """EXPECTED-FAIL-TODAY: with u in [0,40] and r in [-10,30], the SRP (0,0) must land at (u=0, r=0), not at the area centre."""
    sig, pvp, meta, *_ = make_collection(FC, BW, targets=[(0.0, 0.0, 1.0)])
    img, _, _, N_r, N_u, _ = pfa_per_polar(channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[FC], channel_domain_types=["FX"], ref_rcv_time=pvp["RcvTime"],
                                          cphd_meta=meta, u_min=0.0, u_max=40.0, r_min=-10.0, r_max=30.0, image_oversample=1.25, device=DEV)
    iu, ir = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    u_found, r_found = 0.0 + iu * 40.0 / N_u, -10.0 + ir * 40.0 / N_r
    assert abs(u_found) < 0.5 and abs(r_found) < 0.5, f"SRP found at ({u_found:.2f},{r_found:.2f}) m in image-area coordinates"


@pytest.mark.skipif(not os.path.exists(SMALL_CPHD), reason="Umbra CPHD not present")
def test_product_passes_sarkit_consistency_and_declares_uniform_irw(tmp_path):
    """EXPECTED-FAIL-TODAY: SICD XML must pass sarkit SicdConsistency (no [Error]) and ImpRespWid*ImpRespBW ~ 0.886."""
    from diffpfa.IFP import IFAProcessor
    import sarkit.sicd as ss
    proc = IFAProcessor(cphd_path=SMALL_CPHD, output_dir=str(tmp_path), image_plane="SLANT", device=DEV)
    with torch.inference_mode():
        files, *_ = proc.run()
    with open(files[0], "rb") as f:
        h = ss.XmlHelper(ss.NitfReader(f).metadata.xmltree)
    for d in ("Row", "Col"):
        k = h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespWid") * h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespBW")
        assert abs(k - 0.886) < 0.02, f"{d}: ImpRespWid*ImpRespBW = {k:.4f}, expected 0.886 for uniform weighting"
    chk = subprocess.run([os.path.join(os.path.dirname(sys.executable), "sicdcheck"), files[0]], capture_output=True, text=True)
    errs = [l.strip() for l in chk.stdout.splitlines() if "[Error]" in l]
    assert not errs, "sicdcheck errors:\n" + "\n".join(errs)
