"""
a16_remediation_gaps.py -- Targeted checks of two things the remediation (commit 405b657) could
have got wrong that no local data or existing test exercises.

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a16_remediation_gaps.py

(1) Multi-channel metadata.  `_write_sicd` recomputes KCtr / Krg / Kaz / the PFA polynomials from
    `ref_pvp = channel_pvps[0]` and `num_samples = channel_signals[0].shape[1]`, i.e. from the FIRST
    channel of the polarisation group only, while the image was formed on the k-space grid spanning
    ALL channels (gku_ctr, gkr_ctr over the group).  For a stepped-chirp group the written Row/KCtr
    would be the centre of sub-band 0, not of the combined band.  Demonstrated with three synthetic
    200 MHz sub-bands: form the image with pfa_per_polar, write the SICD with IFAProcessor._write_sicd,
    read back KCtr / DeltaK / Krg and compare with the grid actually used; run sicdcheck on the XML.
(2) SCPPixel under non-symmetric framing.  pfa_per_polar now centres the image on the framing-box
    centre (u_c, r_c) (the F7 fix), but `_write_sicd` still writes SCPPixel = (N//2, N//2) and
    projects the ICPs about that pixel.  Form an image with u in [0,40], r in [-10,30]; the SRP
    scatterer is at pixel (0, N_r/4); compare with the SCPPixel written.
"""
import os, sys, io, contextlib, subprocess, tempfile
import numpy as np
import torch
import lxml.etree as ET

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sarkit.sicd as ss
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.IFP import IFAProcessor
from diffpfa.constants import SPEED_OF_LIGHT as C
from diffpfa.types import CPHDMetadata, ImageAreaBounds
from a02_forward_model_vs_exact import make_collection

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SICDCHECK = os.path.join(os.path.dirname(sys.executable), "sicdcheck")


def realistic_geometry(npulse, t):
    """Spaceborne-like arc so that SCPCOA/ICP computations have a sane Earth geometry."""
    # SRP on the WGS-84 surface at ~47N, -97E; ARP 570 km away at 45 deg grazing moving at 7.5 km/s
    import sarkit.wgs84 as w
    srp = w.geodetic_to_cartesian([46.9, -96.8, 250.0])
    up = w.up([46.9, -96.8, 250.0]); east = np.array([-np.sin(np.radians(-96.8)), np.cos(np.radians(-96.8)), 0.0]); north = np.cross(up, east)
    R = 567000.0
    arp0 = srp + R * (np.sin(np.radians(45)) * up + np.cos(np.radians(45)) * (-north))          # radar to the south, 45 deg graze
    vel = 7500.0 * east                                                                     # flying east (broadside-ish)
    pos = arp0[None, :] + (t - t.mean())[:, None] * vel[None, :]
    return srp, pos, np.tile(vel, (npulse, 1))


def make_group(nsub=3, ns=256, npulse=512, bw_total=600e6, fc=9.6e9):
    t = np.arange(npulse) / 1000.0 + 0.5
    srp, pos, vel = realistic_geometry(npulse, t)
    P = srp[None, :] - pos; Pn = np.linalg.norm(P, axis=1)
    u_row = P[npulse // 2] / Pn[npulse // 2]; u_v = vel[0] / np.linalg.norm(vel[0])
    u_col = u_v - np.dot(u_v, u_row) * u_row; u_col /= np.linalg.norm(u_col); u_col = -u_col   # 'L' convention sign irrelevant here
    cos_t = P @ u_col / Pn; sin_t = P @ u_row / Pn
    bw_sub = bw_total / nsub
    chans = []
    for m in range(nsub):
        f0 = fc - bw_total / 2 + m * bw_sub; step = bw_sub / ns; F = f0 + np.arange(ns) * step
        s = np.exp(-1j * 2 * np.pi * (2 * F[None, :] / C) * (3.0 * cos_t - 4.0 * sin_t)[:, None]) + 0.7 * np.exp(-1j * 0 * F[None, :] * cos_t[:, None])
        pvp = dict(SRPPos=np.tile(srp, (npulse, 1)), TxPos=pos, RcvPos=pos, TxVel=vel, RcvVel=vel, SC0=np.full(npulse, f0), SCSS=np.full(npulse, step),
                   TxTime=t + m * 137.3217e-6, RcvTime=t + m * 137.3217e-6 + 2 * Pn / C, SIGNAL=np.ones(npulse, int))
        chans.append((s.astype(np.complex64), pvp, f0 + bw_sub / 2))
    meta = CPHDMetadata(domain_type="FX", sgn=-1, global_fx_min=fc - bw_total / 2, global_fx_max=fc + bw_total / 2, iarp_ecf=srp, uIAX=u_col, uIAY=u_row, ref_ch_id="0",
                        image_area=ImageAreaBounds(-20, -20, 20, 20, None), extended_area=None, collection_start="2026-01-01T00:00:00Z", radar_mode="SPOTLIGHT",
                        classification="UNCLASSIFIED", srp_ecf=srp, arp_pos_coa=pos[npulse // 2], arp_vel_coa=vel[0], side_of_track="L", line_spacing=None,
                        sample_spacing=None, raw_meta=None, ref_uIAX=u_col, ref_uIAY=u_row)
    return chans, meta


def write_and_read(img, meta, chans, u_min, u_max, r_min, r_max, bw_range, bw_azm, N_range, N_azm, rot, tag):
    proc = IFAProcessor(cphd_path="synthetic_CPHD.cphd", output_dir=tempfile.mkdtemp(), image_plane="SLANT", device=DEV)
    out = os.path.join(proc.output_dir, f"{tag}.nitf")
    with contextlib.redirect_stdout(io.StringIO()):
        proc._write_sicd(out, img.T, meta, "V", "V", bw_range, bw_azm, N_range, N_azm, u_min, r_min, (u_max - u_min) / N_azm, (r_max - r_min) / N_range,
                         ref_pvp=chans[0][1], num_samples=chans[0][0].shape[1], is_rotated=rot)
    with open(out, "rb") as f:
        h = ss.XmlHelper(ss.NitfReader(f).metadata.xmltree)
    chk = subprocess.run([SICDCHECK, out], capture_output=True, text=True)
    errs = [l.strip() for l in chk.stdout.splitlines() if "[Error]" in l or "[Warning]" in l]
    return h, errs


def main():
    print("(1) stepped-chirp group of 3 x 200 MHz: XML k-space metadata vs the grid actually used")
    chans, meta = make_group()
    with contextlib.redirect_stdout(io.StringIO()):
        img, bw_r, bw_u, N_r, N_u, rot = pfa_per_polar(channel_signals=[c[0] for c in chans], channel_pvps=[c[1] for c in chans], channel_fxcs=[c[2] for c in chans],
                                                      channel_domain_types=["FX"] * 3, ref_rcv_time=chans[0][1]["RcvTime"], cphd_meta=meta,
                                                      u_min=-20, u_max=20, r_min=-20, r_max=20, image_oversample=1.25, device=DEV)
    # the grid pfa_per_polar used: centre of min/max over ALL channels
    from diffpfa.IFA.kspace import compute_kspace
    kr_all = []; ku_all = []
    for s, pvp, _ in chans:
        Ku, Kr = compute_kspace(pvp, meta.uIAX, meta.uIAY, s.shape[1], "FX", device="cpu"); kr_all += [Kr.min().item(), Kr.max().item()]; ku_all += [Ku.min().item(), Ku.max().item()]
    true_kctr_r = 0.5 * (min(kr_all) + max(kr_all)); true_kctr_u = 0.5 * (min(ku_all) + max(ku_all))
    h, errs = write_and_read(img, meta, chans, -20, 20, -20, 20, bw_r, bw_u, N_r, N_u, rot, "stepped")
    print(f"    combined grid: Row KCtr used = {true_kctr_r:.4f}, bandwidth = {bw_r:.4f}; XML writes Row/KCtr = {h.load('./{*}Grid/{*}Row/{*}KCtr'):.4f}, ImpRespBW = {h.load('./{*}Grid/{*}Row/{*}ImpRespBW'):.4f}, "
          f"Krg1/Krg2 = {h.load('./{*}PFA/{*}Krg1'):.4f}/{h.load('./{*}PFA/{*}Krg2'):.4f} (true {min(kr_all):.4f}/{max(kr_all):.4f})")
    print(f"    Col KCtr used = {true_kctr_u:.5f}; XML Col/KCtr = {h.load('./{*}Grid/{*}Col/{*}KCtr'):.5f}")
    print(f"    sicdcheck on the stepped-chirp product: {len(errs)} error/warning lines" + ("".join("\n       " + e for e in errs)))
    # control: single channel covering the full band
    ns = 768; f0 = meta.global_fx_min; step = 600e6 / ns
    s0, pvp0, _ = chans[0]
    pvp_full = dict(pvp0); pvp_full["SC0"] = np.full(len(pvp0["SC0"]), f0); pvp_full["SCSS"] = np.full(len(pvp0["SC0"]), step)
    F = f0 + np.arange(ns) * step
    srp = meta.srp_ecf; P = srp[None, :] - pvp0["TxPos"]; Pn = np.linalg.norm(P, axis=1); cos_t = P @ meta.uIAX / Pn; sin_t = P @ meta.uIAY / Pn
    s_full = (np.exp(-1j * 2 * np.pi * (2 * F[None, :] / C) * (3.0 * cos_t - 4.0 * sin_t)[:, None]) + 0.7).astype(np.complex64)
    with contextlib.redirect_stdout(io.StringIO()):
        img1, bw_r1, bw_u1, N_r1, N_u1, rot1 = pfa_per_polar(channel_signals=[s_full], channel_pvps=[pvp_full], channel_fxcs=[9.6e9], channel_domain_types=["FX"],
                                                            ref_rcv_time=pvp_full["RcvTime"], cphd_meta=meta, u_min=-20, u_max=20, r_min=-20, r_max=20, image_oversample=1.25, device=DEV)
    h1, errs1 = write_and_read(img1, meta, [(s_full, pvp_full, 9.6e9)], -20, 20, -20, 20, bw_r1, bw_u1, N_r1, N_u1, rot1, "mono")
    print(f"    control (single full-band channel): XML Row/KCtr = {h1.load('./{*}Grid/{*}Row/{*}KCtr'):.4f}; sicdcheck {len(errs1)} error/warning lines" + ("".join("\n       " + e for e in errs1)))

    print("\n(2) non-symmetric framing: where is the SRP, and where does the XML say it is?")
    sig, pvp, meta2, *_ = make_collection(9.6e9, 600e6, targets=[(0.0, 0.0, 1.0)])
    with contextlib.redirect_stdout(io.StringIO()):
        img2, bw_r2, bw_u2, N_r2, N_u2, rot2 = pfa_per_polar(channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[9.6e9], channel_domain_types=["FX"], ref_rcv_time=pvp["RcvTime"],
                                                            cphd_meta=meta2, u_min=0.0, u_max=40.0, r_min=-10.0, r_max=30.0, image_oversample=1.25, device=DEV)
    iu, ir = np.unravel_index(np.argmax(np.abs(img2)), img2.shape)
    print(f"    image {img2.shape} (u,r); SRP scatterer found at (u_idx={iu}, r_idx={ir}) -> u = {iu*40/N_u2:.2f} m, r = {-10+ir*40/N_r2:.2f} m  (correct placement)")
    print(f"    after IFP transpose rows=r, cols=u: SRP is at pixel (row={ir}, col={iu}); _write_sicd writes SCPPixel = (row={N_r2//2}, col={N_u2//2})"
          f" -> SCP mis-registered by ({(N_r2//2-ir)*40/N_r2:+.1f} m range, {(N_u2//2-iu)*40/N_u2:+.1f} m azimuth); ICPs are projected about the same wrong pixel.")
    print("    note: with the new SLANT framing this only bites when IARP != SRP or the ImageArea is not centred on the IARP (true for none of the six local CPHDs).")


if __name__ == "__main__":
    main()
