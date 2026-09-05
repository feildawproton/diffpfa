"""
a11_precision_and_multichannel_gain.py -- Two latent numerical properties.

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a11_precision_and_multichannel_gain.py

(A) Working precision of the CZT.  czt_1d_torch picks float32 when the signal is complex64 (which it
    always is: PFA.py casts to complex64).  Its pre-/post-chirp phases are  2*pi*r_min*k  with
    r_min = -L/2 ~ -2500 m and k ~ 64 cyc/m, i.e. ~1e6 rad.  float32 resolves 1e6 to 0.06 rad, so a
    float32 CZT would inject ~2 deg of random phase per sample.  In the shipped pipeline k_start/k_step
    arrive as float64 tensors and torch's type promotion silently lifts the whole CZT to complex128.
    We (1) record the dtypes that actually reach torch.fft.fft in a realistic-scale run, and (2) force
    the float32 path (cast k_start/k_step to float32, as an innocent-looking .to(real_dtype) would) and
    measure the image error against the double-precision run.
(B) Gain of the CZT resampler vs fast-time sample spacing.  czt_resample_kspace_1d returns
    x(k')/(L*dk_in) rather than x(k') (a02 [2] showed peak amplitude proportional to N_samples).
    Harmless for one channel; for stepped-chirp channels with different SCSS the sub-bands get
    different weights in the shared k-space.  Demonstrated with the shipped simulation geometry:
    3 sub-bands of equal bandwidth, the middle one sampled twice as densely.
"""
import os, sys, json, math
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffpfa.IFA.channel.czt_torch as czt
import diffpfa.IFA.channel.pfa_channel as pc
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT as C
from diffpfa.types import CPHDMetadata, ImageAreaBounds
from a02_forward_model_vs_exact import make_collection, run_diffpfa
from a03_ipr_vs_position import irw_pslr

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def part_a():
    print("[A] CZT working precision at realistic scale (L = 5000 m, k ~ 64 cyc/m)")
    fc, bw = 9.6e9, 60e6                       # narrow band keeps N_r small; phases are set by r_min*k, not by bw
    L = 5000.0
    sig, pvp, meta, *_ = make_collection(fc, bw, ns=256, npulse=2048, span_deg=0.2, R=570e3, targets=[(0.0, 0.0, 1.0), (1200.0, -1700.0, 1.0)])
    seen = []
    orig_fft = torch.fft.fft
    def spy(x, *a, **k):
        seen.append(str(x.dtype)); return orig_fft(x, *a, **k)
    torch.fft.fft = spy
    try:
        img64, bw_r, bw_u, N_r, N_u, _ = run_diffpfa(sig, pvp, meta, fc, L)
    finally:
        torch.fft.fft = orig_fft
    from collections import Counter
    print(f"    dtypes reaching torch.fft.fft inside the CZT during the shipped pipeline: {dict(Counter(seen))}")
    print(f"    grid N_u={N_u} N_r={N_r}; du={L/N_u:.3f} dr={L/N_r:.3f}")
    # force float32 k parameters
    orig = pc.czt_resample_kspace_1d
    def f32(signal, k_start, k_step, *a, **k):
        return orig(signal, k_start.to(torch.float32), k_step.to(torch.float32), *a, **k)
    pc.czt_resample_kspace_1d = f32
    try:
        img32, *_ = run_diffpfa(sig, pvp, meta, fc, L)
    finally:
        pc.czt_resample_kspace_1d = orig
    d = img32.astype(np.complex128) - img64.astype(np.complex128)
    pk = np.abs(img64).max()
    print(f"    forced-float32 CZT vs shipped (promoted) run: max |diff| = {20*np.log10(np.abs(d).max()/pk):.1f} dB rel. peak, RMS = {20*np.log10(np.sqrt(np.mean(np.abs(d)**2))/pk):.1f} dB")
    for (u_t, r_t) in [(0.0, 0.0), (1200.0, -1700.0)]:
        iu = int(round(u_t / (L / N_u) + N_u / 2)); ir = int(round(r_t / (L / N_r) + N_r / 2))
        for name, im in (("float64 path", img64), ("float32 path", img32)):
            sub = np.abs(im[iu - 4:iu + 5, ir - 4:ir + 5]); p = np.unravel_index(np.argmax(sub), sub.shape)
            iu2, ir2 = iu - 4 + p[0], ir - 4 + p[1]
            wr, pr = irw_pslr(im[iu2, :], L / N_r); wu, pu = irw_pslr(im[:, ir2], L / N_u)
            print(f"      target ({u_t:+.0f},{r_t:+.0f}) {name}: peak {np.abs(im[iu2, ir2])/pk:.4f}  IRW_r={wr:.3f} m PSLR_r={pr:.1f} dB  IRW_u={wu:.3f} m PSLR_u={pu:.1f} dB")
    return dict(dtypes=dict(Counter(seen)), max_diff_db=float(20 * np.log10(np.abs(d).max() / pk)))


def part_b():
    print("[B] sub-band weighting vs fast-time sample spacing (3 x 200 MHz, middle band sampled 2x denser)")
    fc, bw_total, nsub = 9.6e9, 600e6, 3
    bw_sub = bw_total / nsub
    npulse, span_deg, R = 512, 2.0, 15000.0
    th = np.linspace(-np.radians(span_deg / 2), np.radians(span_deg / 2), npulse)
    pos = np.stack([R * np.sin(th), -R * np.cos(th), np.zeros_like(th)], 1)
    uIAX, uIAY = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]); srp = np.zeros(3)
    P = srp - pos; cos_t = P @ uIAX / np.linalg.norm(P, axis=1); sin_t = P @ uIAY / np.linalg.norm(P, axis=1)
    targets = [(0.0, 0.0, 1.0), (5.0, -8.0, 1.0)]
    def channel(m, ns):
        f0 = (fc - bw_total / 2) + m * bw_sub; step = bw_sub / ns; F = f0 + np.arange(ns) * step
        s = np.zeros((npulse, ns), np.complex128)
        for u_t, r_t, a in targets:
            s += a * np.exp(-1j * 2 * np.pi * (2 * F[None, :] / C) * (u_t * cos_t + r_t * sin_t)[:, None])
        pvp = dict(SRPPos=np.tile(srp, (npulse, 1)), TxPos=pos, RcvPos=pos, TxVel=np.tile([7500.0, 0, 0], (npulse, 1)), RcvVel=np.tile([7500.0, 0, 0], (npulse, 1)),
                   SC0=np.full(npulse, f0), SCSS=np.full(npulse, step), RcvTime=np.arange(npulse) / 1000.0, TxTime=np.arange(npulse) / 1000.0)
        return s.astype(np.complex64), pvp, f0 + bw_sub / 2
    meta = CPHDMetadata(domain_type="FX", sgn=-1, global_fx_min=fc - bw_total / 2, global_fx_max=fc + bw_total / 2, iarp_ecf=srp, uIAX=uIAX, uIAY=uIAY, ref_ch_id="0",
                        image_area=None, extended_area=None, collection_start=None, radar_mode="SPOTLIGHT", classification="U", srp_ecf=srp,
                        arp_pos_coa=pos[npulse // 2], arp_vel_coa=np.array([7500.0, 0, 0]), side_of_track="R", line_spacing=None, sample_spacing=None, raw_meta=None)
    out = {}
    for label, ns_list in (("equal sampling (256,256,256)", [256, 256, 256]), ("unequal sampling (256,512,256)", [256, 512, 256])):
        chans = [channel(m, ns) for m, ns in enumerate(ns_list)]
        img, bw_r, bw_u, N_r, N_u, _ = pfa_per_polar(channel_signals=[c[0] for c in chans], channel_pvps=[c[1] for c in chans], channel_fxcs=[c[2] for c in chans],
                                                    channel_domain_types=["FX"] * 3, ref_rcv_time=chans[0][1]["RcvTime"], cphd_meta=meta,
                                                    u_min=-20, u_max=20, r_min=-20, r_max=20, image_oversample=1.25, device=DEV)
        dr = 40 / N_r; du = 40 / N_u
        iu, ir = N_u // 2, N_r // 2
        wr, pr = irw_pslr(img[iu, :], dr)
        # k-space amplitude per sub-band: FFT the centre-target range cut back to k-space
        cut = img[iu, :]
        K = np.abs(np.fft.fftshift(np.fft.fft(cut)))
        k = np.fft.fftshift(np.fft.fftfreq(N_r, d=dr))
        occ = K > 0.1 * K.max()
        third = np.array_split(np.where(occ)[0], 3)
        band_means = [float(K[t].mean()) for t in third]
        print(f"    {label:32s}: range IRW={wr:.4f} m (0.886/BW={0.8859/bw_r:.4f})  PSLR_r={pr:.2f} dB  | k-space amplitude per sub-band (low,mid,high) = "
              + ", ".join(f"{b/band_means[0]:.3f}" for b in band_means) + " (relative)")
        out[label] = dict(irw_r=float(wr), pslr_r=float(pr), band_rel=[b / band_means[0] for b in band_means])
    return out


def main():
    res = {"A": part_a(), "B": part_b()}
    with open(os.path.join(HERE, "out", "a11_results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
