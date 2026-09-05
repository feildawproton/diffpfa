"""
a03_ipr_vs_position.py -- Impulse response quality as a function of image position, and the
'rotated dataset' branch, measured properly (16x FFT upsampling before any width is read).

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a03_ipr_vs_position.py

Why: a02 showed exact target localisation but a 2x wider azimuth IPR for a target at 62 % of the
half-extent.  The cross-range axis is formed by Kaiser-Bessel gridding onto a grid with NO
oversampling (nufft_grid_1d: oversample=1.0), followed by division by the kernel's continuous
transform.  Without grid oversampling the apodisation aliases back into the image and the 1/W
deconvolution amplifies it toward the image edges.  Range is formed by an exact CZT and should be
position-independent.  This script measures both, one isolated unit target at a time, against the
analytic uniform-aperture response (half-power width 0.8859/BW, PSLR -13.26 dB), and against the
ideal reference of a02 locally (residual in a window around the target).

Control: the same measurement is run on the ideal reference image itself; if the instrument
reports 0.886/BW and -13.3 dB there, the instrument is trustworthy.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a02_forward_model_vs_exact import make_collection, run_diffpfa, ideal_pfa_reference, fit_scalar
from diffpfa.IFA.PFA import pfa_per_polar

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def upsample_cut(cut, factor=16):
    n = len(cut)
    F = np.fft.fftshift(np.fft.fft(cut))
    Fp = np.zeros(n * factor, complex)
    Fp[(n * factor - n) // 2:(n * factor + n) // 2] = F
    return np.fft.ifft(np.fft.ifftshift(Fp)) * factor


def irw_pslr(cut, dx, factor=16):
    up = np.abs(upsample_cut(cut, factor)); dxu = dx / factor
    i = int(np.argmax(up)); hp = up[i] / np.sqrt(2)
    l = i
    while l > 0 and up[l] > hp: l -= 1
    r = i
    while r < len(up) - 1 and up[r] > hp: r += 1
    width = (r - l) * dxu
    # first nulls
    l2 = i
    while l2 > 0 and up[l2 - 1] < up[l2]: l2 -= 1
    r2 = i
    while r2 < len(up) - 1 and up[r2 + 1] < up[r2]: r2 += 1
    side = max(up[:l2].max() if l2 > 0 else 0, up[r2 + 1:].max() if r2 + 1 < len(up) else 0)
    return width, 20 * np.log10(side / up[i])


def measure(img, u_t, r_t, du, dr, N_u, N_r):
    iu = int(round(u_t / du + N_u / 2)); ir = int(round(r_t / dr + N_r / 2))
    iu = int(np.argmax(np.abs(img[max(0, iu - 3):iu + 4, ir]))) + max(0, iu - 3)
    ir = int(np.argmax(np.abs(img[iu, max(0, ir - 3):ir + 4]))) + max(0, ir - 3)
    wu, pu = irw_pslr(img[:, ir], du); wr, pr = irw_pslr(img[iu, :], dr)
    return dict(iu=iu, ir=ir, irw_u=wu, pslr_u=pu, irw_r=wr, pslr_r=pr, peak=float(np.abs(img[iu, ir])))


def main():
    fc, bw, L = 9.6e9, 600e6, 40.0
    out = {}
    print("[1] single unit target swept along u (cross-range, NUFFT axis) and along r (range, CZT axis)")
    print(f"    {'pos (u,r) m':>16} | {'IRW_u':>7} {'ctrl':>7} | {'PSLR_u dB':>9} {'ctrl':>7} | {'IRW_r':>7} {'ctrl':>7} | {'PSLR_r dB':>9} {'ctrl':>7} | {'peak/peak0':>10} | {'local resid dB':>14}")
    rows = []
    peak0 = None
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9, 0.97):
        for axis in ("u", "r"):
            if frac == 0.0 and axis == "r":
                continue
            u_t, r_t = (frac * L / 2, 0.0) if axis == "u" else (0.0, frac * L / 2)
            sig, pvp, meta, Ku, Kr, F, th = make_collection(fc, bw, targets=[(u_t, r_t, 1.0)])
            img, bw_r, bw_u, N_r, N_u, _ = run_diffpfa(sig, pvp, meta, fc, L)
            du, dr = L / N_u, L / N_r
            ref, _ = ideal_pfa_reference(Ku, Kr, [(u_t, r_t, 1.0)], N_u, N_r, L)
            m = measure(img, u_t, r_t, du, dr, N_u, N_r)
            c = measure(ref, u_t, r_t, du, dr, N_u, N_r)
            if peak0 is None:
                peak0 = m["peak"]
            # local residual vs reference (scalar fitted on the whole image once, from the centre case)
            g, _ = fit_scalar(img.astype(np.complex128), ref)
            win = 10
            sl = (slice(max(0, m["iu"] - win), m["iu"] + win), slice(max(0, m["ir"] - win), m["ir"] + win))
            res = img[sl] - g * ref[sl]
            loc = 20 * np.log10(np.sqrt(np.mean(np.abs(res) ** 2)) / np.abs(g * ref[sl]).max())
            print(f"    ({u_t:6.2f},{r_t:6.2f}) | {m['irw_u']:7.4f} {c['irw_u']:7.4f} | {m['pslr_u']:9.2f} {c['pslr_u']:7.2f} | {m['irw_r']:7.4f} {c['irw_r']:7.4f} | {m['pslr_r']:9.2f} {c['pslr_r']:7.2f} | {m['peak']/peak0:10.4f} | {loc:14.2f}")
            rows.append(dict(u=u_t, r=r_t, **{k: float(v) for k, v in m.items()}, ctrl={k: float(v) for k, v in c.items()}, local_resid_db=float(loc), gain_abs=float(abs(g))))
    print(f"    analytic uniform aperture: IRW_u = 0.8859/{bw_u:.4f} = {0.8859/bw_u:.4f} m, IRW_r = 0.8859/{bw_r:.4f} = {0.8859/bw_r:.4f} m, PSLR = -13.26 dB")
    out["sweep"] = rows

    # ------------------------------------------------------------ 2. NUFFT deconvolution magnitude vs position
    import math
    beta, J = 13.9086, 6
    xi = (np.arange(N_u) - N_u / 2) / N_u
    z2 = beta ** 2 - (math.pi * J * xi) ** 2
    W = np.where(z2 >= 0, np.sinh(np.sqrt(np.abs(z2))) / np.sqrt(np.abs(z2)), np.sin(np.sqrt(np.abs(z2))) / np.sqrt(np.abs(z2)))
    W = W / (np.sinh(beta) / beta)
    print(f"[2] KB deconvolution factor 1/W(xi) applied along u: at centre {1/W[N_u//2]:.3f}, at 75 % of half-extent {1/W[int(N_u*0.875)]:.3f}, at edge {1/W[0]:.3f}")
    out["deconv_edge_gain"] = float(1 / W[0])

    # ------------------------------------------------------------ 3. rotated-axes branch, done right this time
    # Keep the physical geometry fixed (radar at -y, LOS along +y, moving along +x); relabel the CPHD axes so
    # uIAX (the axis diffpfa treats as cross-range) is the LOS direction.  Target at physical (x=+5, y=-5).
    print("[3] rotated-dataset branch: uIAX := +y (LOS), uIAY := +x (along-track); target at physical x=+5, y=-5")
    sig, pvp, meta, Ku, Kr, F, th = make_collection(fc, bw, targets=[(5.0, -5.0, 1.0)])
    meta.uIAX, meta.uIAY = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    img_r, bw_range, bw_azm, N_range, N_azm, rot = pfa_per_polar(
        channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[fc], channel_domain_types=["FX"], ref_rcv_time=pvp["RcvTime"],
        cphd_meta=meta, u_min=-L / 2, u_max=L / 2, r_min=-L / 2, r_max=L / 2, image_oversample=1.25, device=DEV)
    iu, ir = np.unravel_index(np.argmax(np.abs(img_r)), img_r.shape)
    print(f"    branch taken: {rot}; img shape {img_r.shape}; returned bw_range={bw_range:.4f} bw_azm={bw_azm:.4f} N_range={N_range} N_azm={N_azm}")
    print(f"    non-rotated run for same physics: bw_r(range)={bw_r:.4f} bw_u(azimuth)={bw_u:.4f} N_r={N_r} N_u={N_u}")
    print(f"    peak at img[{iu},{ir}]: dim0 coord {(iu-img_r.shape[0]/2)*L/img_r.shape[0]:+.3f} m, dim1 coord {(ir-img_r.shape[1]/2)*L/img_r.shape[1]:+.3f} m")
    print(f"    after IFP's img.T, rows are dim1 (len {img_r.shape[1]}) and cols dim0 (len {img_r.shape[0]}). IFP then computes")
    print(f"      dr_range = L/N_range = {L/N_range:.4f} (true row spacing is L/{img_r.shape[1]} = {L/img_r.shape[1]:.4f}),  du_azm = L/N_azm = {L/N_azm:.4f} (true col spacing L/{img_r.shape[0]} = {L/img_r.shape[0]:.4f})")
    print(f"      Row/ImpRespBW <- bw_range = {bw_range:.4f} (true range bandwidth {bw_r:.4f}),  Col/ImpRespBW <- bw_azm = {bw_azm:.4f} (true azimuth bandwidth {bw_u:.4f})")
    out["rotated"] = dict(rot=bool(rot), shape=list(img_r.shape), bw_range=float(bw_range), bw_azm=float(bw_azm), N_range=int(N_range), N_azm=int(N_azm),
                          true_bw_r=float(bw_r), true_bw_u=float(bw_u), peak=[int(iu), int(ir)])
    with open(os.path.join(HERE, "out", "a03_results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
