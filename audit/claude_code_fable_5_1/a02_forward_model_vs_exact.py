"""
a02_forward_model_vs_exact.py -- Does pfa_per_polar form the image the maths says it should?

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a02_forward_model_vs_exact.py

Set-up: a synthetic spotlight collection (X-band, 600 MHz, 2 deg polar span) with a few
point targets, generated with exactly the CPHD phase convention the library assumes,
    s[n,k] = sum_t a_t exp(-j 2 pi (2 F[n,k]/c) (u_t cos(theta_n) + r_t sin(theta_n))).

Reference A ("ideal PFA"): the scene's continuous k-space is known analytically,
    S(K) = sum_t a_t exp(-j 2 pi K . x_t)   for K inside the polar support, 0 outside.
Sample S on exactly the Cartesian grid diffpfa uses (same N_u, N_r, dK, k-centres), mask to
the polar support, inverse-FFT.  This is what a perfect polar->Cartesian interpolator would
produce, so the residual isolates interpolation / gridding / deconvolution error.

Reference B ("matched filter"): direct back-projection of the polar samples with the polar
Jacobian |K| dK dtheta, evaluated on the pixel grid.  This is the physical target response and
is independent of PFA's Cartesian resampling.

Measurements (all reported, none asserted):
  * complex fit of diffpfa image to reference A: scalar gain, residual RMS in dB below peak
  * peak location, half-power width along u and r, PSLR along u and r vs analytic sinc
  * phase at the peak (should be 0 up to the demodulation ramp)
  * amplitude-scaling dependence on image extent L and on fast-time sample spacing SCSS
  * the 'rotated dataset' branch: swap uIAX/uIAY and check where targets land
"""
import os, sys, json, math
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT as C
from diffpfa.types import CPHDMetadata, ImageAreaBounds
from scipy.fft import next_fast_len

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def make_collection(fc=9.6e9, bw=600e6, ns=256, npulse=256, span_deg=2.0, R=10000.0,
                    targets=((0.0, 0.0, 1.0),), scss_scale=1.0, uIAX=(1, 0, 0), uIAY=(0, 1, 0)):
    """Return (signal, pvp, meta pieces).  Radar sits at -y (range axis = +y), moves along +x."""
    uIAX = np.array(uIAX, float); uIAY = np.array(uIAY, float)
    f0 = fc - bw / 2
    scss = (bw / ns) * scss_scale
    ns_eff = int(round(ns / scss_scale))
    F = f0 + np.arange(ns_eff) * scss                               # (ns_eff,)
    th = np.linspace(-np.radians(span_deg / 2), np.radians(span_deg / 2), npulse)
    # physical positions (in the plane spanned by uIAX,uIAY)
    pos = np.stack([R * np.sin(th), -R * np.cos(th), np.zeros_like(th)], 1)   # local (x,y,z)
    # rotate local frame into (uIAX,uIAY, n)
    n_vec = np.cross(uIAX, uIAY)
    B = np.stack([uIAX, uIAY, n_vec], 1)                          # columns = basis
    pos_ecf = pos @ B.T
    srp = np.zeros(3)
    P = srp - pos_ecf                                              # look vectors
    cos_t = P @ uIAX / np.linalg.norm(P, axis=1)
    sin_t = P @ uIAY / np.linalg.norm(P, axis=1)
    sig = np.zeros((npulse, ns_eff), np.complex128)
    for (u_t, r_t, a) in targets:
        dR = u_t * cos_t + r_t * sin_t                              # (npulse,)
        sig += a * np.exp(-1j * 2 * np.pi * (2 * F[None, :] / C) * dR[:, None])
    pvp = {
        "SRPPos": np.tile(srp, (npulse, 1)), "TxPos": pos_ecf, "RcvPos": pos_ecf,
        "TxVel": np.tile(np.array([100.0, 0, 0]) @ B.T, (npulse, 1)),
        "RcvVel": np.tile(np.array([100.0, 0, 0]) @ B.T, (npulse, 1)),
        "SC0": np.full(npulse, f0), "SCSS": np.full(npulse, scss),
        "RcvTime": np.linspace(0, 1, npulse), "TxTime": np.linspace(0, 1, npulse),
    }
    meta = CPHDMetadata(domain_type="FX", sgn=-1, global_fx_min=f0, global_fx_max=f0 + ns_eff * scss,
                        iarp_ecf=srp, uIAX=uIAX, uIAY=uIAY, ref_ch_id="P", image_area=None,
                        extended_area=None, collection_start=None, radar_mode="SPOTLIGHT",
                        classification="U", srp_ecf=srp, arp_pos_coa=pos_ecf[npulse // 2],
                        arp_vel_coa=np.array([100.0, 0, 0]) @ B.T, side_of_track="R",
                        line_spacing=None, sample_spacing=None, raw_meta=None)
    Ku = (2 * F[None, :] / C) * cos_t[:, None]
    Kr = (2 * F[None, :] / C) * sin_t[:, None]
    return sig.astype(np.complex64), pvp, meta, Ku, Kr, F, th


def run_diffpfa(sig, pvp, meta, fc, L, oversample=1.25, custom=None):
    img, bw_r, bw_u, N_r, N_u, rot = pfa_per_polar(
        channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[fc], channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"], cphd_meta=meta, u_min=-L / 2, u_max=L / 2, r_min=-L / 2, r_max=L / 2,
        custom_pixel_spacing=custom, image_oversample=oversample, device=DEV)
    return img, bw_r, bw_u, N_r, N_u, rot


def ideal_pfa_reference(Ku, Kr, targets, N_u, N_r, L):
    """Reference A. Cartesian grid identical to diffpfa's: centre = (min+max)/2, step 1/L."""
    ku_c = (Ku.min() + Ku.max()) / 2; kr_c = (Kr.min() + Kr.max()) / 2
    dK = 1.0 / L
    ku = ku_c + (np.arange(N_u) - N_u / 2) * dK
    kr = kr_c + (np.arange(N_r) - N_r / 2) * dK
    KU, KR = np.meshgrid(ku, kr, indexing="ij")
    # polar support mask: |K| in [kmin,kmax], angle in [th_min, th_max]
    kmag_all = np.sqrt(Ku ** 2 + Kr ** 2)
    ang_all = np.arctan2(Ku, Kr)
    KM = np.sqrt(KU ** 2 + KR ** 2); AN = np.arctan2(KU, KR)
    mask = (KM >= kmag_all.min()) & (KM <= kmag_all.max()) & (AN >= ang_all.min()) & (AN <= ang_all.max())
    S = np.zeros_like(KU, dtype=np.complex128)
    for (u_t, r_t, a) in targets:
        S += a * np.exp(-1j * 2 * np.pi * (KU * u_t + KR * r_t))
    S *= mask
    # same ifft convention as _apply_ifft_and_deconv: ifftshift -> ifft2 -> *N -> fftshift
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(S))) * (N_u * N_r)
    return img, mask


def matched_filter_reference(sig, Ku, Kr, F, th, N_u, N_r, L):
    """Reference B: back-projection with polar Jacobian, evaluated on the pixel grid."""
    u = (np.arange(N_u) - N_u / 2) * (L / N_u); r = (np.arange(N_r) - N_r / 2) * (L / N_r)
    kmag = 2 * F / C
    dk = kmag[1] - kmag[0]; dth = th[1] - th[0]
    jac = (kmag[None, :] * dk * dth)                                 # |K| dK dtheta
    w = sig.astype(np.complex128) * jac
    # image(u,r) = sum_{n,k} w exp(+j2pi(Ku u + Kr r)); do it as two matmuls per pulse
    img = np.zeros((N_u, N_r), np.complex128)
    Eu = np.exp(1j * 2 * np.pi * np.einsum("nk,u->nku", Ku, u))     # (n,k,u) -- ok for 256x256x~110
    Er = np.exp(1j * 2 * np.pi * np.einsum("nk,r->nkr", Kr, r))
    for n in range(sig.shape[0]):
        img += (Eu[n].T * w[n][None, :]) @ Er[n]
    return img


def fit_scalar(a, b):
    """least-squares complex scalar g minimising |a - g b|; returns g, residual RMS / |a|max"""
    g = np.vdot(b, a) / np.vdot(b, b)
    res = a - g * b
    return g, np.sqrt(np.mean(np.abs(res) ** 2)) / np.abs(a).max()


def half_power_width(cut, dx):
    """3 dB width of a 1-D |cut| around its max via linear interpolation (in units of dx)."""
    m = np.abs(cut); i = int(np.argmax(m)); hp = m[i] / np.sqrt(2)
    # walk left
    l = i
    while l > 0 and m[l] > hp: l -= 1
    r = i
    while r < len(m) - 1 and m[r] > hp: r += 1
    # interpolate
    xl = l + (hp - m[l]) / (m[l + 1] - m[l]) if m[l + 1] != m[l] else l
    xr = r - (hp - m[r]) / (m[r - 1] - m[r]) if m[r - 1] != m[r] else r
    return (xr - xl) * dx


def pslr_db(cut):
    m = np.abs(cut); i = int(np.argmax(m))
    # find first minima each side
    l = i
    while l > 0 and m[l - 1] < m[l]: l -= 1
    r = i
    while r < len(m) - 1 and m[r + 1] < m[r]: r += 1
    side = max(m[:l].max() if l > 0 else 0, m[r + 1:].max() if r + 1 < len(m) else 0)
    return 20 * np.log10(side / m[i])


def upsampled_peak(img, factor=16):
    """Sub-pixel peak location via zero-padded FFT interpolation around the max."""
    N_u, N_r = img.shape
    F = np.fft.fft2(img)
    Fp = np.zeros((N_u * factor, N_r * factor), np.complex128)
    # place spectrum with proper zero-padding (ifftshift trick)
    Fs = np.fft.fftshift(F)
    Fp[(N_u * factor - N_u) // 2:(N_u * factor + N_u) // 2, (N_r * factor - N_r) // 2:(N_r * factor + N_r) // 2] = Fs
    up = np.fft.ifft2(np.fft.ifftshift(Fp)) * factor ** 2
    idx = np.unravel_index(np.argmax(np.abs(up)), up.shape)
    return idx[0] / factor, idx[1] / factor, up


def main():
    results = {}
    fc, bw = 9.6e9, 600e6
    L = 40.0
    targets = [(0.0, 0.0, 1.0), (5.0, -5.0, 1.0), (-12.5, 7.25, 0.5)]

    # ------------------------------------------------------------------ 1. baseline
    sig, pvp, meta, Ku, Kr, F, th = make_collection(fc, bw, targets=targets)
    img, bw_r, bw_u, N_r, N_u, rot = run_diffpfa(sig, pvp, meta, fc, L)
    du, dr = L / N_u, L / N_r
    print(f"[1] diffpfa grid: N_u={N_u} N_r={N_r} du={du:.4f} dr={dr:.4f} bw_u={bw_u:.4f} bw_r={bw_r:.4f} rotated={rot}")
    refA, mask = ideal_pfa_reference(Ku, Kr, targets, N_u, N_r, L)
    g, res = fit_scalar(img.astype(np.complex128), refA)
    print(f"    fit to ideal-PFA reference: gain |g|={abs(g):.6g} arg(g)={np.degrees(np.angle(g)):.4f} deg, residual RMS = {20*np.log10(res):.2f} dB rel. peak")
    results["baseline"] = dict(N_u=N_u, N_r=N_r, gain_abs=float(abs(g)), gain_arg_deg=float(np.degrees(np.angle(g))), resid_db=float(20 * np.log10(res)))
    refB = matched_filter_reference(sig, Ku, Kr, F, th, N_u, N_r, L)
    gB, resB = fit_scalar(img.astype(np.complex128), refB)
    print(f"    fit to matched-filter reference: |g|={abs(gB):.6g} residual RMS = {20*np.log10(resB):.2f} dB rel. peak")
    results["baseline"]["resid_vs_matched_filter_db"] = float(20 * np.log10(resB))

    # per-target: location, width, PSLR
    for (u_t, r_t, a) in targets:
        iu = int(round(u_t / du + N_u / 2)); ir = int(round(r_t / dr + N_r / 2))
        win = 12
        sub = img[max(0, iu - win):iu + win, max(0, ir - win):ir + win]
        pu, pr, up = upsampled_peak(sub.astype(np.complex128), 16)
        pu += max(0, iu - win); pr += max(0, ir - win)
        u_found = (pu - N_u / 2) * du; r_found = (pr - N_r / 2) * dr
        cut_u = img[:, int(round(pr))]; cut_r = img[int(round(pu)), :]
        w_u = half_power_width(cut_u, du); w_r = half_power_width(cut_r, dr)
        # analytic uniform-aperture widths
        exp_w_u = 0.8859 / bw_u; exp_w_r = 0.8859 / bw_r
        ph = np.degrees(np.angle(img[int(round(pu)), int(round(pr))]))
        ph_ref = np.degrees(np.angle(refA[int(round(pu)), int(round(pr))]))
        print(f"    target ({u_t:+.2f},{r_t:+.2f}) -> found ({u_found:+.4f},{r_found:+.4f}) err=({u_found-u_t:+.4f},{r_found-r_t:+.4f}) m | "
              f"IRW_u={w_u:.4f} (0.886/BW={exp_w_u:.4f}) IRW_r={w_r:.4f} ({exp_w_r:.4f}) | PSLR_u={pslr_db(cut_u):.2f} dB PSLR_r={pslr_db(cut_r):.2f} dB | "
              f"peak phase {ph:+.2f} deg (ref {ph_ref:+.2f})")
        results.setdefault("targets", []).append(dict(u=u_t, r=r_t, u_found=float(u_found), r_found=float(r_found), irw_u=float(w_u), irw_r=float(w_r),
                                                      exp_irw_u=float(exp_w_u), exp_irw_r=float(exp_w_r), pslr_u=float(pslr_db(cut_u)), pslr_r=float(pslr_db(cut_r)),
                                                      phase_deg=float(ph), phase_ref_deg=float(ph_ref)))

    # ------------------------------------------------------------------ 2. amplitude scaling with L and SCSS
    sig1, pvp1, meta1, *_ = make_collection(fc, bw, targets=[(0, 0, 1)])
    peaks = {}
    for Lx in (20.0, 40.0, 80.0):
        im, *_ = run_diffpfa(sig1, pvp1, meta1, fc, Lx)
        peaks[f"L={Lx}"] = float(np.abs(im).max())
    for s in (0.5, 2.0):
        sg, pv, mt, *_ = make_collection(fc, bw, targets=[(0, 0, 1)], scss_scale=s)
        im, *_ = run_diffpfa(sg, pv, mt, fc, 40.0)
        peaks[f"SCSS x{s} (N_s={sg.shape[1]})"] = float(np.abs(im).max())
    print("[2] peak |img| of a unit point target vs image extent / fast-time sampling:")
    for k, v in peaks.items():
        print(f"    {k:28s} {v:.6g}   ratio to L=40: {v/peaks['L=40.0']:.4f}")
    results["peak_scaling"] = peaks

    # ------------------------------------------------------------------ 3. rotated-axes branch
    # Same physical scene, but the CPHD image-area axes are given with X along range and Y along cross-range.
    sig_r, pvp_r, meta_r, Ku_r, Kr_r, *_ = make_collection(fc, bw, targets=[(5.0, -5.0, 1.0)], uIAX=(0, 1, 0), uIAY=(-1, 0, 0))
    # in this frame the target that was at (u=5,r=-5) in the (x,y) plane... recompute: target position vector
    # p = 5*ex - 5*ey.  Coordinates along new uIAX=ey -> -5 ; along new uIAY=-ex -> -5.
    img_r, bw_r2, bw_u2, N_r2, N_u2, rot2 = run_diffpfa(sig_r, pvp_r, meta_r, fc, L)
    print(f"[3] rotated branch triggered: {rot2}; returned (bw_range,bw_azm)=({bw_r2:.4f},{bw_u2:.4f}) (N_range,N_azm)=({N_r2},{N_u2}); img shape {img_r.shape}")
    iu, ir = np.unravel_index(np.argmax(np.abs(img_r)), img_r.shape)
    print(f"    peak at img[{iu},{ir}] -> along dim0: {(iu-img_r.shape[0]/2)*L/img_r.shape[0]:+.3f} m, along dim1: {(ir-img_r.shape[1]/2)*L/img_r.shape[1]:+.3f} m")
    print(f"    expected physical: -5 m along uIAX(new)=+y_old, -5 m along uIAY(new)=-x_old.  Non-rotated run: bw_u={bw_u:.4f} bw_r={bw_r:.4f}, N_u={N_u}, N_r={N_r}")
    results["rotated"] = dict(rot=bool(rot2), bw_range=float(bw_r2), bw_azm=float(bw_u2), N_range=int(N_r2), N_azm=int(N_u2), shape=list(img_r.shape), peak=[int(iu), int(ir)])

    # ------------------------------------------------------------------ 4. asymmetric image area
    sig_a, pvp_a, meta_a, *_ = make_collection(fc, bw, targets=[(0.0, 0.0, 1.0)])
    img_a, _, _, N_ra, N_ua, _ = pfa_per_polar(channel_signals=[sig_a], channel_pvps=[pvp_a], channel_fxcs=[fc], channel_domain_types=["FX"],
                                              ref_rcv_time=pvp_a["RcvTime"], cphd_meta=meta_a, u_min=0.0, u_max=40.0, r_min=-10.0, r_max=30.0,
                                              image_oversample=1.25, device=DEV)
    iu, ir = np.unravel_index(np.argmax(np.abs(img_a)), img_a.shape)
    print(f"[4] asymmetric area u in [0,40], r in [-10,30]; SRP target (0,0) should be at pixel (u=0 -> idx 0, r=0 -> idx {N_ra*10/40:.1f}); found idx ({iu},{ir}) "
          f"= ({iu*40/N_ua:.2f} m from u_min, {-10+ir*40/N_ra:.2f} m from r_min)  [centre-of-area pixel would be ({N_ua//2},{N_ra//2})]")
    results["asymmetric_area"] = dict(N_u=int(N_ua), N_r=int(N_ra), peak=[int(iu), int(ir)])

    with open(os.path.join(HERE, "out", "a02_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.save(os.path.join(HERE, "out", "a02_img.npy"), img); np.save(os.path.join(HERE, "out", "a02_refA.npy"), refA)


if __name__ == "__main__":
    main()
