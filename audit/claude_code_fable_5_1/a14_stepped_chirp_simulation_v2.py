"""
a14_stepped_chirp_simulation_v2.py -- Stepped-chirp simulation that models the receiver's local
oscillator and the CPHD producer's compensation explicitly, then shows what the inter-channel phase
rotation in diffpfa/IFA/PFA.py (step 3.2) does to the combined image.

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a14_stepped_chirp_simulation_v2.py

Why a new simulation.  simulation/stepped_chirp_simulation.py writes the CPHD channels directly in
their final form (zero phase at the SRP, RF frequency labels), so it never contains a carrier term
for the rotation to remove; and with its defaults the rotation is exactly an integer number of cycles
(0, 0, -60000.000), so it is invisible.  This script starts one step earlier, at the receiver.

Signal model (monostatic, point scatterers, planar wavefronts, same geometry as the shipped script):
  * Step m transmits carrier f_m, burst pulse n at time t_mn = t_n + m*delta_tau (+ optional jitter),
    from the platform position at t_mn.
  * The receiver demodulates with a local oscillator  exp(-j(2 pi f_m t + theta_mn)).  theta_mn is the
    LO phase convention of the hardware; we use the worst case, an LO whose phase is reset at each
    pulse's transmit time (theta_mn = 2 pi f_m t_mn mod 2 pi), so the raw data carry a large,
    channel- and pulse-dependent phase.
  * After range compression the raw sample at baseband frequency f_b for scatterer i at range R_i is
        exp(-j 2 pi (f_m + f_b) (2 R_i / c) - j theta_mn).
  * CPHD compensation (CPHD DIDD sec. 1.4): the producer knows R_SRP(t_mn) and its own receiver timing,
    and multiplies by exp(+j 2 pi (f_m+f_b)(2 R_SRP/c) + j theta_mn) so that the SRP echo has zero
    phase in every vector of every channel.  The result is labelled with the RF frequency
    fx = f_m + f_b (SC0 = f_m + f_b,0, SCSS = step).  This is the CPHD channel.

Experiments:
  [1] Verify the compliant channels have zero SRP phase (all channels, all pulses) although the raw
      data did not.
  [2] For several delta_tau values (integer-cycle and not, with and without PRI jitter) form the
      3-channel image with the shipped code (rotation ON), with the rotation neutralised (OFF: pass
      fxc = fc_global for every channel, the only other use of fxc is the dead RVP branch), and the
      monolithic 600 MHz reference; report complex coherence, range IRW, PSLR, peak ratio.
  [3] Inter-band phase diagnostic: image each channel alone and read the phase of the SRP scatterer;
      compliant data give equal phases; this is the check to run on real multi-step CPHD.
  [4] A NON-compliant producer that forgets theta in its compensation, leaving a constant residual
      phase per channel: the shipped rotation does not repair it (its model is 2 pi (fc-f_m) m dtau,
      not the hardware's theta), a per-channel phase MEASURED from [3] does.
Writes out/a14_results.json and out/a14_range_cuts.png.
"""
import os, sys, json, io, contextlib
import numpy as np
import torch

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT as C
from diffpfa.types import CPHDMetadata, ImageAreaBounds
from a03_ipr_vs_position import irw_pslr

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

FC, BW_TOTAL, NSUB = 9.6e9, 600e6, 3
NS_SUB, NPULSE, PRF = 256, 512, 1000.0
SLANT_RANGE, SPAN_DEG, SAT_VEL = 15000.0, 2.0, 7500.0
TARGETS = [(0.0, 0.0, 1.0), (5.0, 8.0, 1.0), (-7.0, -12.0, 1.0), (10.0, -5.0, 1.0)]   # (u, r, amplitude), SRP scatterer first
L = 40.0
uIAX, uIAY, SRP = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.zeros(3)


def platform(t):
    """Position on the arc at slow time t (t=0..t_dur maps to the +-SPAN/2 polar span)."""
    t_dur = (NPULSE - 1) / PRF
    th = -np.radians(SPAN_DEG / 2) + (t / t_dur) * np.radians(SPAN_DEG)
    return np.stack([SLANT_RANGE * np.sin(th), -SLANT_RANGE * np.cos(th), np.zeros_like(th)], -1)


def look_components(pos):
    P = SRP[None, :] - pos; Pn = np.linalg.norm(P, axis=1)
    return P @ uIAX / Pn, P @ uIAY / Pn, Pn


def make_channels(delta_tau, jitter_us=0.0, compliant=True, seed=0):
    """Return list of (signal, pvp, fxc, theta_residual) CPHD channels and the burst timing."""
    rng = np.random.default_rng(seed)
    bw_sub = BW_TOTAL / NSUB
    t_burst = np.arange(NPULSE) / PRF
    chans = []
    for m in range(NSUB):
        f_m = (FC - BW_TOTAL / 2) + (m + 0.5) * bw_sub                 # step carrier
        step = bw_sub / NS_SUB
        f_b = -bw_sub / 2 + np.arange(NS_SUB) * step                    # baseband frequency axis
        fx = f_m + f_b                                                  # RF frequency labels
        t_mn = t_burst + m * delta_tau + rng.uniform(-jitter_us, jitter_us, NPULSE) * 1e-6
        pos = platform(t_mn)
        cos_t, sin_t, R_srp = look_components(pos)
        theta = np.mod(2 * np.pi * f_m * t_mn, 2 * np.pi)              # LO phase (reset at each pulse)
        # raw demodulated, range-compressed data: absolute ranges, LO phase included
        def raw_for(targets):
            r = np.zeros((NPULSE, NS_SUB), np.complex128)
            for (u_t, r_t, a) in targets:
                R_i = R_srp + (u_t * cos_t + r_t * sin_t)               # planar-wavefront range to scatterer
                r += a * np.exp(-1j * 2 * np.pi * fx[None, :] * (2 * R_i[:, None] / C) - 1j * theta[:, None])
            return r
        raw = raw_for(TARGETS)
        raw_srp_only = raw_for([(0.0, 0.0, 1.0)])                       # for the phase diagnostic only
        # CPHD producer's compensation: zero the SRP echo phase using known R_SRP and receiver timing
        comp = np.exp(+1j * 2 * np.pi * fx[None, :] * (2 * R_srp[:, None] / C) + 1j * theta[:, None])
        theta_resid = 0.0
        if not compliant:
            # a producer that forgets the LO term; modelled as a per-channel constant (LO reset at burst start)
            theta_c = np.mod(2 * np.pi * f_m * (m * delta_tau), 2 * np.pi)
            comp = comp * np.exp(-1j * theta_c)
            theta_resid = theta_c
        sig = raw * comp
        pvp = dict(SRPPos=np.tile(SRP, (NPULSE, 1)), TxPos=pos, RcvPos=pos, TxVel=np.tile([SAT_VEL, 0, 0], (NPULSE, 1)), RcvVel=np.tile([SAT_VEL, 0, 0], (NPULSE, 1)),
                   SC0=np.full(NPULSE, fx[0]), SCSS=np.full(NPULSE, step), TxTime=t_mn, RcvTime=t_mn)
        chans.append(dict(sig=sig.astype(np.complex64), pvp=pvp, fxc=f_m, theta_resid=theta_resid,
                          srp_phase_raw=np.angle(raw_srp_only[:, NS_SUB // 2]), srp_phase_cphd=np.angle((raw_srp_only * comp)[:, NS_SUB // 2])))
    return chans, t_burst


def make_monolithic(t_burst):
    ns = NSUB * NS_SUB; step = BW_TOTAL / ns
    fx = (FC - BW_TOTAL / 2) + np.arange(ns) * step
    pos = platform(t_burst); cos_t, sin_t, R_srp = look_components(pos)
    sig = np.zeros((NPULSE, ns), np.complex128)
    for (u_t, r_t, a) in TARGETS:
        sig += a * np.exp(-1j * 2 * np.pi * fx[None, :] * (2 * (u_t * cos_t + r_t * sin_t)[:, None] / C))
    pvp = dict(SRPPos=np.tile(SRP, (NPULSE, 1)), TxPos=pos, RcvPos=pos, TxVel=np.tile([SAT_VEL, 0, 0], (NPULSE, 1)), RcvVel=np.tile([SAT_VEL, 0, 0], (NPULSE, 1)),
               SC0=np.full(NPULSE, fx[0]), SCSS=np.full(NPULSE, step), TxTime=t_burst, RcvTime=t_burst)
    return sig.astype(np.complex64), pvp


def meta(fmin, fmax, pos0):
    return CPHDMetadata(domain_type="FX", sgn=-1, global_fx_min=fmin, global_fx_max=fmax, iarp_ecf=SRP, uIAX=uIAX, uIAY=uIAY, ref_ch_id="0",
                        image_area=ImageAreaBounds(-L / 2, -L / 2, L / 2, L / 2, None), extended_area=None, collection_start=None, radar_mode="SPOTLIGHT",
                        classification="U", srp_ecf=SRP, arp_pos_coa=pos0, arp_vel_coa=np.array([SAT_VEL, 0, 0]), side_of_track="R", line_spacing=None, sample_spacing=None, raw_meta=None)


def form(chans, ref_rcv_time, rotation, spacing=None, phase_fix=None):
    """rotation=True: shipped code path.  False: fxc := fc_global (as pfa_per_polar computes it from the
    channels given) for every channel, which makes the rotation phase identically zero; fxc has no other
    live use (the RVP branch needs a PVP that CPHD does not define)."""
    sigs = [c["sig"] for c in chans]
    if phase_fix is not None:
        sigs = [s * np.exp(-1j * p).astype(np.complex64) for s, p in zip(sigs, phase_fix)]
    fmin = min(c["pvp"]["SC0"][0] for c in chans); fmax = max(c["pvp"]["SC0"][0] + NS_SUB * c["pvp"]["SCSS"][0] for c in chans)
    fxcs = [c["fxc"] for c in chans] if rotation else [(fmin + fmax) / 2] * len(chans)
    with contextlib.redirect_stdout(io.StringIO()):
        img, bw_r, bw_u, N_r, N_u, _ = pfa_per_polar(channel_signals=sigs, channel_pvps=[c["pvp"] for c in chans], channel_fxcs=fxcs, channel_domain_types=["FX"] * len(chans),
                                                    ref_rcv_time=ref_rcv_time, cphd_meta=meta(fmin, fmax, chans[0]["pvp"]["TxPos"][NPULSE // 2]),
                                                    u_min=-L / 2, u_max=L / 2, r_min=-L / 2, r_max=L / 2, custom_pixel_spacing=spacing, image_oversample=1.25, device=DEV)
    return img, (L / N_u, L / N_r), bw_r


def metrics(img, ref, spacing):
    du, dr = spacing
    coh = float(abs(np.vdot(img, ref)) / (np.linalg.norm(img) * np.linalg.norm(ref)))
    N_u, N_r = img.shape
    cut = img[N_u // 2, :]; cut_ref = ref[N_u // 2, :]
    irw, pslr = irw_pslr(cut, dr); irw0, pslr0 = irw_pslr(cut_ref, dr)
    return dict(coherence=coh, irw_r=float(irw), irw_r_ref=float(irw0), pslr_r=float(pslr), pslr_r_ref=float(pslr0), peak_ratio=float(np.abs(cut).max() / np.abs(cut_ref).max()))


def main():
    out = {}
    print("=" * 100)
    print("[1] compliant CPHD channels: SRP phase per channel (max |phase| over pulses, radians)")
    chans, t_burst = make_channels(137.3217e-6)
    for m, c in enumerate(chans):
        print(f"    channel {m}: raw demodulated data max|SRP phase| = {np.abs(c['srp_phase_raw']).max():.3f}   after CPHD compensation = {np.abs(c['srp_phase_cphd']).max():.2e}")
    out["srp_phase_after_compensation_max"] = float(max(np.abs(c["srp_phase_cphd"]).max() for c in chans))

    print("\n[2] 3 x 200 MHz stepped chirp vs monolithic 600 MHz reference; rotation ON = shipped code, OFF = rotation phase forced to zero")
    print(f"    {'burst timing':34s} | {'rotation cycles ch2':>19s} | {'ON: coh':>8s} {'IRW_r':>6s} {'PSLR':>6s} {'peak':>5s} | {'OFF: coh':>8s} {'IRW_r':>6s} {'PSLR':>6s} {'peak':>5s} | {'ref IRW_r':>9s}")
    sweep = [(150e-6, 0.0, "150 us (shipped default)"), (137e-6, 0.0, "137 us"), (137.3217e-6, 0.0, "137.3217 us"), (100.00025e-6, 0.0, "100.00025 us"),
             (150e-6, 5.0, "150 us + PRI jitter +-5 us")]
    rows = []
    cuts_for_plot = None
    for dtau, jit, label in sweep:
        chans, t_burst = make_channels(dtau, jitter_us=jit)
        ref_sig, ref_pvp = make_monolithic(t_burst)
        ref_rcv = chans[0]["pvp"]["RcvTime"]                             # IFP.run() uses the reference channel's own RcvTime
        img_on, spacing, bw_r = form(chans, ref_rcv, rotation=True)
        img_off, _, _ = form(chans, ref_rcv, rotation=False, spacing=spacing)
        ref_ch = [dict(sig=ref_sig, pvp=ref_pvp, fxc=FC)]
        img_ref, _, _ = form(ref_ch, t_burst, rotation=False, spacing=spacing)
        m_on = metrics(img_on, img_ref, spacing); m_off = metrics(img_off, img_ref, spacing)
        cyc = (FC - chans[2]["fxc"]) * (chans[2]["pvp"]["RcvTime"][0] - ref_rcv[0])
        print(f"    {label:34s} | {cyc:+19.3f} | {m_on['coherence']:8.4f} {m_on['irw_r']:6.3f} {m_on['pslr_r']:6.1f} {m_on['peak_ratio']:5.3f} | "
              f"{m_off['coherence']:8.4f} {m_off['irw_r']:6.3f} {m_off['pslr_r']:6.1f} {m_off['peak_ratio']:5.3f} | {m_on['irw_r_ref']:9.3f}")
        rows.append(dict(label=label, delta_tau=dtau, jitter_us=jit, rotation_cycles_ch2=float(cyc), on=m_on, off=m_off))
        if label.startswith("137.3217"):
            cuts_for_plot = (img_ref, img_on, img_off, spacing)
    out["sweep"] = rows
    print(f"    analytic: uniform 600 MHz -> IRW_r = 0.8859/{bw_r:.3f} = {0.8859/bw_r:.3f} m; single 200 MHz sub-band -> {3*0.8859/bw_r:.3f} m")

    print("\n[3] inter-band phase diagnostic (each channel imaged alone; phase of the SRP scatterer)")
    chans, t_burst = make_channels(137.3217e-6)
    _, spacing, _ = form(chans, t_burst, rotation=False)
    ph = []
    for m, c in enumerate(chans):
        im, _, _ = form([c], t_burst, rotation=False, spacing=spacing)
        N_u, N_r = im.shape; ph.append(float(np.angle(im[N_u // 2, N_r // 2])))
    print("    compliant data: SRP phase per sub-band image (deg) =", ", ".join(f"{np.degrees(p):+.2f}" for p in ph), " -> inter-band differences ~ 0: nothing to correct")
    out["interband_phase_compliant_deg"] = [float(np.degrees(p)) for p in ph]

    print("\n[4] NON-compliant producer: constant residual LO phase per channel left in the data")
    chans_nc, t_burst = make_channels(137.3217e-6, compliant=False)
    ref_sig, ref_pvp = make_monolithic(t_burst)
    img_ref, spacing, _ = form([dict(sig=ref_sig, pvp=ref_pvp, fxc=FC)], t_burst, rotation=False)
    ph_nc = []
    for c in chans_nc:
        im, _, _ = form([c], t_burst, rotation=False, spacing=spacing); N_u, N_r = im.shape; ph_nc.append(float(np.angle(im[N_u // 2, N_r // 2])))
    print("    residual theta per channel (deg, truth)   :", ", ".join(f"{np.degrees(c['theta_resid']):+8.2f}" for c in chans_nc))
    print("    measured SRP phase per sub-band image (deg):", ", ".join(f"{np.degrees(p):+8.2f}" for p in ph_nc))
    print("    shipped rotation model 2pi(fc-fm) m dtau (deg):", ", ".join(f"{np.degrees(np.mod(2*np.pi*(FC-c['fxc'])*(c['pvp']['RcvTime'][0]-t_burst[0]), 2*np.pi)):+8.2f}" for c in chans_nc))
    res4 = {}
    for label, rot, fix in [("no correction", False, None), ("shipped rotation", True, None), ("measured per-channel phase (from [3])", False, [p - ph_nc[0] for p in ph_nc])]:
        im, _, _ = form(chans_nc, t_burst, rotation=rot, spacing=spacing, phase_fix=fix)
        mm = metrics(im, img_ref, spacing); res4[label] = mm
        print(f"    {label:38s}: coherence={mm['coherence']:.4f}  IRW_r={mm['irw_r']:.3f} m  PSLR={mm['pslr_r']:.1f} dB  peak ratio={mm['peak_ratio']:.3f}")
    out["noncompliant"] = dict(theta_true_deg=[float(np.degrees(c["theta_resid"])) for c in chans_nc], measured_deg=[float(np.degrees(p)) for p in ph_nc], results=res4)

    with open(os.path.join(HERE, "out", "a14_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from a03_ipr_vs_position import upsample_cut
        img_ref, img_on, img_off, spacing = cuts_for_plot
        N_u, N_r = img_ref.shape; up = 8
        r = (np.arange(N_r * up) - N_r * up / 2) * spacing[1] / up
        ref_pk = np.abs(upsample_cut(img_ref[N_u // 2, :], up)).max()
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for im, lab, st in ((img_ref, "monolithic 600 MHz reference", "k-"), (img_off, "3 sub-bands, rotation OFF", "b--"), (img_on, "3 sub-bands, rotation ON (shipped code)", "r-")):
            c = np.abs(upsample_cut(im[N_u // 2, :], up)); ax.plot(r, 20 * np.log10(c / ref_pk + 1e-9), st, label=lab, lw=1.4)
        ax.set_xlim(-3, 3); ax.set_ylim(-45, 2); ax.grid(alpha=.4); ax.legend(); ax.set_xlabel("range r (m)"); ax.set_ylabel("dB rel. reference peak")
        ax.set_title("Range cut through the SRP scatterer (8x upsampled), delta_tau = 137.3217 us\nrotation phase on channel 2 = -54928.68 cycles")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "out", "a14_range_cuts.png"), dpi=120)
        print("\nwrote", os.path.join(HERE, "out", "a14_range_cuts.png"))
    except Exception as e:
        print("no figure:", e)


if __name__ == "__main__":
    main()
