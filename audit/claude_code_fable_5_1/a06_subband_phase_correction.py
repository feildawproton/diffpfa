"""
a06_subband_phase_correction.py -- Is the inter-channel phase correction
        phi = -2 pi (f_c,global - f_xc) * (RcvTime - RcvTime_ref)
physically justified for CPHD input, and did the shipped simulation actually exercise it?

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a06_subband_phase_correction.py

Argument.  A CPHD FX-domain signal is, by definition (CPHD DIDD 1.1.0 sec. 4/5), motion
compensated to the SRP: for a scatterer at delta-R from the SRP the phase is
        phi(n,k) = SGN * 2 pi * fx(n,k) * (2 dR_n / c)
with fx the *RF* frequency.  A scatterer at the SRP has zero phase on every vector of every
channel, whatever LO each channel used.  So no per-channel carrier term survives, and there is
nothing for the correction to remove.  Its only effect is to multiply channel m by a constant
        exp(-j 2 pi (f_c,global - f_xc,m) * m * delta_tau)
(constant because RcvTime - RcvTime_ref is the same for every pulse of a burst channel).
That constant destroys the coherence between sub-bands unless it happens to be a multiple of
2 pi.  In simulation/stepped_chirp_simulation.py the defaults are
        f_c,global - f_xc,m in {+200 MHz, 0, -200 MHz},   delta_tau = 150 us
so (f_c - f_xc) * m * delta_tau = 0, 0, -60000 cycles exactly -> the correction is an identity
and the shipped simulation cannot distinguish "correction present" from "absent".

Test: (a) reproduce that the default is an integer number of cycles; (b) rerun the shipped
simulation's own coherence metric with delta_tau = 137 us with the library as-is and with
the correction monkey-patched to zero (in-process, no source edits); (c) explain the sign of
what happens to the range IPR.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import diffpfa.IFA.PFA as PFA
from simulation.stepped_chirp_simulation import run_stepped_chirp_simulation

HERE = os.path.dirname(os.path.abspath(__file__))


def cycles_default():
    fc, bw, nsub = 9.6e9, 600e6, 3
    bw_sub = bw / nsub
    out = []
    for m in range(nsub):
        fsub = (fc - bw / 2) + (m + 0.5) * bw_sub
        for dtau in (150e-6, 137e-6, 137.3217e-6):
            cyc = (fc - fsub) * m * dtau
            out.append(dict(m=m, delta_tau_us=dtau * 1e6, f_offset_MHz=(fc - fsub) / 1e6, cycles=cyc, frac=cyc - round(cyc)))
    return out


def run(delta_tau, patch_off):
    """Run the shipped simulation with/without the correction, return complex coherence and range IRW."""
    import io, contextlib
    orig_exp = torch.exp
    hits = [0]
    if patch_off:
        # neutralise ONLY the correction: pfa_per_polar computes corr_term = torch.exp(1j*phase_corr).unsqueeze(1)
        # phase_corr is 1-D of length num_pulses (512 in the shipped simulation); every other torch.exp in the
        # chain is >= 2-D.  Wrap torch.exp so that call returns ones, and count interceptions.
        def fake_exp(x, *a, **k):
            if x.is_complex() and x.ndim == 1 and x.shape[0] == 512 and torch.all(x.real == 0):
                hits[0] += 1
                return torch.ones_like(x)
            return orig_exp(x, *a, **k)
        PFA.torch.exp = fake_exp
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = run_stepped_chirp_simulation(delta_tau=delta_tau)
    finally:
        PFA.torch.exp = orig_exp
    if patch_off:
        print(f"      (correction neutralised {hits[0]} times; expected 3+1+2+1 = 7 channel passes)")
    img = res["img_multi"]; ref = res["img_ref"]
    N_azm, N_range, dr = res["N_azm"], res["N_range"], res["dr"]
    cut = np.abs(img[N_azm // 2, :]); cut_ref = np.abs(ref[N_azm // 2, :])
    def irw(c):
        m = c / c.max(); i = int(np.argmax(m)); hp = 1 / np.sqrt(2)
        l = i
        while l > 0 and m[l] > hp: l -= 1
        r = i
        while r < len(m) - 1 and m[r] > hp: r += 1
        return (r - l) * dr
    # peak ratio at centre target
    return dict(coherence=res["complex_coherence"], irw_multi_m=irw(cut), irw_ref_m=irw(cut_ref),
                peak_ratio=float(cut.max() / cut_ref.max()))


def main():
    out = {"default_cycles": cycles_default()}
    print("[a] cycles of correction phase per channel for the shipped simulation defaults:")
    for d in out["default_cycles"]:
        print(f"    m={d['m']} dtau={d['delta_tau_us']:.0f}us  (fc-fxc)={d['f_offset_MHz']:+.0f} MHz  cycles={d['cycles']:+.3f}  fractional part={d['frac']:+.3f}")
    print("[b] shipped simulation metric, correction on vs off")
    for dtau in (150e-6, 137.3217e-6):
        for off in (False, True):
            r = run(dtau, off)
            key = f"dtau={dtau*1e6:.4f}us correction={'OFF' if off else 'on'}"
            out[key] = r
            print(f"    {key:38s} coherence={r['coherence']:.6f}  range IRW multi={r['irw_multi_m']:.3f} m (ref {r['irw_ref_m']:.3f} m)  peak ratio={r['peak_ratio']:.4f}")
    with open(os.path.join(HERE, "out", "a06_results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
