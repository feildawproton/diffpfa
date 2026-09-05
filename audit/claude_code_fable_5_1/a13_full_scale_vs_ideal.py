"""
a13_full_scale_vs_ideal.py -- Same comparison as a03 but at the real products' scale (5 km scene,
k ~ 64 cyc/m, spaceborne range), single targets at the centre and near the edges.

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a13_full_scale_vs_ideal.py

Bandwidth and angular span are reduced (60 MHz, 0.2 deg) so the grid stays ~1400 x 2500; the CZT
phase magnitudes (2 pi r_min k ~ 1e6 rad) and the NUFFT bin counts are those of the real 5 km products.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a02_forward_model_vs_exact import make_collection, run_diffpfa, ideal_pfa_reference
from a03_ipr_vs_position import irw_pslr

fc, bw, L = 9.6e9, 60e6, 5000.0
for tgt in [(0.0, 0.0), (1200.0, -1700.0), (2000.0, 0.0), (0.0, 2000.0), (-2300.0, 2300.0)]:
    sig, pvp, meta, Ku, Kr, F, th = make_collection(fc, bw, ns=256, npulse=2048, span_deg=0.2, R=570e3, targets=[(tgt[0], tgt[1], 1.0)])
    img, bw_r, bw_u, N_r, N_u, _ = run_diffpfa(sig, pvp, meta, fc, L)
    ref, _ = ideal_pfa_reference(Ku, Kr, [(tgt[0], tgt[1], 1.0)], N_u, N_r, L)
    du, dr = L / N_u, L / N_r
    for name, im in (("diffpfa", img), ("ideal  ", ref)):
        iu, ir = np.unravel_index(np.argmax(np.abs(im)), im.shape)
        wu, pu = irw_pslr(im[:, ir], du); wr, pr = irw_pslr(im[iu, :], dr)
        print(f"target {tgt}: {name} peak@({iu},{ir}) -> ({(iu-N_u/2)*du:+.2f},{(ir-N_r/2)*dr:+.2f}) m  IRW_u={wu:.3f} PSLR_u={pu:.1f}  IRW_r={wr:.3f}")
    g = np.vdot(ref, img) / np.vdot(ref, ref); res = img - g * ref
    print(f"   residual vs ideal: {20*np.log10(np.sqrt(np.mean(np.abs(res)**2))/np.abs(img).max()):.1f} dB rel peak, |g|={abs(g):.4g}")
