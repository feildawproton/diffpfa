"""
a05_test_bite.py -- Can the shipped tests fail?  Mutation testing without touching the source.

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a05_test_bite.py

Each scenario runs in a fresh subprocess: import the library, monkey-patch one defect in at
runtime, then call one shipped test function.  A useful test goes RED for a defect it is meant
to guard against.  A test that stays GREEN under a plausible defect is blind to that defect.
Nothing in tests/ or diffpfa/ is modified.
"""
import os, sys, subprocess, json, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

PRELUDE = textwrap.dedent("""
    import sys, os, math, numpy as np, torch
    sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'tests')
    import diffpfa.IFA.kspace as ks, diffpfa.IFA.PFA as PFA, diffpfa.IFA.channel.czt_torch as czt
    import diffpfa.IFA.channel.nufft_torch as nf, diffpfa.IFA.channel.pfa_channel as pc, diffpfa.sicd_geometry as sg
""")

SCENARIOS = [
    # ---- test_czt_nufft.py ----
    ("test_czt_nufft.test_czt_1d_vs_direct_sum", "baseline (no mutation)", ""),
    ("test_czt_nufft.test_czt_1d_vs_direct_sum", "CZT post-chirp sign flipped", """
        _orig = czt.czt_1d_torch
        def bad(x, M, r_min, r_max, k_step, k_start, dim=-1):
            out = _orig(x, M, r_min, r_max, k_step, k_start, dim)
            m = torch.arange(M, dtype=torch.float64, device=x.device); dr=(r_max-r_min)/max(M-1,1)
            return out * torch.exp(-2j*math.pi*k_start*m*dr)  # remove the linear post-chirp -> wrong
        czt.czt_1d_torch = bad
        import test_czt_nufft; test_czt_nufft.czt_1d_torch = bad
    """),
    ("test_czt_nufft.test_kaiser_bessel_kernel_properties", "baseline (no mutation)", ""),
    ("test_czt_nufft.test_kaiser_bessel_kernel_properties", "kernel shape wrong (beta ignored -> rect window)", """
        def bad(x, J=6, beta=13.9086):
            return (torch.abs(x) <= J/2.0).to(x.dtype)
        nf.kaiser_bessel_kernel_1d = bad
        import test_czt_nufft; test_czt_nufft.kaiser_bessel_kernel_1d = bad
    """),
    # ---- test_pfa_coherence.py ----
    ("test_pfa_coherence.test_point_target_localization", "baseline (no mutation)", ""),
    ("test_pfa_coherence.test_point_target_localization", "look-vector sign flipped (P = APC - SRP)", """
        _o = ks._compute_look_vectors
        def bad(pvp, device=torch.device('cpu')):
            return -_o(pvp, device)
        ks._compute_look_vectors = bad
    """),
    ("test_pfa_coherence.test_point_target_localization", "spatial frequency uses F/c instead of 2F/c", """
        ks.SPEED_OF_LIGHT = ks.SPEED_OF_LIGHT * 2.0
    """),
    ("test_pfa_coherence.test_point_target_localization", "KB deconvolution removed", """
        def no_deconv(grid, M_u, M_r, device):
            grid = torch.fft.ifftshift(grid); img = torch.fft.ifft2(grid); img.mul_(M_u*M_r); return torch.fft.fftshift(img)
        PFA._apply_ifft_and_deconv = no_deconv
    """),
    ("test_pfa_coherence.test_point_target_localization", "NUFFT kernel replaced by nearest-neighbour (J=1 rect)", """
        def nn(x, J=6, beta=13.9086):
            return (torch.abs(x) <= 0.5).to(x.dtype)
        nf.kaiser_bessel_kernel_1d = nn
    """),
    ("test_pfa_coherence.test_point_target_localization", "RVP deskew applies a huge quadratic phase (gamma bogus)", """
        _o = pc._deskew_rvp
        def bad(signal, fxc, pvp, N_samples, device):
            pvp = dict(pvp); pvp['TxFMRate'] = np.full(len(pvp['SC0']), 1e11)  # 100 GHz/s
            return _o(signal, fxc, pvp, N_samples, device)
        pc._deskew_rvp = bad
    """),
    ("test_pfa_coherence.test_point_target_localization", "CZT resampler conj dropped (spectrum reversed)", """
        _o = czt.czt_resample_kspace_1d
        def bad(*a, **k):
            return torch.conj(_o(*a, **k))
        czt.czt_resample_kspace_1d = bad; pc.czt_resample_kspace_1d = bad
    """),
    ("test_pfa_coherence.test_point_target_localization", "image amplitude scaled by 1e-3 and phase by 90 deg", """
        _o = PFA._apply_ifft_and_deconv
        def bad(grid, M_u, M_r, device):
            return _o(grid, M_u, M_r, device) * 1e-3 * 1j
        PFA._apply_ifft_and_deconv = bad
    """),
    ("test_pfa_coherence.test_point_target_localization", "azimuth k-space centre wrong by +bw/2 (k_ctr_u += bw_u/2)", """
        _o = pc.process_cztnufft
        def bad(**k):
            k = dict(k); k['k_ctr_u'] = k['k_ctr_u'] + 0.5*1.0  # ~ +0.5 cyc/m vs bw_u 2.3
            return _o(**k)
        PFA.process_cztnufft = bad
    """),
    # ---- test_geometry.py ----
    ("test_geometry.test_compute_scp_geometry_umbra_reference", "baseline (no mutation)", ""),
    ("test_geometry.test_compute_scp_geometry_umbra_reference", "twist sign flipped", """
        _o = sg.compute_scp_geometry
        def bad(*a, **k):
            g = _o(*a, **k); g['TwistAng'] = -g['TwistAng']; return g
        sg.compute_scp_geometry = bad
        import test_geometry; test_geometry.compute_scp_geometry = bad
    """),
    ("test_geometry.test_fit_arp_poly_residuals", "baseline (no mutation)", ""),
    ("test_geometry.test_fit_arp_poly_residuals", "TxTime shifted by +0.3 s (collection-start offset)", """
        _o = sg.fit_arp_poly
        def bad(pvp, deg=5):
            pvp = dict(pvp); pvp['TxTime'] = pvp['TxTime'] + 0.3
            return _o(pvp, deg)
        sg.fit_arp_poly = bad
        import test_geometry; test_geometry.fit_arp_poly = bad
    """),
]


def run(test_id, mutation):
    mod, fn = test_id.split(".")
    code = PRELUDE + textwrap.dedent(mutation) + textwrap.dedent(f"""
        import {mod}
        try:
            {mod}.{fn}()
            print("RESULT: PASS")
        except AssertionError as e:
            print("RESULT: FAIL", str(e)[:160].replace(chr(10), ' '))
        except Exception as e:
            print("RESULT: ERROR", type(e).__name__, str(e)[:160].replace(chr(10), ' '))
    """)
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=os.getcwd(), timeout=900)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT:"):
            return line
    return "RESULT: NO-OUTPUT " + (p.stderr[-300:].replace("\n", " | "))


def main():
    rows = []
    print(f"{'test':58s} | {'injected defect':60s} | outcome")
    print("-" * 150)
    for test_id, label, mut in SCENARIOS:
        r = run(test_id, mut)
        rows.append(dict(test=test_id, defect=label, result=r))
        print(f"{test_id:58s} | {label:60s} | {r}")
    with open(os.path.join(HERE, "out", "a05_results.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
