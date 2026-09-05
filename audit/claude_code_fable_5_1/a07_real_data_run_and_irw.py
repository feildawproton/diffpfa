"""
a07_real_data_run_and_irw.py -- Form a real image with the library as shipped, then measure what the
product's pixels actually contain versus what its XML declares.  The vendor SICD of the same
collection is run through the identical measurement as the control.

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a07_real_data_run_and_irw.py [cphd_stem]
default stem: 2023-09-11-10-37-05_UMBRA-05  (smallest collection; ~140 MB CPHD)

Measurements per SICD (vendor control and diffpfa product):
  * k-space support along Row and Col from a 2048^2 centre chip (occupied extent at -10 dB of the
    in-band mean, centroid, in-band flatness)          -> measured ImpRespBW, weighting, KCtr-in-pixels
  * aperture -> IPR: |W(k)| = sqrt(mean power spectrum), zero-pad x8, inverse transform, half-power
    width; k_meas = IRW_meas * BW_meas                  -> is ImpRespWid = k/BW with k = 0.886 or 1.0?
  * declared k = ImpRespWid * ImpRespBW
Also: run sarkit's SicdConsistency on the fresh product and save the report.
"""
import os, sys, json, time, subprocess
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import sarkit.sicd as ss
from diffpfa.IFP import IFAProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = sys.argv[1] if len(sys.argv) > 1 else "2023-09-11-10-37-05_UMBRA-05"
DATA = "/home/feildaw/data"


def kspace_and_irw(path, n=2048, thresh_db=-10.0):
    with open(path, "rb") as f, ss.NitfReader(f) as r:
        x = r.metadata.xmltree; h = ss.XmlHelper(x)
        nr = h.load("./{*}ImageData/{*}NumRows"); nc = h.load("./{*}ImageData/{*}NumCols")
        n = min(n, nr, nc)
        r0 = nr // 2 - n // 2; c0 = nc // 2 - n // 2
        chip, _ = r.read_sub_image(start_row=r0, start_col=c0, stop_row=r0 + n, stop_col=c0 + n)
        decl = {}
        for d in ("Row", "Col"):
            decl[d] = dict(SS=h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}SS"), BW=h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespBW"),
                           IRW=h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespWid"), KCtr=h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}KCtr"))
            decl[d]["k_declared"] = decl[d]["IRW"] * decl[d]["BW"]
    chip = chip.astype(np.complex64)
    out = {"declared": decl, "chip": n}
    for ax, d in ((0, "Row"), (1, "Col")):
        ssv = decl[d]["SS"]
        F = np.fft.fftshift(np.fft.fft(chip, axis=ax), axes=ax)
        P = (np.abs(F) ** 2).mean(axis=1 - ax)
        k = np.fft.fftshift(np.fft.fftfreq(n, d=ssv))
        # in-band level: median of the top half of P; support = where P > level*10^(thresh/10)
        lvl = np.median(np.sort(P)[n // 2:])
        occ = P > lvl * 10 ** (thresh_db / 10)
        idx = np.where(occ)[0]
        dk = k[1] - k[0]
        k_lo, k_hi = k[idx[0]] - dk / 2, k[idx[-1]] + dk / 2
        bw_meas = k_hi - k_lo
        inband = P[occ]
        centroid = float((k * P).sum() / P.sum())
        # aperture -> IPR
        W = np.sqrt(P) * occ
        pad = 8
        Wp = np.zeros(n * pad, complex); Wp[(n * pad - n) // 2:(n * pad + n) // 2] = W
        ipr = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Wp))))
        dx = ssv / pad
        i = int(np.argmax(ipr)); hp = ipr[i] / np.sqrt(2)
        l = i
        while l > 0 and ipr[l] > hp: l -= 1
        rr = i
        while rr < len(ipr) - 1 and ipr[rr] > hp: rr += 1
        # sub-sample interpolation
        xl = l + (hp - ipr[l]) / (ipr[l + 1] - ipr[l]); xr = rr - (hp - ipr[rr]) / (ipr[rr - 1] - ipr[rr])
        irw_meas = (xr - xl) * dx
        # taper shape: ratio of in-band edge mean to centre mean (uniform -> ~1)
        q = len(inband) // 5
        edge_over_centre = float((inband[:q].mean() + inband[-q:].mean()) / (2 * inband[2 * q:3 * q].mean()))
        out[d] = dict(nyquist=0.5 / ssv, k_support=[float(k_lo), float(k_hi)], bw_measured=float(bw_meas), bw_declared=decl[d]["BW"],
                      bw_ratio=float(bw_meas / decl[d]["BW"]), centroid=centroid, inband_std_over_mean=float(inband.std() / inband.mean()),
                      edge_over_centre=edge_over_centre, irw_measured=float(irw_meas), irw_declared=decl[d]["IRW"],
                      k_measured=float(irw_meas * bw_meas), k_declared=decl[d]["k_declared"],
                      irw_declared_over_measured=float(decl[d]["IRW"] / irw_meas))
    return out


def main():
    cphd = os.path.join(DATA, f"{STEM}_CPHD.cphd")
    vendor = os.path.join(DATA, f"{STEM}_SICD.nitf")
    outdir = os.path.join(HERE, "out", f"run_{STEM}")
    os.makedirs(outdir, exist_ok=True)
    res = {"stem": STEM}

    t0 = time.perf_counter()
    proc = IFAProcessor(cphd_path=cphd, output_dir=outdir, image_plane="SLANT", batch_size=256, device="cuda")
    with torch.inference_mode():
        files, rt, pt, wt = proc.run()
    total = time.perf_counter() - t0
    res["timing_s"] = dict(read=rt, proc=pt, write=wt, wall=total)
    res["peak_gpu_alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    print(f"[run] {files[0]}\n      read={rt:.2f}s proc={pt:.2f}s write={wt:.2f}s wall={total:.2f}s peak GPU alloc={res['peak_gpu_alloc_GB']:.2f} GB")
    prod = files[0]

    # consistency check
    chk = subprocess.run([os.path.join(os.path.dirname(sys.executable), "sicdcheck"), prod], capture_output=True, text=True)
    with open(os.path.join(outdir, "sicdcheck.txt"), "w") as f:
        f.write(chk.stdout + chk.stderr)
    errs = [l.strip() for l in chk.stdout.splitlines() if "[Error]" in l or "[Warning]" in l]
    res["sicdcheck"] = errs
    print(f"[sicdcheck] {len(errs)} error/warning lines:")
    for e in errs:
        print("     ", e)

    # pixel measurements, vendor first as control
    for label, path in (("vendor_control", vendor), ("diffpfa_fresh", prod)):
        m = kspace_and_irw(path)
        res[label] = m
        print(f"[{label}] {os.path.basename(path)}  chip {m['chip']}^2")
        for d in ("Row", "Col"):
            v = m[d]
            print(f"   {d}: declared BW={v['bw_declared']:.4f} measured BW={v['bw_measured']:.4f} (ratio {v['bw_ratio']:.3f}) | spectrum centroid {v['centroid']:+.4f} cyc/m (Nyquist {v['nyquist']:.3f}) "
                  f"| in-band edge/centre {v['edge_over_centre']:.3f} std/mean {v['inband_std_over_mean']:.3f}")
            print(f"        declared IRW={v['irw_declared']:.4f} m  measured IRW={v['irw_measured']:.4f} m  ratio decl/meas={v['irw_declared_over_measured']:.4f} | k declared={v['k_declared']:.4f} k measured={v['k_measured']:.4f}")
    with open(os.path.join(HERE, "out", f"a07_{STEM}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
