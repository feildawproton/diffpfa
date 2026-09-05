"""
a09_ground_mode_real_data.py -- GROUND image plane on a real collection: does the 'rotated dataset'
branch trigger, and is the written Row/Col metadata consistent with the image that was formed?

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a09_ground_mode_real_data.py [stem]

GROUND mode is low priority for the project (ATR consumes slant-plane imagery), so this only records
what happens.  Checks: (a) whether pfa_per_polar reports the swap; (b) Row/Col SS in the XML versus
the CPHD LineSpacing/SampleSpacing the processor was asked to use; (c) Row/Col ImpRespBW versus the
k-space extents recomputed along the written Row/Col unit vectors; (d) sarkit consistency.
"""
import os, sys, json, io, contextlib, subprocess
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import sarkit.sicd as sksicd
import sarkit.cphd as skcphd
from diffpfa.IFP import IFAProcessor
from diffpfa.IFA.kspace import compute_kspace

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = sys.argv[1] if len(sys.argv) > 1 else "2023-09-11-10-37-05_UMBRA-05"


def main():
    cphd = f"/home/feildaw/data/{STEM}_CPHD.cphd"
    outdir = os.path.join(HERE, "out", f"run_ground_{STEM}"); os.makedirs(outdir, exist_ok=True)
    buf = io.StringIO()
    proc = IFAProcessor(cphd_path=cphd, output_dir=outdir, image_plane="GROUND", batch_size=256, device="cuda")
    with torch.inference_mode(), contextlib.redirect_stdout(buf):
        files, *_ = proc.run()
    log = buf.getvalue()
    rotated = "Data is rotated" in log
    print(f"[run] {files[0]}\n      pfa_per_polar reported rotated dataset: {rotated}")
    with open(files[0], "rb") as f:
        r = sksicd.NitfReader(f); x = r.metadata.xmltree; h = sksicd.XmlHelper(x)
    with open(cphd, "rb") as f:
        rc = skcphd.Reader(f); hc = skcphd.XmlHelper(rc.metadata.xmltree)
        pv = rc.read_pvps(hc.load("./{*}Channel/{*}RefChId")); pvp = {n: np.ascontiguousarray(pv[n].astype(pv[n].dtype.newbyteorder("="))) for n in pv.dtype.names}
        ns = int(rc.metadata.xmltree.find(".//{*}Data/{*}Channel/{*}NumSamples").text)
        uIAX = hc.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"); uIAY = hc.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY")
        ls = hc.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}LineSpacing"); ss_c = hc.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}SampleSpacing")
    u_row = h.load("./{*}Grid/{*}Row/{*}UVectECF"); u_col = h.load("./{*}Grid/{*}Col/{*}UVectECF")
    nrows = h.load("./{*}ImageData/{*}NumRows"); ncols = h.load("./{*}ImageData/{*}NumCols")
    print(f"      Row vector is uIAX: {np.allclose(u_row, uIAX)} / uIAY: {np.allclose(u_row, uIAY)};  Col vector is uIAX: {np.allclose(u_col, uIAX)} / uIAY: {np.allclose(u_col, uIAY)}")
    print(f"      image {nrows} rows x {ncols} cols; XML Row/SS={h.load('./{*}Grid/{*}Row/{*}SS'):.6f} Col/SS={h.load('./{*}Grid/{*}Col/{*}SS'):.6f}; CPHD LineSpacing={ls:.6f} SampleSpacing={ss_c:.6f}")
    print(f"      implied extents: rows*SS={nrows*h.load('./{*}Grid/{*}Row/{*}SS'):.1f} m, cols*SS={ncols*h.load('./{*}Grid/{*}Col/{*}SS'):.1f} m (ImageArea is 4000 x 4000 m)")
    # k-space extents along the written axes
    Ku, Kr = compute_kspace(pvp, u_col, u_row, ns, "FX", device="cpu")   # Ku along Col, Kr along Row
    bw_row_true = float(Kr.max() - Kr.min()); bw_col_true = float(Ku.max() - Ku.min())
    print(f"      XML Row/ImpRespBW={h.load('./{*}Grid/{*}Row/{*}ImpRespBW'):.4f} vs k-extent along Row vector {bw_row_true:.4f};  Col/ImpRespBW={h.load('./{*}Grid/{*}Col/{*}ImpRespBW'):.4f} vs along Col vector {bw_col_true:.4f}")
    # pixel spectrum support along each axis (independent of XML)
    n = 1024; r0 = nrows // 2 - n // 2; c0 = ncols // 2 - n // 2
    with open(files[0], "rb") as f:
        chip, _ = sksicd.NitfReader(f).read_sub_image(start_row=r0, start_col=c0, stop_row=r0 + n, stop_col=c0 + n)
    for ax, d in ((0, "Row"), (1, "Col")):
        ssv = h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}SS")
        P = (np.abs(np.fft.fftshift(np.fft.fft(chip.astype(np.complex64), axis=ax), axes=ax)) ** 2).mean(axis=1 - ax)
        k = np.fft.fftshift(np.fft.fftfreq(n, d=ssv)); lvl = np.median(np.sort(P)[n // 2:]); idx = np.where(P > lvl * 0.1)[0]
        print(f"      pixel-spectrum support along {d} (using XML SS): {k[idx[-1]]-k[idx[0]]:.4f} cyc/m  (XML ImpRespBW {h.load(f'./{{*}}Grid/{{*}}{d}/{{*}}ImpRespBW'):.4f})")
    chk = subprocess.run([os.path.join(os.path.dirname(sys.executable), "sicdcheck"), files[0]], capture_output=True, text=True)
    errs = [l.strip() for l in chk.stdout.splitlines() if "[Error]" in l or "[Warning]" in l]
    print(f"      sicdcheck: {len(errs)} error/warning lines")
    for e in errs: print("        ", e)
    with open(os.path.join(HERE, "out", f"a09_{STEM}.json"), "w") as f:
        json.dump(dict(rotated=rotated, errs=errs), f, indent=1)


if __name__ == "__main__":
    main()
