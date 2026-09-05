"""
a12_gold_pair_registration.py -- Are diffpfa's pixels where the vendor's pixels are?  (gold pair only)

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a12_gold_pair_registration.py

Both products of 2025-10-26 UMBRA-08 are slant-plane images with identical Row/Col unit vectors
(a10: 0.0000 deg apart) and the same SCP, so they differ only in sample spacing and extent.  For a
set of chips at known SCP-relative image coordinates, resample the diffpfa magnitude onto the
vendor grid (bilinear, about the SCP pixel) and cross-correlate the two magnitude images.  The
correlation-peak offset is the geometric registration error in metres; a non-zero offset that grows
with distance from the SCP would indicate a sample-spacing (scale) error; a constant offset a
SCPPixel/origin error.  Magnitude correlation also gives a coarse "how alike" number.
No metadata from the XML is trusted beyond SS and SCPPixel.  Writes a side-by-side PNG if matplotlib
is available.
"""
import os, sys, json
import numpy as np
import sarkit.sicd as ss
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
VEND = "/home/feildaw/data/2025-10-26-05-00-15_UMBRA-08_SICD.nitf"
PROD = "/home/feildaw/diffpfa/workspace/output/2025-10-26-05-00-15_UMBRA-08_SICDU_V_V.nitf"


def info(path):
    with open(path, "rb") as f:
        r = ss.NitfReader(f); h = ss.XmlHelper(r.metadata.xmltree)
    return dict(ss_r=h.load("./{*}Grid/{*}Row/{*}SS"), ss_c=h.load("./{*}Grid/{*}Col/{*}SS"), scp_r=h.load("./{*}ImageData/{*}SCPPixel/{*}Row"),
                scp_c=h.load("./{*}ImageData/{*}SCPPixel/{*}Col"), nr=h.load("./{*}ImageData/{*}NumRows"), nc=h.load("./{*}ImageData/{*}NumCols"),
                urow=h.load("./{*}Grid/{*}Row/{*}UVectECF"), ucol=h.load("./{*}Grid/{*}Col/{*}UVectECF"), scp=h.load("./{*}GeoData/{*}SCP/{*}ECF"))


def read_chip(path, r0, c0, n_r, n_c):
    with open(path, "rb") as f:
        chip, _ = ss.NitfReader(f).read_sub_image(start_row=r0, start_col=c0, stop_row=r0 + n_r, stop_col=c0 + n_c)
    return np.abs(chip.astype(np.complex64))


def main():
    V, P = info(VEND), info(PROD)
    print("vendor :", {k: V[k] for k in ("ss_r", "ss_c", "scp_r", "scp_c", "nr", "nc")})
    print("diffpfa:", {k: P[k] for k in ("ss_r", "ss_c", "scp_r", "scp_c", "nr", "nc")})
    print("angle between Row vectors (deg):", np.degrees(np.arccos(np.clip(np.dot(V["urow"], P["urow"]), -1, 1))), " Col:", np.degrees(np.arccos(np.clip(np.dot(V["ucol"], P["ucol"]), -1, 1))),
          " |SCP diff| m:", np.linalg.norm(V["scp"] - P["scp"]))
    n = 768                                  # vendor-pixel chip size (~490 m x 683 m)
    results = []
    panels = None
    for (xrow_c, ycol_c) in [(0.0, 0.0), (1500.0, 0.0), (-1500.0, 0.0), (0.0, 2000.0), (0.0, -2000.0), (1200.0, 1800.0), (-1200.0, -1800.0)]:
        # vendor chip centred at SCP-relative (xrow, ycol) metres
        vr0 = int(round(V["scp_r"] + xrow_c / V["ss_r"] - n / 2)); vc0 = int(round(V["scp_c"] + ycol_c / V["ss_c"] - n / 2))
        if vr0 < 0 or vc0 < 0 or vr0 + n > V["nr"] or vc0 + n > V["nc"]:
            continue
        v = read_chip(VEND, vr0, vc0, n, n)
        # metres of every vendor pixel in the chip
        xr = (vr0 + np.arange(n) - V["scp_r"]) * V["ss_r"]; yc = (vc0 + np.arange(n) - V["scp_c"]) * V["ss_c"]
        # corresponding diffpfa pixel (fractional) coordinates
        pr = P["scp_r"] + xr / P["ss_r"]; pc = P["scp_c"] + yc / P["ss_c"]
        r0 = int(np.floor(pr.min())) - 2; c0 = int(np.floor(pc.min())) - 2
        r1 = int(np.ceil(pr.max())) + 3; c1 = int(np.ceil(pc.max())) + 3
        if r0 < 0 or c0 < 0 or r1 > P["nr"] or c1 > P["nc"]:
            continue
        p_full = read_chip(PROD, r0, c0, r1 - r0, c1 - c0)
        RR, CC = np.meshgrid(pr - r0, pc - c0, indexing="ij")
        p = map_coordinates(p_full, [RR, CC], order=1, mode="nearest")
        # normalised cross-correlation of log-magnitudes (robust to the different dynamic range)
        a = np.log10(v + 1e-3); b = np.log10(p + 1e-3)
        a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
        # zero-mean windowed correlation via FFT, search +-40 vendor pixels
        w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
        xc = fftconvolve(a * w, (b * w)[::-1, ::-1], mode="same") / (w ** 2).sum()
        c = np.array(xc.shape) // 2; s = 40
        sub = xc[c[0] - s:c[0] + s + 1, c[1] - s:c[1] + s + 1]
        pk = np.unravel_index(np.argmax(sub), sub.shape)
        dr_pix, dc_pix = pk[0] - s, pk[1] - s
        # sub-pixel refinement (parabolic)
        def refine(m, i):
            if 0 < i < len(m) - 1:
                d = (m[i - 1] - m[i + 1]) / (2 * (m[i - 1] - 2 * m[i] + m[i + 1]) + 1e-12); return i + d
            return i
        dr_pix = refine(sub[:, pk[1]], pk[0]) - s; dc_pix = refine(sub[pk[0], :], pk[1]) - s
        ncc_peak = float(sub.max()); ncc0 = float(xc[c[0], c[1]])
        res = dict(xrow_m=xrow_c, ycol_m=ycol_c, offset_row_m=float(dr_pix * V["ss_r"]), offset_col_m=float(dc_pix * V["ss_c"]), ncc_peak=ncc_peak, ncc_at_zero=ncc0)
        results.append(res)
        print(f"chip at (xrow={xrow_c:+6.0f}, ycol={ycol_c:+6.0f}) m: registration offset diffpfa vs vendor = ({res['offset_row_m']:+.2f}, {res['offset_col_m']:+.2f}) m ; "
              f"log-magnitude NCC peak {ncc_peak:.3f} (at zero shift {ncc0:.3f})")
        if panels is None:
            panels = (v, p)
    with open(os.path.join(HERE, "out", "a12_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sys.path.insert(0, os.getcwd()); from tools.convert2png import density_remap_8bit
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for a_, im, t in zip(ax, panels, ("Umbra SAR Processor 4.25.1 (gold)", "diffpfa (resampled to vendor grid)")):
            a_.imshow(density_remap_8bit(im), cmap="gray", vmin=0, vmax=255); a_.set_title(t); a_.set_xlabel("col (azimuth)"); a_.set_ylabel("row (range)")
        fig.suptitle("2025-10-26 UMBRA-08, 768x768 vendor-pixel chip at the SCP")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "out", "a12_scp_chip_vendor_vs_diffpfa.png"), dpi=110)
        print("wrote", os.path.join(HERE, "out", "a12_scp_chip_vendor_vs_diffpfa.png"))
    except Exception as e:
        print("no figure:", e)


if __name__ == "__main__":
    main()
