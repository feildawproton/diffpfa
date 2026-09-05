"""
a15_cross_check_other_audit.py -- Independent re-derivation of the checkable claims in
audit/agy_with_gemini_3.8_flash/DIFFPFA_INDEPENDENT_AUDIT_REPORT.md that my own audit did not cover.

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a15_cross_check_other_audit.py

(a) Their Finding 2: vendor image extent = 1.20 x (CPHD ground ImageArea projected onto the slant
    axes).  Recomputed for all six collections using the VENDOR's own Row/Col vectors, and tested
    against the alternative "fixed additive margin" hypothesis.
(b) Their Finding 6: SigmaZero/BetaZero = cos(SlopeAng) and (their remedy) GammaZero = BetaZero*sin(SlopeAng).
    Checked on every vendor file that carries a Radiometric block, against several candidate ratios.
(c) Their Finding 5 remedy: ICP = geodetic of the slant-plane corner point SCP + dr*uRow + du*uCol.
    Applied to the VENDOR's own grid and compared with the vendor's ICPs, and with sarkit's
    R/Rdot ground-plane projection.
(d) Their 'verified' claim that SCPCOA angles match the gold product to < 0.05 deg.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.getcwd())
import sarkit.sicd as ss
import sarkit.cphd as sc
import sarkit.wgs84 as w

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/feildaw/data"; OUT = "/home/feildaw/diffpfa/workspace/output"
res = {}

print("(a) vendor extent vs CPHD ImageArea projected onto the vendor's slant axes")
print(f"    {'collection':32s} | {'IA size':>7s} | {'proj row':>8s} {'vendor row':>10s} {'ratio':>6s} {'add/side':>8s} | {'proj col':>8s} {'vendor col':>10s} {'ratio':>6s} {'add/side':>8s}")
for cphd in sorted(glob.glob(os.path.join(DATA, "*_CPHD.cphd"))):
    stem = os.path.basename(cphd).replace("_CPHD.cphd", ""); vend = os.path.join(DATA, f"{stem}_SICD.nitf")
    with open(cphd, "rb") as f:
        hc = sc.XmlHelper(sc.Reader(f).metadata.xmltree)
        x1y1 = hc.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1"); x2y2 = hc.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
        uIAX = hc.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"); uIAY = hc.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY")
    with open(vend, "rb") as f:
        hv = ss.XmlHelper(ss.NitfReader(f).metadata.xmltree)
    urow = hv.load("./{*}Grid/{*}Row/{*}UVectECF"); ucol = hv.load("./{*}Grid/{*}Col/{*}UVectECF")
    nr = hv.load("./{*}ImageData/{*}NumRows"); nc = hv.load("./{*}ImageData/{*}NumCols"); ssr = hv.load("./{*}Grid/{*}Row/{*}SS"); ssc = hv.load("./{*}Grid/{*}Col/{*}SS")
    corners = [x * uIAX + y * uIAY for x in (x1y1[0], x2y2[0]) for y in (x1y1[1], x2y2[1])]
    r = [np.dot(c, urow) for c in corners]; u = [np.dot(c, ucol) for c in corners]
    pr, pu = max(r) - min(r), max(u) - min(u); vr, vc = nr * ssr, nc * ssc
    ia = x2y2[0] - x1y1[0]
    print(f"    {stem:32s} | {ia:7.0f} | {pr:8.1f} {vr:10.1f} {vr/pr:6.3f} {(vr-pr)/2:8.1f} | {pu:8.1f} {vc:10.1f} {vc/pu:6.3f} {(vc-pu)/2:8.1f}")
    res.setdefault("a", {})[stem] = dict(ia=ia, proj_row=pr, vendor_row=vr, ratio_row=vr / pr, proj_col=pu, vendor_col=vc, ratio_col=vc / pu)

print("\n(b) vendor radiometric scale-factor ratios (constant terms)")
for vend in sorted(glob.glob(os.path.join(DATA, "*_SICD.nitf"))):
    with open(vend, "rb") as f:
        hv = ss.XmlHelper(ss.NitfReader(f).metadata.xmltree)
    if hv.load("./{*}Radiometric/{*}BetaZeroSFPoly") is None:
        print(f"    {os.path.basename(vend):45s} no Radiometric block"); continue
    b = hv.load("./{*}Radiometric/{*}BetaZeroSFPoly")[0, 0]; s = hv.load("./{*}Radiometric/{*}SigmaZeroSFPoly")[0, 0]; g = hv.load("./{*}Radiometric/{*}GammaZeroSFPoly")[0, 0]
    gz = np.radians(hv.load("./{*}SCPCOA/{*}GrazeAng")); sl = np.radians(hv.load("./{*}SCPCOA/{*}SlopeAng")); inc = np.radians(hv.load("./{*}SCPCOA/{*}IncidenceAng"))
    print(f"    {os.path.basename(vend):45s} sigma/beta={s/b:.6f}  cos(slope)={np.cos(sl):.6f} cos(graze)={np.cos(gz):.6f} | gamma/beta={g/b:.6f}  sin(slope)={np.sin(sl):.6f} sin(graze)={np.sin(gz):.6f} cos(slope)/sin(graze)={np.cos(sl)/np.sin(gz):.6f} | gamma/sigma={g/s:.6f} 1/cos(inc)={1/np.cos(inc):.6f}")
    res.setdefault("b", {})[os.path.basename(vend)] = dict(sigma_over_beta=s / b, cos_slope=np.cos(sl), gamma_over_beta=g / b, sin_slope=np.sin(sl), cos_slope_over_sin_graze=np.cos(sl) / np.sin(gz), gamma_over_sigma=g / s, inv_cos_inc=1 / np.cos(inc))

print("\n(c) ICP remedies applied to the VENDOR gold grid (2025-10-26) vs the vendor's own ICPs")
vend = os.path.join(DATA, "2025-10-26-05-00-15_UMBRA-08_SICD.nitf")
with open(vend, "rb") as f:
    tree = ss.NitfReader(f).metadata.xmltree
hv = ss.XmlHelper(tree)
scp = hv.load("./{*}GeoData/{*}SCP/{*}ECF"); urow = hv.load("./{*}Grid/{*}Row/{*}UVectECF"); ucol = hv.load("./{*}Grid/{*}Col/{*}UVectECF")
nr = hv.load("./{*}ImageData/{*}NumRows"); nc = hv.load("./{*}ImageData/{*}NumCols"); ssr = hv.load("./{*}Grid/{*}Row/{*}SS"); ssc = hv.load("./{*}Grid/{*}Col/{*}SS")
sr = hv.load("./{*}ImageData/{*}SCPPixel/{*}Row"); scc = hv.load("./{*}ImageData/{*}SCPPixel/{*}Col")
icp_v = hv.load("./{*}GeoData/{*}ImageCorners")
hae = w.cartesian_to_geodetic(scp)[2]
corners_rc = np.array([[0, 0], [0, nc - 1], [nr - 1, nc - 1], [nr - 1, 0]], float)
xr = (corners_rc[:, 0] - sr) * ssr; yc = (corners_rc[:, 1] - scc) * ssc
# their remedy: slant-plane point
slant_pts = scp[None, :] + xr[:, None] * urow[None, :] + yc[:, None] * ucol[None, :]
llh_slant = w.cartesian_to_geodetic(slant_pts)
# sarkit R/Rdot ground-plane projection
gpp, _, ok = ss.image_to_ground_plane(tree, np.stack([xr, yc], 1), scp, w.up(w.cartesian_to_geodetic(scp)))
llh_gnd = w.cartesian_to_geodetic(gpp)
vend_ecf = w.geodetic_to_cartesian(np.concatenate([icp_v, np.full((4, 1), hae)], 1))
for i in range(4):
    d_slant = np.linalg.norm(w.geodetic_to_cartesian([llh_slant[i, 0], llh_slant[i, 1], hae]) - vend_ecf[i])
    d_gnd = np.linalg.norm(gpp[i] - vend_ecf[i])
    print(f"    ICP{i+1}: slant-plane-point remedy vs vendor = {d_slant:8.1f} m (point sits {llh_slant[i,2]-hae:+8.1f} m above SCP height) | sarkit ground projection vs vendor = {d_gnd:6.1f} m")
    res.setdefault("c", []).append(dict(icp=i + 1, slant_remedy_err_m=float(d_slant), slant_point_height_m=float(llh_slant[i, 2] - hae), ground_proj_err_m=float(d_gnd)))

print("\n(d) SCPCOA angles: diffpfa 2025 product vs vendor gold")
prod = os.path.join(OUT, "2025-10-26-05-00-15_UMBRA-08_SICDU_V_V.nitf")
with open(prod, "rb") as f:
    hp = ss.XmlHelper(ss.NitfReader(f).metadata.xmltree)
for k in ["SCPTime", "DopplerConeAng", "GrazeAng", "IncidenceAng", "TwistAng", "SlopeAng", "AzimAng", "LayoverAng", "SlantRange", "GroundRange"]:
    a = hp.load(f"./{{*}}SCPCOA/{{*}}{k}"); b = hv.load(f"./{{*}}SCPCOA/{{*}}{k}")
    print(f"    {k:15s} diffpfa={a:14.6f} vendor={b:14.6f} diff={a-b:+.6f}")
    res.setdefault("d", {})[k] = float(a - b)
json.dump(res, open(os.path.join(HERE, "out", "a15_results.json"), "w"), indent=1, default=str)
