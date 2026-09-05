"""
a08_metadata_fix_demo.py -- Drop-in reference for the SICD metadata diffpfa writes, verified by
re-running sarkit's SicdConsistency on the corrected XML.

Run from repo root (after a07 has produced a fresh product):
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a08_metadata_fix_demo.py [product.nitf] [cphd]

The function `corrected_metadata()` below is the deliverable: given the product's own Row/Col unit
vectors, its grid parameters, and the reference-channel PVPs, it computes every field that the audit
found inconsistent with NGA.STND.0024-1 and applies it to a copy of the XML.  Nothing under diffpfa/
is modified; the corrected XML is written next to this script and checked with `sicdcheck`.

Fields corrected (DIDD reference in brackets):
  Grid/Type = RGAZIM                                   [Table 3-4; 4.15.1 "The resulting PFA image is Grid Type = RGAZIM"]
  Grid/TimeCOAPoly(0,0) = SCP COA time                 [Table 3-4 note; 4.4.3]
  Grid/Row|Col/ImpRespWid = 0.886 / ImpRespBW          [4.14.6: Rg_IRW = k_RG / Krg_IRBW, k = 0.886 uniform]
  Grid/Row|Col/KCtr = centre of the k-space grid the IFFT was taken over   [Table 3-4: "zero frequency of the DFT"]
  Grid/Row|Col/WgtType/WindowName = UNIFORM
  Timeline/CollectDuration, ImageFormation/TStartProc,TEndProc measured from CollectStart
  Position/ARPPoly fitted against time since CollectStart (CPHD TxTime/RcvTime are already so referenced)
  SCPCOA/* recomputed by sarkit from the corrected ARPPoly and SCPTime (and cross-checked with
      diffpfa.sicd_geometry.compute_scp_geometry)
  PFA/PolarAngRefTime = the time at which the polar angle w.r.t. the product's Row axis is zero
  PFA/PolarAngPoly = polynomial in absolute time (since CollectStart)             [Table 3-15]
  PFA/SpatialFreqSFPoly = polynomial in polar angle, not time                      [Table 3-15]
  GeoData/ImageCorners = corner pixels projected to the SCP-height ground plane (sarkit image_to_ground_plane)
"""
import os, sys, json, glob, subprocess, copy
import numpy as np
import numpy.polynomial.polynomial as npp
import lxml.etree as ET

sys.path.insert(0, os.getcwd())
import sarkit.sicd as sksicd
import sarkit.cphd as skcphd
import sarkit.wgs84 as wgs84
from diffpfa.IFA.kspace import compute_kspace
from diffpfa.sicd_geometry import compute_scp_geometry
from diffpfa.constants import SPEED_OF_LIGHT as C

HERE = os.path.dirname(os.path.abspath(__file__))
K_UNIFORM = 0.8859          # DIDD 4.14.6 / 4.15 give 0.886 for uniform weighting; sarkit's SicdConsistency (KAPFAC) and
                            # the Umbra products use 0.8859 (= 2*0.44295, the exact half-power half-width of sinc^2)


def _set(root, path, text):
    el = root.find(path)
    assert el is not None, path
    el.text = text
    return el


def _poly1d(parent_el, coefs, ns):
    for c in list(parent_el):
        parent_el.remove(c)
    parent_el.set("order1", str(len(coefs) - 1))
    for k, v in enumerate(coefs):
        e = ET.SubElement(parent_el, f"{{{ns}}}Coef"); e.set("exponent1", str(k)); e.text = f"{v:.15e}"


def corrected_metadata(xmltree, pvp, deg=5):
    """Return a corrected copy of the SICD XML tree.  pvp: dict of reference-channel PVP arrays."""
    tree = copy.deepcopy(xmltree); root = tree.getroot(); ns = ET.QName(root).namespace
    h = sksicd.XmlHelper(tree)
    u_row = h.load("./{*}Grid/{*}Row/{*}UVectECF"); u_col = h.load("./{*}Grid/{*}Col/{*}UVectECF")
    scp = h.load("./{*}GeoData/{*}SCP/{*}ECF")
    nrows = h.load("./{*}ImageData/{*}NumRows"); ncols = h.load("./{*}ImageData/{*}NumCols")
    ss_row = h.load("./{*}Grid/{*}Row/{*}SS"); ss_col = h.load("./{*}Grid/{*}Col/{*}SS")
    bw_row = h.load("./{*}Grid/{*}Row/{*}ImpRespBW"); bw_col = h.load("./{*}Grid/{*}Col/{*}ImpRespBW")

    # ---- times (CPHD TxTime/RcvTime are relative to CollectionStart already)
    t_tx = np.asarray(pvp["TxTime"], float); t_rcv = np.asarray(pvp["RcvTime"], float)
    t_mid = 0.5 * (t_tx + t_rcv)
    arp_mid = 0.5 * (np.asarray(pvp["TxPos"], float) + np.asarray(pvp["RcvPos"], float))
    num_samples = int(pvp["__num_samples__"])

    # ---- k-space exactly as the processor mapped it (uIAX := Col, uIAY := Row)
    Ku, Kr = compute_kspace(pvp, u_col, u_row, num_samples, "FX", device="cpu")
    Ku = Ku.numpy(); Kr = Kr.numpy()
    ns_ = Ku.shape[1]
    Ku_mid, Kr_mid = Ku[:, ns_ // 2], Kr[:, ns_ // 2]
    plr = np.arctan2(Ku_mid, Kr_mid)                         # polar angle w.r.t. the Row axis (DIDD 4.15.2)
    plr_coef = npp.polyfit(t_mid, plr, deg)
    # reference time: zero of the polar angle nearest the aperture centre
    roots = npp.polyroots(plr_coef); roots = roots[np.isreal(roots)].real
    t_ref = float(roots[np.argmin(np.abs(roots - t_mid.mean()))])
    # spatial frequency scale factor as a function of polar angle
    F_mid = np.asarray(pvp["SC0"], float) + (ns_ // 2) * np.asarray(pvp["SCSS"], float)
    ksf = np.sqrt(Ku_mid ** 2 + Kr_mid ** 2) / (2 * F_mid / C)
    ksf_coef = npp.polyfit(plr, ksf, deg)
    # ARP polynomial in absolute time
    arp_coef = np.stack([npp.polyfit(t_mid, arp_mid[:, i], deg) for i in range(3)])

    # ---- Grid
    _set(root, "./{*}Grid/{*}Type", "RGAZIM")
    tc = root.find("./{*}Grid/{*}TimeCOAPoly"); tc.set("order1", "0"); tc.set("order2", "0")
    for c in list(tc): tc.remove(c)
    e = ET.SubElement(tc, f"{{{ns}}}Coef"); e.set("exponent1", "0"); e.set("exponent2", "0"); e.text = f"{t_ref:.12f}"
    kctr = {"Row": 0.5 * (Kr.min() + Kr.max()), "Col": 0.5 * (Ku.min() + Ku.max())}
    for d, bw in (("Row", bw_row), ("Col", bw_col)):
        g = root.find(f"./{{*}}Grid/{{*}}{d}")
        _set(g, "./{*}ImpRespWid", f"{K_UNIFORM / bw:.12f}")
        _set(g, "./{*}KCtr", f"{kctr[d]:.12f}")
        if g.find("./{*}WgtType") is None:
            w = ET.Element(f"{{{ns}}}WgtType"); n = ET.SubElement(w, f"{{{ns}}}WindowName"); n.text = "UNIFORM"
            anchor = g.find("./{*}DeltaKCOAPoly")
            anchor = anchor if anchor is not None else g.find("./{*}DeltaK2")
            anchor.addnext(w)

    # ---- Timeline / ImageFormation
    _set(root, "./{*}Timeline/{*}CollectDuration", f"{max(t_tx[-1], t_rcv[-1]):.9f}")
    _set(root, "./{*}ImageFormation/{*}TStartProc", f"{t_tx[0]:.9f}")
    _set(root, "./{*}ImageFormation/{*}TEndProc", f"{t_tx[-1]:.9f}")

    # ---- Position/ARPPoly
    for i, coord in enumerate("XYZ"):
        _poly1d(root.find(f"./{{*}}Position/{{*}}ARPPoly/{{*}}{coord}"), arp_coef[i], ns)

    # ---- SCPCOA: set SCPTime, then let sarkit derive the rest from ARPPoly (DIDD 5.x definitions)
    _set(root, "./{*}SCPCOA/{*}SCPTime", f"{t_ref:.12f}")
    new_scpcoa = sksicd.compute_scp_coa(tree)
    root.replace(root.find("./{*}SCPCOA"), new_scpcoa)

    # ---- PFA
    _set(root, "./{*}PFA/{*}PolarAngRefTime", f"{t_ref:.12f}")
    _poly1d(root.find("./{*}PFA/{*}PolarAngPoly"), plr_coef, ns)
    _poly1d(root.find("./{*}PFA/{*}SpatialFreqSFPoly"), ksf_coef, ns)
    _set(root, "./{*}PFA/{*}Krg1", f"{Kr.min():.12f}"); _set(root, "./{*}PFA/{*}Krg2", f"{Kr.max():.12f}")
    _set(root, "./{*}PFA/{*}Kaz1", f"{Ku.min():.12f}"); _set(root, "./{*}PFA/{*}Kaz2", f"{Ku.max():.12f}")
    # IPN = uRow x uCol (DIDD 4.15.1: uRG = -uIPX, uAZ = -uIPY, uIPZ = uIPX x uIPY)
    ipn = np.cross(u_row, u_col); ipn /= np.linalg.norm(ipn)
    for i, coord in enumerate("XYZ"):
        _set(root, f"./{{*}}PFA/{{*}}IPN/{{*}}{coord}", f"{ipn[i]:.12f}")

    # ---- ImageCorners: project corner pixels to the SCP-height ground plane
    h2 = sksicd.XmlHelper(tree)
    scp_row = h2.load("./{*}ImageData/{*}SCPPixel/{*}Row"); scp_col = h2.load("./{*}ImageData/{*}SCPPixel/{*}Col")
    corners_rc = np.array([[0, 0], [0, ncols - 1], [nrows - 1, ncols - 1], [nrows - 1, 0]], float)
    xrow_ycol = np.stack([(corners_rc[:, 0] - scp_row) * ss_row, (corners_rc[:, 1] - scp_col) * ss_col], 1)
    up = wgs84.up(wgs84.cartesian_to_geodetic(scp))
    gpp, delta, ok = sksicd.image_to_ground_plane(tree, xrow_ycol, scp, up)
    llh = wgs84.cartesian_to_geodetic(gpp)
    icps = root.findall("./{*}GeoData/{*}ImageCorners/{*}ICP")
    for icp, (lat, lon, _) in zip(icps, llh):
        _set(icp, "./{*}Lat", f"{lat:.9f}"); _set(icp, "./{*}Lon", f"{lon:.9f}")

    diag = dict(t_ref=t_ref, plr_at_tref=float(npp.polyval(t_ref, plr_coef)), kctr=kctr, ksf_coef=ksf_coef.tolist(), icp_success=bool(ok),
                icp_delta_max_m=float(np.max(delta)), arp_fit_resid_mm=float(1e3 * np.max(np.linalg.norm(np.stack([npp.polyval(t_mid, arp_coef[i]) for i in range(3)], 1) - arp_mid, axis=1))))
    return tree, diag


def sicdcheck(path):
    p = subprocess.run([os.path.join(os.path.dirname(sys.executable), "sicdcheck"), path], capture_output=True, text=True)
    return [l.strip() for l in p.stdout.splitlines() if "[Error]" in l or "[Warning]" in l], p.stdout


def main():
    prods = sorted(glob.glob(os.path.join(HERE, "out", "run_*", "*.nitf")))
    prod = sys.argv[1] if len(sys.argv) > 1 else prods[0]
    stem = os.path.basename(prod).split("_SICDU")[0]
    cphd = sys.argv[2] if len(sys.argv) > 2 else f"/home/feildaw/data/{stem}_CPHD.cphd"
    print(f"product: {prod}\ncphd:    {cphd}")

    with open(prod, "rb") as f:
        xmltree = sksicd.NitfReader(f).metadata.xmltree
    with open(cphd, "rb") as f:
        r = skcphd.Reader(f); hc = skcphd.XmlHelper(r.metadata.xmltree)
        ref_id = hc.load("./{*}Channel/{*}RefChId"); pv = r.read_pvps(ref_id)
        pvp = {n: np.ascontiguousarray(pv[n].astype(pv[n].dtype.newbyteorder("="))) for n in pv.dtype.names}
        ch = [c for c in r.metadata.xmltree.findall(".//{*}Data/{*}Channel") if c.find("./{*}Identifier").text == ref_id][0]
        pvp["__num_samples__"] = int(ch.find("./{*}NumSamples").text)
        t_ref_cphd = hc.load("./{*}ReferenceGeometry/{*}ReferenceTime")
        arp_ref_cphd = hc.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPPos")

    before, _ = sicdcheck(prod)
    fixed, diag = corrected_metadata(xmltree, pvp)
    out_xml = os.path.join(HERE, "out", f"{stem}_corrected.xml")
    fixed.write(out_xml, pretty_print=True, xml_declaration=True, encoding="utf-8")
    after, full = sicdcheck(out_xml)
    with open(os.path.join(HERE, "out", f"{stem}_corrected_sicdcheck.txt"), "w") as f:
        f.write(full)

    # schema
    xsd = os.path.join("schemas", "SICD_schema_V1.3.0_2021_11_30.xsd")
    schema = ET.XMLSchema(ET.parse(xsd))
    ok = schema.validate(fixed)
    print(f"\nschema valid after correction: {ok}" + ("" if ok else f"\n{schema.error_log}"))

    print(f"\nsicdcheck before: {len(before)} error/warning lines")
    for l in before: print("   ", l)
    print(f"sicdcheck after:  {len(after)} error/warning lines")
    for l in after: print("   ", l)

    h = sksicd.XmlHelper(fixed)
    print("\ndiagnostics:")
    print(f"   t_ref (polar angle zero) = {diag['t_ref']:.6f} s ; CPHD ReferenceTime = {t_ref_cphd:.6f} s ; diff = {(diag['t_ref']-t_ref_cphd)*1e3:.3f} ms")
    print(f"   PolarAngPoly(t_ref) = {diag['plr_at_tref']:.2e} rad ; ARPPoly fit residual = {diag['arp_fit_resid_mm']:.3f} mm")
    arp_new = h.load("./{*}SCPCOA/{*}ARPPos")
    print(f"   SCPCOA/ARPPos (sarkit from ARPPoly) vs CPHD ReferenceGeometry/ARPPos: {np.linalg.norm(arp_new-arp_ref_cphd):.3f} m")
    geom = compute_scp_geometry(h.load("./{*}GeoData/{*}SCP/{*}ECF"), arp_new, h.load("./{*}SCPCOA/{*}ARPVel"), side_of_track=h.load("./{*}SCPCOA/{*}SideOfTrack"))
    for k in ["SlantRange", "GroundRange", "DopplerConeAng", "GrazeAng", "IncidenceAng", "TwistAng", "SlopeAng", "AzimAng", "LayoverAng"]:
        print(f"   SCPCOA/{k:15s} sarkit={h.load(f'./{{*}}SCPCOA/{{*}}{k}'):14.6f}  diffpfa.compute_scp_geometry={geom[k]:14.6f}")
    print(f"   Row: KCtr={h.load('./{*}Grid/{*}Row/{*}KCtr'):.4f} ImpRespWid={h.load('./{*}Grid/{*}Row/{*}ImpRespWid'):.4f}  Col: KCtr={h.load('./{*}Grid/{*}Col/{*}KCtr'):.4f} ImpRespWid={h.load('./{*}Grid/{*}Col/{*}ImpRespWid'):.4f}")
    print(f"   SpatialFreqSFPoly = {np.array(diag['ksf_coef'])}")
    print(f"   ImageCorners projection success={diag['icp_success']} max residual {diag['icp_delta_max_m']:.3f} m")
    for icp in fixed.getroot().findall("./{*}GeoData/{*}ImageCorners/{*}ICP"):
        print(f"      {icp.get('index')}: {icp.find('./{*}Lat').text}, {icp.find('./{*}Lon').text}")
    with open(os.path.join(HERE, "out", f"a08_{stem}.json"), "w") as f:
        json.dump(dict(before=before, after=after, diag=diag, schema_valid=bool(ok)), f, indent=1, default=str)


if __name__ == "__main__":
    main()
