"""
a10_geometry_and_focus.py -- Geometry formulas against independent references, ARP polynomial
fidelity, and the phase error implied by diffpfa's slant-plane polar mapping for ground scatterers
at the image-area corners, all computed from real CPHD PVPs (no image formation).

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a10_geometry_and_focus.py

(1) compute_scp_geometry(SRP, ARP_ref, VARP_ref) vs the CPHD's own ReferenceGeometry angles for all
    six collections (the CPHD producer computed these independently; the vendor SICD copies them).
(2) diffpfa's slant-plane Row/Col unit vectors vs the vendor's, all six collections (includes one
    left-looking collection, 2023-07-30).
(3) fit_arp_poly: degree-5 fit residual on the real 8229-vector trajectory; and the time-origin issue
    (ARPPoly(SCPTime) vs the ARP at the same absolute time from the PVPs).
(4) PFA planar-wavefront / slant-plane mapping error for ground-plane scatterers at the ImageArea
    corners: exact differential range from PVPs vs the K.x model diffpfa inverts; residual phase
    after removing the linear (position) part, in cycles at the carrier.  This is the space-variant
    defocus a consumer will see at the image edges; it is inherent to PFA but its size depends on
    the focus-plane choice (diffpfa: FPN = IPN = slant; vendor: FPN = geodetic up).
"""
import os, sys, json, glob
import numpy as np
import numpy.polynomial.polynomial as npp

sys.path.insert(0, os.getcwd())
import sarkit.cphd as skcphd
import sarkit.sicd as sksicd
from diffpfa.sicd_geometry import compute_scp_geometry, fit_arp_poly, get_geodetic_up_vector
from diffpfa.constants import SPEED_OF_LIGHT as C

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/feildaw/data"
OUT = "/home/feildaw/diffpfa/workspace/output"


def load(h, q):
    try:
        return h.load(q)
    except Exception:
        return None


def main():
    res = {}
    print("[1] compute_scp_geometry vs CPHD ReferenceGeometry (degrees / metres)")
    print(f"    {'collection':32s} {'SoT':>3s} | {'dGraze':>8s} {'dTwist':>8s} {'dSlope':>8s} {'dAzim':>8s} {'dLayover':>8s} {'dDCA':>8s} {'dInc':>8s} | {'dSlantRng m':>11s}")
    for cphd in sorted(glob.glob(os.path.join(DATA, "*_CPHD.cphd"))):
        stem = os.path.basename(cphd).replace("_CPHD.cphd", "")
        with open(cphd, "rb") as f:
            r = skcphd.Reader(f); x = r.metadata.xmltree; h = skcphd.XmlHelper(x)
            srp = h.load("./{*}ReferenceGeometry/{*}SRP/{*}ECF")
            arp = h.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPPos"); vel = h.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPVel")
            sot = h.load("./{*}ReferenceGeometry/{*}Monostatic/{*}SideOfTrack")
            ref = {k: h.load(f"./{{*}}ReferenceGeometry/{{*}}Monostatic/{{*}}{k}") for k in
                   ["GrazeAngle", "TwistAngle", "SlopeAngle", "AzimuthAngle", "LayoverAngle", "DopplerConeAngle", "IncidenceAngle", "SlantRange"]}
            tref = h.load("./{*}ReferenceGeometry/{*}ReferenceTime")
            pv = r.read_pvps(h.load("./{*}Channel/{*}RefChId"))
        g = compute_scp_geometry(srp, arp, vel, side_of_track=sot, image_plane="SLANT")
        d = {k: g[a] - ref[b] for k, a, b in [("dGraze", "GrazeAng", "GrazeAngle"), ("dTwist", "TwistAng", "TwistAngle"), ("dSlope", "SlopeAng", "SlopeAngle"),
                                              ("dAzim", "AzimAng", "AzimuthAngle"), ("dLayover", "LayoverAng", "LayoverAngle"), ("dDCA", "DopplerConeAng", "DopplerConeAngle"),
                                              ("dInc", "IncidenceAng", "IncidenceAngle"), ("dSlantRng", "SlantRange", "SlantRange")]}
        print(f"    {stem:32s} {sot:>3s} | {d['dGraze']:8.5f} {d['dTwist']:8.5f} {d['dSlope']:8.5f} {d['dAzim']:8.5f} {d['dLayover']:8.5f} {d['dDCA']:8.5f} {d['dInc']:8.5f} | {d['dSlantRng']:11.4f}")
        res[stem] = {"geom_diff_vs_cphd": d}

        # ---- (2) slant-plane axes vs vendor
        vend = os.path.join(DATA, f"{stem}_SICD.nitf"); prod = glob.glob(os.path.join(OUT, f"{stem}_SICDU_*.nitf"))
        if os.path.exists(vend) and prod:
            with open(vend, "rb") as f:
                hv = sksicd.XmlHelper(sksicd.NitfReader(f).metadata.xmltree)
            with open(prod[0], "rb") as f:
                hp = sksicd.XmlHelper(sksicd.NitfReader(f).metadata.xmltree)
            rows = (hv.load("./{*}Grid/{*}Row/{*}UVectECF"), hp.load("./{*}Grid/{*}Row/{*}UVectECF"))
            cols = (hv.load("./{*}Grid/{*}Col/{*}UVectECF"), hp.load("./{*}Grid/{*}Col/{*}UVectECF"))
            ang = lambda a, b: np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
            res[stem]["axes_vs_vendor_deg"] = dict(row=float(ang(*rows)), col=float(ang(*cols)))
            # also: does the vendor's uRow x uCol point away from earth (IPN convention)?
            ipn_v = np.cross(rows[0], cols[0]); ipn_p = np.cross(rows[1], cols[1]); up = get_geodetic_up_vector(srp)
            res[stem]["ipn_dot_up"] = dict(vendor=float(np.dot(ipn_v, up)), diffpfa=float(np.dot(ipn_p, up)))
            print(f"        axes vs vendor: angle(uRow)={ang(*rows):.4f} deg angle(uCol)={ang(*cols):.4f} deg | (uRow x uCol).up: vendor {np.dot(ipn_v, up):+.3f} diffpfa {np.dot(ipn_p, up):+.3f}")

        # ---- (3) ARP poly fidelity and time origin (reference channel PVPs)
        pvp = {n: np.ascontiguousarray(pv[n]) for n in pv.dtype.names}
        info = fit_arp_poly(pvp)
        t_rel = pvp["TxTime"] - pvp["TxTime"][0]
        arp_mid = 0.5 * (pvp["TxPos"] + pvp["RcvPos"])
        poly = np.vstack([info["ARPPoly"]["X"], info["ARPPoly"]["Y"], info["ARPPoly"]["Z"]])
        fit = np.stack([npp.polyval(t_rel, poly[i]) for i in range(3)], 1)
        resid = np.linalg.norm(fit - arp_mid, axis=1)
        # evaluate the written polynomial at SCPTime as a SICD consumer would (t measured from CollectStart)
        t_scp_written = info["SCPTime"]
        arp_consumer = np.array([npp.polyval(t_scp_written, poly[i]) for i in range(3)])
        # truth at that absolute time: interpolate PVPs at TxTime = t_scp_written
        truth = np.array([np.interp(t_scp_written, pvp["TxTime"], arp_mid[:, i]) for i in range(3)])
        res[stem]["arp_fit_resid_max_m"] = float(resid.max())
        res[stem]["arp_time_origin_error_m"] = float(np.linalg.norm(arp_consumer - truth))
        res[stem]["txtime0"] = float(pvp["TxTime"][0])
        print(f"        ARPPoly deg-5 fit residual max = {resid.max()*1e3:.3f} mm; TxTime[0] = {pvp['TxTime'][0]*1e3:.3f} ms; written SCPTime = {t_scp_written:.6f}; "
              f"consumer ARP(SCPTime) vs truth at that absolute time: {np.linalg.norm(arp_consumer-truth):.2f} m")

        # ---- (4) mapping error for ground scatterers at the ImageArea corners
        uIAX = h.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"); uIAY = h.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY")
        x1y1 = h.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1"); x2y2 = h.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
        # diffpfa slant axes exactly as IFP.run() builds them
        p_vec = srp - arp; u_row = p_vec / np.linalg.norm(p_vec)
        u_v = vel / np.linalg.norm(vel); u_col = u_v - np.dot(u_v, u_row) * u_row; u_col /= np.linalg.norm(u_col)
        if sot == "L":
            u_col = -u_col
        P = srp[None, :] - arp_mid                       # (N,3) look vectors
        Pn = np.linalg.norm(P, axis=1)
        cos_t = P @ u_col / Pn; sin_t = P @ u_row / Pn
        fc = 0.5 * (h.load("./{*}Global/{*}FxBand/{*}FxMin") + h.load("./{*}Global/{*}FxBand/{*}FxMax"))
        k = 2 * fc / C
        worst = 0.0; table = []
        for (gx, gy) in [(x1y1[0], x1y1[1]), (x1y1[0], x2y2[1]), (x2y2[0], x2y2[1]), (x2y2[0], x1y1[1]), (x2y2[0], 0.0), (0.0, x2y2[1])]:
            xt = srp + gx * uIAX + gy * uIAY                            # ground-plane scatterer
            dR_exact = np.linalg.norm(xt[None, :] - arp_mid, axis=1) - Pn   # true differential range (monostatic, APC midpoint)
            # diffpfa's model: signal phase ~ K.(u,r) with (u,r) the in-plane coords of the scatterer -> the image
            # places the peak where sum_n exp(j2pi k [dR_exact(n) - (u cos_t + r sin_t)]) is maximal; the residual
            # after the best linear fit in (cos_t, sin_t) is the defocus phase.
            A = np.stack([cos_t, sin_t, np.ones_like(cos_t)], 1)
            coef, *_ = np.linalg.lstsq(A, dR_exact, rcond=None)
            resid_m = dR_exact - A @ coef
            cyc = k * (resid_m.max() - resid_m.min())
            worst = max(worst, cyc)
            table.append(dict(gx=gx, gy=gy, fitted_u=float(coef[0]), fitted_r=float(coef[1]), pp_resid_cycles=float(cyc), pp_resid_mm=float(1e3 * (resid_m.max() - resid_m.min()))))
        res[stem]["corner_defocus"] = table
        print(f"        ground-corner defocus (peak-to-peak residual phase after linear fit): worst {worst:.3f} cycles at fc; per corner: " +
              ", ".join(f"({t['gx']:+.0f},{t['gy']:+.0f}): {t['pp_resid_cycles']:.3f}" for t in table))
    with open(os.path.join(HERE, "out", "a10_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
