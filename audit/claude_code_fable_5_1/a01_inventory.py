"""
a01_inventory.py -- Inventory of every CPHD / vendor-SICD pair in /home/feildaw/data
and every diffpfa product in workspace/output.

Run from the repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a01_inventory.py

Purpose: ground every later measurement in what the real data actually contains
(domain type, PVP fields present, SIGNAL flags, vendor grid/IPR/weighting metadata).
Read-only; writes a JSON summary next to this script.
"""
import glob, json, os, sys
import numpy as np
import sarkit.cphd as skcphd
import sarkit.sicd as sksicd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/feildaw/data"
OUT = "/home/feildaw/diffpfa/workspace/output"


def L(h, q):
    try:
        v = h.load(q)
    except Exception as e:
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def cphd_summary(path):
    with open(path, "rb") as f:
        r = skcphd.Reader(f)
        x = r.metadata.xmltree
        h = skcphd.XmlHelper(x)
        s = {"file": os.path.basename(path), "size_GB": os.path.getsize(path) / 1e9}
        for q in [
            "./{*}CollectionID/{*}CollectorName", "./{*}CollectionID/{*}RadarMode/{*}ModeType",
            "./{*}Global/{*}DomainType", "./{*}Global/{*}SGN",
            "./{*}Global/{*}FxBand/{*}FxMin", "./{*}Global/{*}FxBand/{*}FxMax",
            "./{*}Global/{*}Timeline/{*}CollectionStart",
            "./{*}Channel/{*}RefChId",
            "./{*}ReferenceGeometry/{*}Monostatic/{*}SideOfTrack",
            "./{*}ReferenceGeometry/{*}Monostatic/{*}ARPPos", "./{*}ReferenceGeometry/{*}Monostatic/{*}ARPVel",
            "./{*}ReferenceGeometry/{*}SRP/{*}ECF", "./{*}ReferenceGeometry/{*}ReferenceTime",
            "./{*}ReferenceGeometry/{*}Monostatic/{*}SlantRange", "./{*}ReferenceGeometry/{*}Monostatic/{*}GrazeAngle",
            "./{*}ReferenceGeometry/{*}Monostatic/{*}TwistAngle", "./{*}ReferenceGeometry/{*}Monostatic/{*}SlopeAngle",
            "./{*}ReferenceGeometry/{*}Monostatic/{*}AzimuthAngle", "./{*}ReferenceGeometry/{*}Monostatic/{*}LayoverAngle",
            "./{*}ReferenceGeometry/{*}Monostatic/{*}DopplerConeAngle", "./{*}ReferenceGeometry/{*}Monostatic/{*}IncidenceAngle",
            "./{*}SceneCoordinates/{*}IARP/{*}ECF", "./{*}SceneCoordinates/{*}IARP/{*}LLH",
            "./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX",
            "./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY",
            "./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1", "./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2",
            "./{*}SceneCoordinates/{*}ExtendedArea/{*}X1Y1", "./{*}SceneCoordinates/{*}ExtendedArea/{*}X2Y2",
            "./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}LineSpacing",
            "./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}FirstLine",
            "./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}NumLines",
            "./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}SampleSpacing",
            "./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}FirstSample",
            "./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}NumSamples",
        ]:
            s[q.replace("./{*}", "").replace("/{*}", "/")] = L(h, q)
        s["channels"] = []
        for c in x.findall(".//{*}Data/{*}Channel"):
            cid = c.find("./{*}Identifier").text
            nv = int(c.find("./{*}NumVectors").text)
            ns = int(c.find("./{*}NumSamples").text)
            par = None
            for p in x.findall(".//{*}Channel/{*}Parameters"):
                if p.find("./{*}Identifier").text == cid:
                    par = p
            ch = {"id": cid, "NumVectors": nv, "NumSamples": ns}
            if par is not None:
                for tag in ["FxC", "FxBW", "FXFixed", "TOAFixed", "SRPFixed", "SignalNormal"]:
                    e = par.find(f"./{{*}}{tag}")
                    ch[tag] = e.text if e is not None else None
                pol = par.find("./{*}Polarization")
                if pol is not None:
                    ch["TxPol"] = pol.find("./{*}TxPol").text
                    ch["RcvPol"] = pol.find("./{*}RcvPol").text
            pv = r.read_pvps(cid)
            ch["pvp_fields"] = list(pv.dtype.names)
            for n in ["TxTime", "RcvTime", "SC0", "SCSS", "FX1", "FX2", "TOA1", "TOA2", "aFDOP", "SIGNAL"]:
                if n in pv.dtype.names:
                    a = pv[n].astype(float)
                    ch[n] = {"min": float(a.min()), "max": float(a.max()), "first": float(a[0]), "last": float(a[-1])}
            if "SIGNAL" in pv.dtype.names:
                sig = pv["SIGNAL"].astype(int)
                bad = np.where(sig != 1)[0]
                ch["SIGNAL_n_not_1"] = int(bad.size)
                ch["SIGNAL_bad_index_ranges"] = _ranges(bad)
            ch["SRPPos_constant"] = bool(np.allclose(pv["SRPPos"], pv["SRPPos"][0]))
            ch["max_|TxPos-RcvPos|_m"] = float(np.abs(pv["TxPos"] - pv["RcvPos"]).max())
            ch["max_|TxTime-RcvTime|_s"] = float(np.abs(pv["TxTime"] - pv["RcvTime"]).max())
            # implied fast-time frequency grid vs FX1/FX2
            if "SC0" in pv.dtype.names:
                f0 = pv["SC0"].astype(float); ss = pv["SCSS"].astype(float)
                ch["F_first_sample_mean"] = float(f0.mean())
                ch["F_last_sample_mean"] = float((f0 + (ns - 1) * ss).mean())
                ch["SCSS_rel_spread"] = float((ss.max() - ss.min()) / ss.mean())
            s["channels"].append(ch)
    return s


def _ranges(idx):
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i != prev + 1:
            out.append([start, prev]); start = i
        prev = i
    out.append([start, prev])
    return out[:20]


def sicd_summary(path):
    with open(path, "rb") as f:
        r = sksicd.NitfReader(f)
        x = r.metadata.xmltree
    h = sksicd.XmlHelper(x)
    s = {"file": os.path.basename(path), "size_GB": os.path.getsize(path) / 1e9}
    for q in [
        "./{*}CollectionInfo/{*}CollectorName", "./{*}CollectionInfo/{*}RadarMode/{*}ModeType",
        "./{*}ImageCreation/{*}Application", "./{*}ImageCreation/{*}DateTime",
        "./{*}ImageData/{*}PixelType", "./{*}ImageData/{*}NumRows", "./{*}ImageData/{*}NumCols",
        "./{*}ImageData/{*}FirstRow", "./{*}ImageData/{*}FirstCol",
        "./{*}ImageData/{*}FullImage/{*}NumRows", "./{*}ImageData/{*}FullImage/{*}NumCols",
        "./{*}ImageData/{*}SCPPixel/{*}Row", "./{*}ImageData/{*}SCPPixel/{*}Col",
        "./{*}GeoData/{*}SCP/{*}ECF", "./{*}GeoData/{*}SCP/{*}LLH",
        "./{*}Grid/{*}ImagePlane", "./{*}Grid/{*}Type", "./{*}Grid/{*}TimeCOAPoly",
        "./{*}Grid/{*}Row/{*}UVectECF", "./{*}Grid/{*}Row/{*}SS", "./{*}Grid/{*}Row/{*}ImpRespWid",
        "./{*}Grid/{*}Row/{*}Sgn", "./{*}Grid/{*}Row/{*}ImpRespBW", "./{*}Grid/{*}Row/{*}KCtr",
        "./{*}Grid/{*}Row/{*}DeltaK1", "./{*}Grid/{*}Row/{*}DeltaK2", "./{*}Grid/{*}Row/{*}DeltaKCOAPoly",
        "./{*}Grid/{*}Row/{*}WgtType/{*}WindowName",
        "./{*}Grid/{*}Col/{*}UVectECF", "./{*}Grid/{*}Col/{*}SS", "./{*}Grid/{*}Col/{*}ImpRespWid",
        "./{*}Grid/{*}Col/{*}Sgn", "./{*}Grid/{*}Col/{*}ImpRespBW", "./{*}Grid/{*}Col/{*}KCtr",
        "./{*}Grid/{*}Col/{*}DeltaK1", "./{*}Grid/{*}Col/{*}DeltaK2", "./{*}Grid/{*}Col/{*}DeltaKCOAPoly",
        "./{*}Grid/{*}Col/{*}WgtType/{*}WindowName",
        "./{*}Timeline/{*}CollectStart", "./{*}Timeline/{*}CollectDuration",
        "./{*}Position/{*}ARPPoly",
        "./{*}RadarCollection/{*}TxFrequency/{*}Min", "./{*}RadarCollection/{*}TxFrequency/{*}Max",
        "./{*}RadarCollection/{*}TxPolarization",
        "./{*}ImageFormation/{*}TStartProc", "./{*}ImageFormation/{*}TEndProc",
        "./{*}ImageFormation/{*}TxFrequencyProc/{*}MinProc", "./{*}ImageFormation/{*}TxFrequencyProc/{*}MaxProc",
        "./{*}ImageFormation/{*}ImageFormAlgo", "./{*}ImageFormation/{*}STBeamComp",
        "./{*}ImageFormation/{*}ImageBeamComp", "./{*}ImageFormation/{*}AzAutofocus", "./{*}ImageFormation/{*}RgAutofocus",
        "./{*}SCPCOA/{*}SCPTime", "./{*}SCPCOA/{*}ARPPos", "./{*}SCPCOA/{*}ARPVel", "./{*}SCPCOA/{*}ARPAcc",
        "./{*}SCPCOA/{*}SideOfTrack", "./{*}SCPCOA/{*}SlantRange", "./{*}SCPCOA/{*}GroundRange",
        "./{*}SCPCOA/{*}DopplerConeAng", "./{*}SCPCOA/{*}GrazeAng", "./{*}SCPCOA/{*}IncidenceAng",
        "./{*}SCPCOA/{*}TwistAng", "./{*}SCPCOA/{*}SlopeAng", "./{*}SCPCOA/{*}AzimAng", "./{*}SCPCOA/{*}LayoverAng",
        "./{*}Radiometric/{*}NoiseLevel/{*}NoiseLevelType", "./{*}Radiometric/{*}RCSSFPoly",
        "./{*}Radiometric/{*}SigmaZeroSFPoly", "./{*}Radiometric/{*}BetaZeroSFPoly", "./{*}Radiometric/{*}GammaZeroSFPoly",
        "./{*}PFA/{*}FPN", "./{*}PFA/{*}IPN", "./{*}PFA/{*}PolarAngRefTime", "./{*}PFA/{*}PolarAngPoly",
        "./{*}PFA/{*}SpatialFreqSFPoly", "./{*}PFA/{*}Krg1", "./{*}PFA/{*}Krg2", "./{*}PFA/{*}Kaz1", "./{*}PFA/{*}Kaz2",
        "./{*}PFA/{*}STDeskew/{*}Applied",
    ]:
        s[q.replace("./{*}", "").replace("/{*}", "/")] = L(h, q)
    icps = []
    for icp in x.findall("./{*}GeoData/{*}ImageCorners/{*}ICP"):
        icps.append({"index": icp.get("index"), "Lat": float(icp.find("./{*}Lat").text), "Lon": float(icp.find("./{*}Lon").text)})
    s["ImageCorners"] = icps
    # top-level blocks present
    s["top_level_blocks"] = [c.tag.split("}")[-1] for c in x.getroot()]
    # weight function present?
    s["Row/WgtFunct_present"] = x.find("./{*}Grid/{*}Row/{*}WgtFunct") is not None
    s["Col/WgtFunct_present"] = x.find("./{*}Grid/{*}Col/{*}WgtFunct") is not None
    for d in ("Row", "Col"):
        bw = s.get(f"Grid/{d}/ImpRespBW"); wid = s.get(f"Grid/{d}/ImpRespWid"); ss = s.get(f"Grid/{d}/SS")
        if bw and wid:
            s[f"Grid/{d}/k=ImpRespWid*ImpRespBW"] = wid * bw
        if bw and ss:
            s[f"Grid/{d}/SS*ImpRespBW (1/oversample)"] = ss * bw
    return s


def main():
    inv = {"cphd": [], "vendor_sicd": [], "diffpfa_sicd": []}
    for p in sorted(glob.glob(os.path.join(DATA, "*_CPHD.cphd"))):
        print("CPHD", p, flush=True)
        inv["cphd"].append(cphd_summary(p))
    for p in sorted(glob.glob(os.path.join(DATA, "*_SICD.nitf"))):
        print("vendor SICD", p, flush=True)
        inv["vendor_sicd"].append(sicd_summary(p))
    for p in sorted(glob.glob(os.path.join(OUT, "*.nitf"))):
        print("diffpfa SICD", p, flush=True)
        d = sicd_summary(p)
        d["mtime"] = os.path.getmtime(p)
        inv["diffpfa_sicd"].append(d)
    with open(os.path.join(HERE, "a01_inventory.json"), "w") as f:
        json.dump(inv, f, indent=1, default=str)
    print("wrote", os.path.join(HERE, "a01_inventory.json"))


if __name__ == "__main__":
    main()
