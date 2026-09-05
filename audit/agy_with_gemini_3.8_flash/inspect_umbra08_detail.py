import os
from pathlib import Path
import sarkit.sicd as ss
import sarkit.cphd as sc
import numpy as np

DATA_DIR = Path("/home/feildaw/data")
WORKSPACE_OUT = Path("/home/feildaw/diffpfa/workspace/output")

stem = "2025-10-26-05-00-15_UMBRA-08"
cphd_path = DATA_DIR / f"{stem}_CPHD.cphd"
umbra_sicd = DATA_DIR / f"{stem}_SICD.nitf"
diffpfa_sicd = WORKSPACE_OUT / f"{stem}_SICDU_V_V.nitf"

print("=== CPHD METADATA ===")
with open(cphd_path, "rb") as f:
    r = sc.Reader(f)
    xh = sc.XmlHelper(r.metadata.xmltree)
    print("DomainType:", xh.load("./{*}Global/{*}DomainType"))
    print("SGN:", xh.load("./{*}Global/{*}SGN"))
    print("FxBand:", xh.load("./{*}Global/{*}FxBand/{*}FxMin"), xh.load("./{*}Global/{*}FxBand/{*}FxMax"))
    print("IARP ECF:", xh.load("./{*}SceneCoordinates/{*}IARP/{*}ECF"))
    print("SRP ECF:", xh.load("./{*}ReferenceGeometry/{*}SRP/{*}ECF"))
    print("uIAX:", xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"))
    print("uIAY:", xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"))
    print("ImageArea X1Y1:", xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1"))
    print("ImageArea X2Y2:", xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2"))
    print("ExtendedArea X1Y1:", xh.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X1Y1"))
    print("ExtendedArea X2Y2:", xh.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X2Y2"))
    print("ImageGrid IAX LineSpacing:", xh.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}LineSpacing"))
    print("ImageGrid IAY SampleSpacing:", xh.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}SampleSpacing"))

for label, p in [("Umbra SICD", umbra_sicd), ("DiffPFA SICD", diffpfa_sicd)]:
    print(f"\n=== {label} ===")
    with open(p, "rb") as f, ss.NitfReader(f) as r:
        xh = ss.XmlHelper(r.metadata.xmltree)
        nr = xh.load("./{*}ImageData/{*}NumRows")
        nc = xh.load("./{*}ImageData/{*}NumCols")
        plane = xh.load("./{*}Grid/{*}ImagePlane")
        row_ss = xh.load("./{*}Grid/{*}Row/{*}SS")
        col_ss = xh.load("./{*}Grid/{*}Col/{*}SS")
        row_u = xh.load("./{*}Grid/{*}Row/{*}UVectECF")
        col_u = xh.load("./{*}Grid/{*}Col/{*}UVectECF")
        row_bw = xh.load("./{*}Grid/{*}Row/{*}ImpRespBW")
        col_bw = xh.load("./{*}Grid/{*}Col/{*}ImpRespBW")
        row_wid = xh.load("./{*}Grid/{*}Row/{*}ImpRespWid")
        col_wid = xh.load("./{*}Grid/{*}Col/{*}ImpRespWid")
        scp_ecf = xh.load("./{*}GeoData/{*}SCP/{*}ECF")
        scp_row = xh.load("./{*}ImageData/{*}SCPPixel/{*}Row")
        scp_col = xh.load("./{*}ImageData/{*}SCPPixel/{*}Col")
        first_row = xh.load("./{*}ImageData/{*}FirstRow")
        first_col = xh.load("./{*}ImageData/{*}FirstCol")
        
        print(f"ImagePlane: {plane}")
        print(f"NumRows x NumCols: {nr} x {nc}")
        print(f"SCPPixel: ({scp_row}, {scp_col}), FirstRow/Col: ({first_row}, {first_col})")
        print(f"Extent: Row = {nr * row_ss:.2f} m, Col = {nc * col_ss:.2f} m")
        print(f"Row SS: {row_ss:.6f} m, Col SS: {col_ss:.6f} m")
        print(f"Row BW: {row_bw:.6f} cpm, Col BW: {col_bw:.6f} cpm")
        print(f"Row BW*SS: {row_bw * row_ss:.4f} (oversample factor = {1.0/(row_bw * row_ss):.4f})")
        print(f"Col BW*SS: {col_bw * col_ss:.4f} (oversample factor = {1.0/(col_bw * col_ss):.4f})")
        print(f"Row ImpRespWid * ImpRespBW: {row_wid * row_bw:.6f}")
        print(f"Col ImpRespWid * Col ImpRespBW: {col_wid * col_bw:.6f}")
        print(f"Row UVect: {row_u}")
        print(f"Col UVect: {col_u}")
        print(f"Row . Col UVect dot product: {np.dot(row_u, col_u):.6e}")
        print(f"SCP ECF: {scp_ecf}")
        
        # Check ImageCorners
        corners = []
        for i in range(1, 5):
            lat = xh.load(f"./{{*}}GeoData/{{*}}ImageCorners/{{*}}ICP[{i}]/{{*}}Lat")
            lon = xh.load(f"./{{*}}GeoData/{{*}}ImageCorners/{{*}}ICP[{i}]/{{*}}Lon")
            corners.append((lat, lon))
        print(f"ImageCorners: {corners}")
