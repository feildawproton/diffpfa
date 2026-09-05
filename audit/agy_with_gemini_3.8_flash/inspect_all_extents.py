import numpy as np
import sarkit.sicd as ss
import sarkit.cphd as sc
from pathlib import Path

DATA_DIR = Path("/home/feildaw/data")

for cphd_path in sorted(DATA_DIR.glob("*_CPHD.cphd")):
    stem = cphd_path.name.replace("_CPHD.cphd", "")
    umbra_sicd = DATA_DIR / f"{stem}_SICD.nitf"
    
    with open(cphd_path, "rb") as f:
        r = sc.Reader(f)
        xh = sc.XmlHelper(r.metadata.xmltree)
        ia_x1y1 = xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1")
        ia_x2y2 = xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
        ea_x1y1 = xh.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X1Y1")
        ea_x2y2 = xh.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X2Y2")
        line_ss = xh.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}LineSpacing")
        samp_ss = xh.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}SampleSpacing")
        uIAX = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"))
        uIAY = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"))
        
    with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
        xh = ss.XmlHelper(r.metadata.xmltree)
        nr = xh.load("./{*}ImageData/{*}NumRows")
        nc = xh.load("./{*}ImageData/{*}NumCols")
        row_ss = xh.load("./{*}Grid/{*}Row/{*}SS")
        col_ss = xh.load("./{*}Grid/{*}Col/{*}SS")
        row_bw = xh.load("./{*}Grid/{*}Row/{*}ImpRespBW")
        col_bw = xh.load("./{*}Grid/{*}Col/{*}ImpRespBW")
        graze = xh.load("./{*}SCPCOA/{*}GrazeAng")
        slope = xh.load("./{*}SCPCOA/{*}SlopeAng")
        twist = xh.load("./{*}SCPCOA/{*}TwistAng")
        row_u = np.array(xh.load("./{*}Grid/{*}Row/{*}UVectECF"))
        col_u = np.array(xh.load("./{*}Grid/{*}Col/{*}UVectECF"))
        
    print(f"\n=== {stem} ===")
    print(f"CPHD ImageArea: X=[{ia_x1y1[0]}, {ia_x2y2[0]}], Y=[{ia_x1y1[1]}, {ia_x2y2[1]}] (Span X={ia_x2y2[0]-ia_x1y1[0]}, Span Y={ia_x2y2[1]-ia_x1y1[1]})")
    if ea_x1y1 is not None:
        print(f"CPHD ExtArea:   X=[{ea_x1y1[0]}, {ea_x2y2[0]}], Y=[{ea_x1y1[1]}, {ea_x2y2[1]}] (Span X={ea_x2y2[0]-ea_x1y1[0]}, Span Y={ea_x2y2[1]-ea_x1y1[1]})")
    print(f"CPHD ImageGrid Spacing: Line={line_ss}, Sample={samp_ss}")
    print(f"Umbra SICD: Rows={nr}, Cols={nc}, Row_SS={row_ss:.4f}, Col_SS={col_ss:.4f}")
    print(f"Umbra Slant Extent: Row={nr*row_ss:.2f} m, Col={nc*col_ss:.2f} m")
    print(f"Angles: Graze={graze:.2f}, Slope={slope:.2f}, Twist={twist:.2f}")
    print(f"Dot products: Row_u.uIAX={np.dot(row_u, uIAX):.4f}, Row_u.uIAY={np.dot(row_u, uIAY):.4f}, Col_u.uIAX={np.dot(col_u, uIAX):.4f}, Col_u.uIAY={np.dot(col_u, uIAY):.4f}")
