import numpy as np
import sarkit.sicd as ss
import sarkit.cphd as sc
from pathlib import Path

stem = "2025-10-26-05-00-15_UMBRA-08"
cphd_path = Path(f"/home/feildaw/data/{stem}_CPHD.cphd")
umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")

with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
    xh = ss.XmlHelper(r.metadata.xmltree)
    graze = xh.load("./{*}SCPCOA/{*}GrazeAng")
    slope = xh.load("./{*}SCPCOA/{*}SlopeAng")
    twist = xh.load("./{*}SCPCOA/{*}TwistAng")
    azim = xh.load("./{*}SCPCOA/{*}AzimAng")
    layover = xh.load("./{*}SCPCOA/{*}LayoverAng")
    dca = xh.load("./{*}SCPCOA/{*}DopplerConeAng")
    print(f"Umbra SICD SCPCOA: Graze={graze:.4f} deg, Slope={slope:.4f} deg, Twist={twist:.4f} deg, Azim={azim:.4f} deg")
    print(f"2500 * cos(GrazeAng): {2500.0 * np.cos(np.radians(graze)):.2f} m")
    print(f"2500 * cos(SlopeAng): {2500.0 * np.cos(np.radians(slope)):.2f} m")

    # Let's also check Grid Row and Col UVect vs CPHD uIAX, uIAY
    row_u = np.array(xh.load("./{*}Grid/{*}Row/{*}UVectECF"))
    col_u = np.array(xh.load("./{*}Grid/{*}Col/{*}UVectECF"))

with open(cphd_path, "rb") as f:
    r = sc.Reader(f)
    xh = sc.XmlHelper(r.metadata.xmltree)
    uIAX = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"))
    uIAY = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"))
    print("\nCPHD ReferenceSurface:")
    print("uIAX:", uIAX)
    print("uIAY:", uIAY)
    print("uIAX . uIAY:", np.dot(uIAX, uIAY))
    print("Umbra SICD Row UVect . uIAX:", np.dot(row_u, uIAX))
    print("Umbra SICD Row UVect . uIAY:", np.dot(row_u, uIAY))
    print("Umbra SICD Col UVect . uIAX:", np.dot(col_u, uIAX))
    print("Umbra SICD Col UVect . uIAY:", np.dot(col_u, uIAY))
