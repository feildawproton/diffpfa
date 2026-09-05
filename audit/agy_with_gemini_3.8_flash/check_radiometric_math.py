import sarkit.sicd as ss
from pathlib import Path
import numpy as np

stem = "2025-10-26-05-00-15_UMBRA-08"
umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")
diffpfa_sicd = Path(f"/home/feildaw/diffpfa/workspace/output/{stem}_SICDU_V_V.nitf")

with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
    xh = ss.XmlHelper(r.metadata.xmltree)
    graze = xh.load("./{*}SCPCOA/{*}GrazeAng")
    inc = xh.load("./{*}SCPCOA/{*}IncidenceAng")
    beta = xh.load("./{*}Radiometric/{*}BetaZeroSFPoly/{*}Coef")
    sigma = xh.load("./{*}Radiometric/{*}SigmaZeroSFPoly/{*}Coef")
    gamma = xh.load("./{*}Radiometric/{*}GammaZeroSFPoly/{*}Coef")
    print("=== UMBRA GOLD RADIOMETRIC ===")
    print(f"Graze: {graze:.4f} deg, Incidence: {inc:.4f} deg")
    print(f"BetaZero:  {beta}")
    print(f"SigmaZero: {sigma}")
    print(f"GammaZero: {gamma}")
    if beta is not None and sigma is not None:
        print(f"Ratio SigmaZero / BetaZero: {sigma / beta:.6f}")
        print(f"sin(GrazeAng):              {np.sin(np.radians(graze)):.6f}")
        print(f"cos(GrazeAng):              {np.cos(np.radians(graze)):.6f}")
        print(f"Ratio GammaZero / BetaZero: {gamma / beta:.6f}")
        print(f"tan(GrazeAng):              {np.tan(np.radians(graze)):.6f}")

with open(diffpfa_sicd, "rb") as f, ss.NitfReader(f) as r:
    xh = ss.XmlHelper(r.metadata.xmltree)
    graze = xh.load("./{*}SCPCOA/{*}GrazeAng")
    beta = xh.load("./{*}Radiometric/{*}BetaZeroSFPoly/{*}Coef")
    sigma = xh.load("./{*}Radiometric/{*}SigmaZeroSFPoly/{*}Coef")
    gamma = xh.load("./{*}Radiometric/{*}GammaZeroSFPoly/{*}Coef")
    print("\n=== DIFFPFA RADIOMETRIC ===")
    print(f"BetaZero:  {beta}")
    print(f"SigmaZero: {sigma}")
    print(f"GammaZero: {gamma}")
    print(f"Ratio SigmaZero / BetaZero: {sigma / beta:.6f}")
    print(f"sin(GrazeAng):              {np.sin(np.radians(graze)):.6f}")
    print(f"cos(GrazeAng):              {np.cos(np.radians(graze)):.6f}")
