import sarkit.sicd as ss
from pathlib import Path
import numpy as np

for stem in ["2023-07-30-17-19-39_UMBRA-05", "2023-09-11-10-37-05_UMBRA-05"]:
    umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")
    with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
        xh = ss.XmlHelper(r.metadata.xmltree)
        graze = xh.load("./{*}SCPCOA/{*}GrazeAng")
        slope = xh.load("./{*}SCPCOA/{*}SlopeAng")
        beta = xh.load("./{*}Radiometric/{*}BetaZeroSFPoly/{*}Coef")
        sigma = xh.load("./{*}Radiometric/{*}SigmaZeroSFPoly/{*}Coef")
        gamma = xh.load("./{*}Radiometric/{*}GammaZeroSFPoly/{*}Coef")
        print(f"\n=== {stem} ===")
        print(f"Graze: {graze:.2f} deg, Slope: {slope:.2f} deg")
        print(f"Sigma / Beta: {sigma / beta:.6f}")
        print(f"sin(Graze):   {np.sin(np.radians(graze)):.6f}")
        print(f"cos(Slope):   {np.cos(np.radians(slope)):.6f}")
        print(f"sin(Slope):   {np.sin(np.radians(slope)):.6f}")
        print(f"cos(Graze):   {np.cos(np.radians(graze)):.6f}")
