import sarkit.sicd as ss
from pathlib import Path
import numpy as np

for p in sorted(Path("/home/feildaw/data").glob("*_SICD.nitf")):
    with open(p, "rb") as f, ss.NitfReader(f) as r:
        xh = ss.XmlHelper(r.metadata.xmltree)
        beta = xh.load("./{*}Radiometric/{*}BetaZeroSFPoly/{*}Coef")
        sigma = xh.load("./{*}Radiometric/{*}SigmaZeroSFPoly/{*}Coef")
        gamma = xh.load("./{*}Radiometric/{*}GammaZeroSFPoly/{*}Coef")
        graze = xh.load("./{*}SCPCOA/{*}GrazeAng")
        slope = xh.load("./{*}SCPCOA/{*}SlopeAng")
        print(f"{p.stem}: Has Radiometric? {beta is not None}")
        if beta is not None:
            print(f"  Graze={graze:.2f} deg, Slope={slope:.2f} deg")
            print(f"  Sigma/Beta = {sigma/beta:.6f}, cos(Slope)={np.cos(np.radians(slope)):.6f}, sin(Graze)={np.sin(np.radians(graze)):.6f}")
