import numpy as np
import sarkit.sicd as ss
from pathlib import Path
from diffpfa.sicd_geometry import get_geodetic_up_vector

stem = "2025-10-26-05-00-15_UMBRA-08"
umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")

with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
    xh = ss.XmlHelper(r.metadata.xmltree)
    scp = np.array(xh.load("./{*}GeoData/{*}SCP/{*}ECF"))
    pfa_elem = r.metadata.xmltree.find(".//{*}PFA")
    fpn = np.array([float(pfa_elem.find(f"./{{*}}FPN/{{*}}{c}").text) for c in ["X", "Y", "Z"]])
    
up_vec = get_geodetic_up_vector(scp)
print("Umbra FPN:       ", fpn)
print("Geodetic Up ECF: ", up_vec)
print("Diff:            ", fpn - up_vec)
print("Dot product:     ", np.dot(fpn, up_vec))
