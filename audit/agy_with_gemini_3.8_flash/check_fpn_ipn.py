import numpy as np
import sarkit.sicd as ss
from pathlib import Path

stem = "2025-10-26-05-00-15_UMBRA-08"
umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")

with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
    xh = ss.XmlHelper(r.metadata.xmltree)
    row_u = np.array(xh.load("./{*}Grid/{*}Row/{*}UVectECF"))
    col_u = np.array(xh.load("./{*}Grid/{*}Col/{*}UVectECF"))
    pfa_elem = r.metadata.xmltree.find(".//{*}PFA")
    fpn = np.array([float(pfa_elem.find(f"./{{*}}FPN/{{*}}{c}").text) for c in ["X", "Y", "Z"]])
    ipn = np.array([float(pfa_elem.find(f"./{{*}}IPN/{{*}}{c}").text) for c in ["X", "Y", "Z"]])
    
    print("Umbra row_u:", row_u)
    print("Umbra col_u:", col_u)
    print("Umbra FPN:", fpn)
    print("Umbra IPN:", ipn)
    print("cross(row_u, col_u):", np.cross(row_u, col_u))
    print("dot(IPN, cross(row_u, col_u)):", np.dot(ipn, np.cross(row_u, col_u)))
    print("dot(FPN, row_u):", np.dot(fpn, row_u))
    print("dot(FPN, col_u):", np.dot(fpn, col_u))
    print("dot(FPN, IPN):", np.dot(fpn, ipn))
