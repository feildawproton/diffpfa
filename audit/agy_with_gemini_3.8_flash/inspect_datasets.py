import os
import glob
from pathlib import Path
import sarkit.sicd as ss
import sarkit.cphd as sc
import numpy as np

DATA_DIR = Path("/home/feildaw/data")
WORKSPACE_OUT = Path("/home/feildaw/diffpfa/workspace/output")

print(f"{'Dataset':<35} | {'Source':<10} | {'Rows x Cols':<13} | {'Row/Col SS (m)':<17} | {'Row/Col BW (cpm)':<17} | {'SCPPixel':<12}")
print("-" * 120)

for cphd_path in sorted(DATA_DIR.glob("*_CPHD.cphd")):
    stem = cphd_path.name.replace("_CPHD.cphd", "")
    umbra_sicd = DATA_DIR / f"{stem}_SICD.nitf"
    
    # DiffPFA output may have _SICDU_X_X.nitf or _SICDU_V_V.nitf
    diffpfa_matches = list(WORKSPACE_OUT.glob(f"{stem}_SICDU_*.nitf"))
    diffpfa_sicd = diffpfa_matches[0] if diffpfa_matches else None
    
    for label, sicd_p in [("Umbra", umbra_sicd), ("DiffPFA", diffpfa_sicd)]:
        if sicd_p and sicd_p.exists():
            with open(sicd_p, "rb") as f, ss.NitfReader(f) as r:
                xml = r.metadata.xmltree
                xh = ss.XmlHelper(xml)
                nr = xh.load("./{*}ImageData/{*}NumRows")
                nc = xh.load("./{*}ImageData/{*}NumCols")
                row_ss = xh.load("./{*}Grid/{*}Row/{*}SS")
                col_ss = xh.load("./{*}Grid/{*}Col/{*}SS")
                row_bw = xh.load("./{*}Grid/{*}Row/{*}ImpRespBW")
                col_bw = xh.load("./{*}Grid/{*}Col/{*}ImpRespBW")
                scp_r = xh.load("./{*}ImageData/{*}SCPPixel/{*}Row")
                scp_c = xh.load("./{*}ImageData/{*}SCPPixel/{*}Col")
                first_r = xh.load("./{*}ImageData/{*}FirstRow")
                first_c = xh.load("./{*}ImageData/{*}FirstCol")
                print(f"{stem:<35} | {label:<10} | {f'{nr}x{nc}':<13} | {f'{row_ss:.4f}, {col_ss:.4f}':<17} | {f'{row_bw:.4f}, {col_bw:.4f}':<17} | {f'({scp_r},{scp_c})':<12}")
        else:
            print(f"{stem:<35} | {label:<10} | MISSING")
