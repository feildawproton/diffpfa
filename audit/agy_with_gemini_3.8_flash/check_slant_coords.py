import numpy as np
import sarkit.sicd as ss
from pathlib import Path

DATA_DIR = Path("/home/feildaw/data")
WORKSPACE_OUT = Path("/home/feildaw/diffpfa/workspace/output")
stem = "2025-10-26-05-00-15_UMBRA-08"

umbra_path = DATA_DIR / f"{stem}_SICD.nitf"
diffpfa_path = WORKSPACE_OUT / f"{stem}_SICDU_V_V.nitf"

def get_sicd_points(sicd_path):
    with open(sicd_path, "rb") as f, ss.NitfReader(f) as r:
        xml = r.metadata.xmltree
        xh = ss.XmlHelper(xml)
        nr = xh.load("./{*}ImageData/{*}NumRows")
        nc = xh.load("./{*}ImageData/{*}NumCols")
        scp_r = xh.load("./{*}ImageData/{*}SCPPixel/{*}Row")
        scp_c = xh.load("./{*}ImageData/{*}SCPPixel/{*}Col")
        row_ss = xh.load("./{*}Grid/{*}Row/{*}SS")
        col_ss = xh.load("./{*}Grid/{*}Col/{*}SS")
        row_u = np.array(xh.load("./{*}Grid/{*}Row/{*}UVectECF"))
        col_u = np.array(xh.load("./{*}Grid/{*}Col/{*}UVectECF"))
        scp_ecf = np.array(xh.load("./{*}GeoData/{*}SCP/{*}ECF"))
        
        # Check corner pixels: (0,0), (0, nc-1), (nr-1, nc-1), (nr-1, 0)
        corner_pixels = np.array([
            [0, 0],
            [0, nc - 1],
            [nr - 1, nc - 1],
            [nr - 1, 0],
            [scp_r, scp_c]
        ], dtype=np.float64)
        
        # In SICD slant plane:
        # P = SCP + (row - scp_r) * row_ss * row_u + (col - scp_c) * col_ss * col_u
        d_row = (corner_pixels[:, 0] - scp_r) * row_ss
        d_col = (corner_pixels[:, 1] - scp_c) * col_ss
        pts_ecf = scp_ecf + np.outer(d_row, row_u) + np.outer(d_col, col_u)
        
        # Project each point to WGS-84 ellipsoid along Earth normal or LOS?
        # In SICD IPDD:
        # Ground projection from slant plane uses projection model.
        # But let's look at the slant plane coordinates:
        return {
            "nr": nr, "nc": nc,
            "scp_r": scp_r, "scp_c": scp_c,
            "row_ss": row_ss, "col_ss": col_ss,
            "row_u": row_u, "col_u": col_u,
            "scp_ecf": scp_ecf,
            "d_row_min": d_row[:4].min(), "d_row_max": d_row[:4].max(),
            "d_col_min": d_col[:4].min(), "d_col_max": d_col[:4].max(),
            "xml": xml
        }

u_info = get_sicd_points(umbra_path)
d_info = get_sicd_points(diffpfa_path)

print("=== Slant Plane Physical Coordinate Extents relative to SCP ===")
print("Umbra:")
print(f"  Row (Range): [{u_info['d_row_min']:.2f}, {u_info['d_row_max']:.2f}] m (Total: {u_info['d_row_max'] - u_info['d_row_min']:.2f} m)")
print(f"  Col (Azimuth): [{u_info['d_col_min']:.2f}, {u_info['d_col_max']:.2f}] m (Total: {u_info['d_col_max'] - u_info['d_col_min']:.2f} m)")
print("DiffPFA:")
print(f"  Row (Range): [{d_info['d_row_min']:.2f}, {d_info['d_row_max']:.2f}] m (Total: {d_info['d_row_max'] - d_info['d_row_min']:.2f} m)")
print(f"  Col (Azimuth): [{d_info['d_col_min']:.2f}, {d_info['d_col_max']:.2f}] m (Total: {d_info['d_col_max'] - d_info['d_col_min']:.2f} m)")
