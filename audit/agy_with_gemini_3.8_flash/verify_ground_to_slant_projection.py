import numpy as np
import sarkit.sicd as ss
import sarkit.cphd as sc
from pathlib import Path
from scipy.fft import next_fast_len

DATA_DIR = Path("/home/feildaw/data")

def project_ground_to_slant(cphd_path, pad_factor=1.20):
    with open(cphd_path, "rb") as f:
        r = sc.Reader(f)
        xml = r.metadata.xmltree
        xh = sc.XmlHelper(xml)
        
        uIAX = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"))
        uIAY = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"))
        ia_x1y1 = np.array(xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1"))
        ia_x2y2 = np.array(xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2"))
        srp = np.array(xh.load("./{*}ReferenceGeometry/{*}SRP/{*}ECF"))
        arp = np.array(xh.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPPos"))
        arp_v = np.array(xh.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPVel"))
        side = str(xh.load("./{*}ReferenceGeometry/{*}Monostatic/{*}SideOfTrack") or "L")
        
        # Calculate Slant Basis Vectors
        p_vec = srp - arp
        u_row = p_vec / np.linalg.norm(p_vec)
        
        u_v = arp_v / np.linalg.norm(arp_v)
        u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
        u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
        if side == "L":
            u_col = -u_col
            
        # Ground Corners
        x1, y1 = ia_x1y1
        x2, y2 = ia_x2y2
        corners = np.array([
            [x1, y1],
            [x1, y2],
            [x2, y2],
            [x2, y1]
        ])
        
        # Project each corner to slant plane
        # dP = x * uIAX + y * uIAY
        # r = dP . u_row
        # u = dP . u_col
        proj_r = [np.dot(c[0]*uIAX + c[1]*uIAY, u_row) for c in corners]
        proj_u = [np.dot(c[0]*uIAX + c[1]*uIAY, u_col) for c in corners]
        
        r_min, r_max = min(proj_r), max(proj_r)
        u_min, u_max = min(proj_u), max(proj_u)
        
        # Apply padding factor
        r_min_pad = r_min * pad_factor
        r_max_pad = r_max * pad_factor
        u_min_pad = u_min * pad_factor
        u_max_pad = u_max * pad_factor
        
        extent_r = r_max_pad - r_min_pad
        extent_u = u_max_pad - u_min_pad
        
        return {
            "u_row": u_row,
            "u_col": u_col,
            "r_bounds_raw": (r_min, r_max),
            "u_bounds_raw": (u_min, u_max),
            "extent_r_padded": extent_r,
            "extent_u_padded": extent_u,
            "r_bounds_padded": (r_min_pad, r_max_pad),
            "u_bounds_padded": (u_min_pad, u_max_pad)
        }

print(f"{'Dataset':<32} | {'Axis':<5} | {'Projected+Pad (m)':<18} | {'Umbra Gold (m)':<18} | {'Ratio':<7}")
print("-" * 90)

for cphd_p in sorted(DATA_DIR.glob("*_CPHD.cphd")):
    stem = cphd_p.name.replace("_CPHD.cphd", "")
    umbra_sicd = DATA_DIR / f"{stem}_SICD.nitf"
    
    proj = project_ground_to_slant(cphd_p, pad_factor=1.20)
    
    with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
        xh = ss.XmlHelper(r.metadata.xmltree)
        nr = xh.load("./{*}ImageData/{*}NumRows")
        nc = xh.load("./{*}ImageData/{*}NumCols")
        row_ss = xh.load("./{*}Grid/{*}Row/{*}SS")
        col_ss = xh.load("./{*}Grid/{*}Col/{*}SS")
        u_ext_r = nr * row_ss
        u_ext_u = nc * col_ss
        
    ratio_r = proj["extent_r_padded"] / u_ext_r
    ratio_u = proj["extent_u_padded"] / u_ext_u
    
    print(f"{stem:<32} | {'Row':<5} | {proj['extent_r_padded']:<18.2f} | {u_ext_r:<18.2f} | {ratio_r:<7.3f}")
    print(f"{'':<32} | {'Col':<5} | {proj['extent_u_padded']:<18.2f} | {u_ext_u:<18.2f} | {ratio_u:<7.3f}")
