"""
Reference Ground-to-Slant Projection and Image Extent Calculator
================================================================
Auditor: Agy with Gemini 3.8 Flash
Reference: NGA.STND.0024-1 (SICD DIDD §5.2-5.4), NGA.STND.0068-1 (CPHD DIDD §6.2)

Provides the exact analytical transformation to map CPHD ReferenceSurface (Ground)
ImageArea bounds [X1, X2] x [Y1, Y2] into SICD Slant Plane [R_min, R_max] x [U_min, U_max]
with standard over-formation margin (pad_factor = 1.20) and rigorous corner geodetics.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.fft import next_fast_len

from diffpfa.sicd_geometry import cartesian_to_geodetic

def compute_slant_framing_from_cphd(
    ia_x1y1: np.ndarray,
    ia_x2y2: np.ndarray,
    uIAX: np.ndarray,
    uIAY: np.ndarray,
    srp_ecf: np.ndarray,
    arp_pos_coa: np.ndarray,
    arp_vel_coa: np.ndarray,
    side_of_track: str = "R",
    bw_range: float = 1.0,
    bw_azm: float = 1.0,
    oversample: float = 1.25,
    pad_factor: float = 1.20
) -> Dict:
    """
    Computes standard-compliant Slant Plane image extents and grid dimensions
    from CPHD ground ImageArea bounds.
    """
    # 1. Slant Plane Basis Vectors
    p_vec = srp_ecf - arp_pos_coa
    u_row = p_vec / np.linalg.norm(p_vec)  # LOS unit vector
    
    u_v = arp_vel_coa / np.linalg.norm(arp_vel_coa)
    u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
    u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
    if side_of_track == "L":
        u_col = -u_col
        
    # 2. Four Corners of Ground ImageArea relative to IARP
    x1, y1 = min(ia_x1y1[0], ia_x2y2[0]), min(ia_x1y1[1], ia_x2y2[1])
    x2, y2 = max(ia_x1y1[0], ia_x2y2[0]), max(ia_x1y1[1], ia_x2y2[1])
    
    ground_corners = [
        np.array([x1, y1]),
        np.array([x1, y2]),
        np.array([x2, y2]),
        np.array([x2, y1])
    ]
    
    # 3. Project Corners into Slant Plane
    proj_r = []
    proj_u = []
    for c in ground_corners:
        dp = c[0] * uIAX + c[1] * uIAY
        proj_r.append(float(np.dot(dp, u_row)))
        proj_u.append(float(np.dot(dp, u_col)))
        
    r_min_raw, r_max_raw = min(proj_r), max(proj_r)
    u_min_raw, u_max_raw = min(proj_u), max(proj_u)
    
    # 4. Apply Over-formation Padding
    r_min_pad = r_min_raw * pad_factor
    r_max_pad = r_max_raw * pad_factor
    u_min_pad = u_min_raw * pad_factor
    u_max_pad = u_max_raw * pad_factor
    
    L_range = r_max_pad - r_min_pad
    L_azm = u_max_pad - u_min_pad
    
    # 5. Native Nyquist Spacing with Oversample
    dr = 1.0 / (bw_range * oversample)
    du = 1.0 / (bw_azm * oversample)
    
    N_range = next_fast_len(int(np.ceil(L_range / dr)))
    N_azm = next_fast_len(int(np.ceil(L_azm / du)))
    
    # Recomputed exact sample spacing matching discrete grid
    ss_row = L_range / N_range
    ss_col = L_azm / N_azm
    
    # SCP Pixel Index
    scp_row = int(round(-r_min_pad / ss_row))
    scp_col = int(round(-u_min_pad / ss_col))
    
    # 6. Rigorous Corner Geodetic Points (1:FRFC, 2:FRLC, 3:LRLC, 4:LRFC)
    # FRFC: row=0, col=0
    # FRLC: row=0, col=N_azm-1
    # LRLC: row=N_range-1, col=N_azm-1
    # LRFC: row=N_range-1, col=0
    pixel_indices = [
        (0, 0),
        (0, N_azm - 1),
        (N_range - 1, N_azm - 1),
        (N_range - 1, 0)
    ]
    corner_llh = []
    for r_idx, c_idx in pixel_indices:
        d_r = (r_idx - scp_row) * ss_row
        d_u = (c_idx - scp_col) * ss_col
        pt_ecf = srp_ecf + d_r * u_row + d_u * u_col
        lat, lon, hae = cartesian_to_geodetic(pt_ecf)
        corner_llh.append((np.degrees(lat), np.degrees(lon), hae))
        
    return {
        "u_row": u_row,
        "u_col": u_col,
        "r_bounds_raw": (r_min_raw, r_max_raw),
        "u_bounds_raw": (u_min_raw, u_max_raw),
        "r_bounds_padded": (r_min_pad, r_max_pad),
        "u_bounds_padded": (u_min_pad, u_max_pad),
        "L_range": L_range,
        "L_azm": L_azm,
        "N_range": N_range,
        "N_azm": N_azm,
        "ss_row": ss_row,
        "ss_col": ss_col,
        "scp_row": scp_row,
        "scp_col": scp_col,
        "corner_llh": corner_llh
    }
