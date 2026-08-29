import numpy as np

# WGS-84 ellipsoid parameters
A_AXIS = 6378137.0
B_AXIS = 6356752.314245

def get_geodetic_up_vector(ecf_pos: np.ndarray) -> np.ndarray:
    """
    Returns the normal vector to the WGS-84 ellipsoid (the 'up' vector) at the given ECF position.
    """
    x, y, z = ecf_pos[0], ecf_pos[1], ecf_pos[2]
    
    nx = x / (A_AXIS**2)
    ny = y / (A_AXIS**2)
    nz = z / (B_AXIS**2)
    
    n_vec = np.array([nx, ny, nz], dtype=np.float64)
    return n_vec / np.linalg.norm(n_vec)

def cartesian_to_geodetic(x: np.ndarray) -> np.ndarray:
    """
    Converts ECF Cartesian coordinates [X, Y, Z] (meters) to Geodetic coordinates [Lat (rad), Lon (rad), HAE (meters)]
    using Bowring's algorithm on the WGS-84 ellipsoid.
    """
    a = A_AXIS
    b = B_AXIS
    e2 = 1.0 - (b**2) / (a**2)
    ep2 = (a**2 - b**2) / (b**2)
    p = np.sqrt(x[0]**2 + x[1]**2)
    th = np.arctan2(a * x[2], b * p)
    lon = np.arctan2(x[1], x[0])
    lat = np.arctan2((x[2] + ep2 * b * np.sin(th)**3), (p - e2 * a * np.cos(th)**3))
    n = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    if np.abs(lat) < np.pi / 4.0:
        alt = p / np.cos(lat) - n
    else:
        alt = x[2] / np.sin(lat) - n + e2 * n
    return np.array([lat, lon, alt])

def compute_scp_geometry(
    srp_ecf: np.ndarray,
    arp_pos_coa: np.ndarray,
    arp_vel_coa: np.ndarray,
    side_of_track: str = "R",
    image_plane: str = "SLANT"
) -> dict:
    """
    Computes SICD geometry parameters evaluated at the Scene Center Point (SCP) at Center of Aperture (COA)
    per NGA.STND.0024-1 (SICD DIDD Section 5.3).
    
    Returns:
        dict containing:
        - SlantRange (meters)
        - GroundRange (meters)
        - DopplerConeAng (degrees)
        - GrazeAng (degrees)
        - IncidenceAng (degrees)
        - TwistAng (degrees)
        - SlopeAng (degrees)
        - AzimAng (degrees)
        - LayoverAng (degrees)
    """
    scp = srp_ecf
    arp_coa = arp_pos_coa
    varp_coa = arp_vel_coa
    
    # 1. Slant Range and Line-of-Sight (LOS)
    r_coa = float(np.linalg.norm(scp - arp_coa))
    u_los_coa = (scp - arp_coa) / r_coa
    
    # 2. Ground Range (spherical Earth central angle from SCP to ARP nadir)
    arp_dec_coa = np.linalg.norm(arp_coa)
    u_arp_coa = arp_coa / arp_dec_coa
    scp_dec = np.linalg.norm(scp)
    u_scp = scp / scp_dec
    ea_coa = np.arccos(np.clip(np.dot(u_arp_coa, u_scp), -1.0, 1.0))
    rg_coa = float(scp_dec * ea_coa)
    
    # 3. Doppler Cone Angle
    vm_coa = np.linalg.norm(varp_coa)
    u_varp_coa = varp_coa / vm_coa
    dca_coa = np.arccos(np.clip(np.dot(u_varp_coa, u_los_coa), -1.0, 1.0))
    dca_deg = float(np.rad2deg(dca_coa))
    
    # Look direction determination (+1 for Left look, -1 for Right look)
    left_coa = np.cross(u_arp_coa, u_varp_coa)
    look = 1.0 if np.dot(left_coa, u_los_coa) > 0 else -1.0
    if side_of_track == "L":
        look = 1.0
    elif side_of_track == "R":
        look = -1.0
        
    # 4. SCP Geodetic coordinates and Earth Tangent Plane (ETP) Basis
    lat_rad, lon_rad, _ = cartesian_to_geodetic(scp)
    u_gpz = get_geodetic_up_vector(scp)
    
    u_east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0], dtype=np.float64)
    u_east /= np.linalg.norm(u_east)
    u_north = np.cross(u_gpz, u_east)
    u_north /= np.linalg.norm(u_north)
    
    # Project ARP onto ETP to define ETP X (ground range) and Y (ground cross-range)
    arp_gpz_coa = np.dot(arp_coa - scp, u_gpz)
    aetp_coa = (arp_coa - scp) - u_gpz * arp_gpz_coa
    arp_gpx_coa = np.linalg.norm(aetp_coa)
    u_gpx = aetp_coa / arp_gpx_coa
    u_gpy = np.cross(u_gpz, u_gpx)
    
    # 5. Grazing and Incidence Angles
    sin_graz = np.clip(arp_gpz_coa / r_coa, -1.0, 1.0)
    graze_deg = float(np.rad2deg(np.arcsin(sin_graz)))
    incidence_deg = 90.0 - graze_deg
    
    # 6. Slant Plane Normal (u_spz)
    spz = look * np.cross(u_varp_coa, u_los_coa)
    u_spz = spz / np.linalg.norm(spz)
    
    # 7. Slope Angle
    if image_plane.upper() == "GROUND":
        slope_deg = 0.0
    else:
        slope = np.arccos(np.clip(np.dot(u_gpz, u_spz), -1.0, 1.0))
        slope_deg = float(np.rad2deg(slope))
        
    # 8. Azimuth Angle (Measured from True North to SCP-to-ARP line along ETP)
    az_north = np.dot(u_north, u_gpx)
    az_east = np.dot(u_east, u_gpx)
    azim_deg = float(np.rad2deg(np.arctan2(az_east, az_north)) % 360.0)
    
    # 9. Layover Angle (Direction of vertical target layover on ETP)
    cos_slope = np.cos(np.deg2rad(slope_deg))
    lodir_coa = u_gpz - u_spz / max(cos_slope, 1e-6)
    lo_north = np.dot(u_north, lodir_coa)
    lo_east = np.dot(u_east, lodir_coa)
    layover_deg = float(np.rad2deg(np.arctan2(lo_east, lo_north)) % 360.0)
    
    # 10. Twist Angle (Twist of image plane relative to ETP cross-range axis)
    twst_rad = -np.arcsin(np.clip(np.dot(u_gpy, u_spz), -1.0, 1.0))
    twist_deg = float(np.rad2deg(twst_rad))
    
    return {
        "SlantRange": r_coa,
        "GroundRange": rg_coa,
        "DopplerConeAng": dca_deg,
        "GrazeAng": graze_deg,
        "IncidenceAng": incidence_deg,
        "TwistAng": twist_deg,
        "SlopeAng": slope_deg,
        "AzimAng": azim_deg,
        "LayoverAng": layover_deg
    }

import numpy.polynomial.polynomial as npp

def fit_arp_poly(pvp: dict, deg: int = 5) -> dict:
    """
    Fits and extracts the ARP position polynomials and COA kinematics from CPHD PVP data.
    
    Args:
        pvp: Dict containing CPHD PVP arrays ('TxTime', 'TxPos', 'RcvPos', optionally 'TxVel', 'RcvVel').
        deg: Polynomial degree (default 5, per NGA standard).
        
    Returns:
        dict containing:
        - CollectDuration (float, seconds)
        - SCPTime (float, seconds relative to CollectStart)
        - ARPPoly: dict with 'X', 'Y', 'Z' lists of polynomial coefficients in increasing powers of time.
        - ARPPos_COA: np.ndarray (3,)
        - ARPVel_COA: np.ndarray (3,)
        - ARPAcc_COA: np.ndarray (3,)
    """
    if pvp is None or "TxTime" not in pvp or "TxPos" not in pvp:
        return {
            "CollectDuration": 0.0,
            "SCPTime": 0.0,
            "ARPPoly": {"X": [0.0], "Y": [0.0], "Z": [0.0]},
            "ARPPos_COA": np.array([0.0, 0.0, 0.0]),
            "ARPVel_COA": np.array([1.0, 0.0, 0.0]),
            "ARPAcc_COA": np.array([0.0, 0.0, 0.0]),
        }
        
    tx_time = np.ascontiguousarray(pvp["TxTime"]).astype(np.float64)
    tx_pos = np.ascontiguousarray(pvp["TxPos"]).astype(np.float64)
    if "RcvPos" in pvp:
        rcv_pos = np.ascontiguousarray(pvp["RcvPos"]).astype(np.float64)
        arp_pos = 0.5 * (tx_pos + rcv_pos)
    else:
        arp_pos = tx_pos
        
    t_start = float(tx_time[0])
    t_end = float(tx_time[-1])
    duration = max(0.0, t_end - t_start)
    t_rel = tx_time - t_start
    
    num_pts = len(tx_time)
    fit_deg = min(deg, max(1, num_pts - 1))
    
    coef_x = npp.polyfit(t_rel, arp_pos[:, 0], deg=fit_deg)
    coef_y = npp.polyfit(t_rel, arp_pos[:, 1], deg=fit_deg)
    coef_z = npp.polyfit(t_rel, arp_pos[:, 2], deg=fit_deg)
    
    mid_idx = num_pts // 2
    t_coa = float(t_rel[mid_idx])
    
    arp_poly = np.vstack([coef_x, coef_y, coef_z])
    varp_poly = npp.polyder(arp_poly, m=1, axis=1)
    aarp_poly = npp.polyder(arp_poly, m=2, axis=1)
    
    pos_coa = npp.polyval(t_coa, arp_poly.T)
    vel_coa = npp.polyval(t_coa, varp_poly.T)
    acc_coa = npp.polyval(t_coa, aarp_poly.T)
    
    return {
        "CollectDuration": duration,
        "SCPTime": t_coa,
        "ARPPoly": {
            "X": coef_x.tolist(),
            "Y": coef_y.tolist(),
            "Z": coef_z.tolist()
        },
        "ARPPos_COA": pos_coa,
        "ARPVel_COA": vel_coa,
        "ARPAcc_COA": acc_coa
    }

def compute_pfa_metadata(
    pvp: dict,
    uIAX: np.ndarray,
    uIAY: np.ndarray,
    num_samples: int,
    domain_type: str = "FX",
    side_of_track: str = "R",
    deg: int = 5
) -> dict:
    """
    Computes SICD <PFA> metadata block parameters and space-variant polynomials per NGA.STND.0024-1 Section 5.6.
    
    Returns:
        dict containing:
        - FPN: np.ndarray (3,)
        - IPN: np.ndarray (3,)
        - PolarAngRefTime: float
        - PolarAngPoly: list of float coefficients
        - SpatialFreqSFPoly: list of float coefficients
        - Krg1: float
        - Krg2: float
        - Kaz1: float
        - Kaz2: float
    """
    from diffpfa.IFA.kspace import compute_kspace
    
    Ku, Kr = compute_kspace(pvp, uIAX, uIAY, num_samples, domain_type=domain_type, device="cpu")
    
    tx_time = np.ascontiguousarray(pvp["TxTime"]).astype(np.float64)
    t_start = float(tx_time[0])
    t_rel = tx_time - t_start
    num_pts = len(tx_time)
    mid_idx = num_pts // 2
    t_ref = float(t_rel[mid_idx])
    dt = t_rel - t_ref
    
    # Pulse center spatial frequency coordinates
    N_s = Ku.shape[1]
    Ku_mid = Ku[:, N_s // 2].numpy()
    Kr_mid = Kr[:, N_s // 2].numpy()
    
    polar_ang = np.arctan2(Ku_mid, Kr_mid)
    mag_k = np.sqrt(Ku_mid**2 + Kr_mid**2)
    sf_scale = mag_k / max(mag_k[mid_idx], 1e-12)
    
    fit_deg = min(deg, max(1, num_pts - 1))
    polar_ang_coef = npp.polyfit(dt, polar_ang, deg=fit_deg)
    sf_coef = npp.polyfit(dt, sf_scale, deg=fit_deg)
    
    krg1 = float(Kr.min().item())
    krg2 = float(Kr.max().item())
    kaz1 = float(Ku.min().item())
    kaz2 = float(Ku.max().item())
    
    # Image Plane Normal (IPN) and Focus Plane Normal (FPN)
    ipn = np.cross(uIAY, uIAX)
    ipn /= np.linalg.norm(ipn)
    
    # Slant Plane Normal (FPN)
    srp_coa = pvp["SRPPos"][mid_idx]
    arp_coa = 0.5 * (pvp["TxPos"][mid_idx] + pvp["RcvPos"][mid_idx])
    u_los = (srp_coa - arp_coa) / np.linalg.norm(srp_coa - arp_coa)
    u_vel = pvp["TxVel"][mid_idx] / np.linalg.norm(pvp["TxVel"][mid_idx])
    
    look = 1.0 if side_of_track == "L" else -1.0
    fpn = look * np.cross(u_vel, u_los)
    fpn /= np.linalg.norm(fpn)
    
    return {
        "FPN": fpn,
        "IPN": ipn,
        "PolarAngRefTime": t_ref,
        "PolarAngPoly": polar_ang_coef.tolist(),
        "SpatialFreqSFPoly": sf_coef.tolist(),
        "Krg1": krg1,
        "Krg2": krg2,
        "Kaz1": kaz1,
        "Kaz2": kaz2
    }
