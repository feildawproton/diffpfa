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

def compute_scp_geometry(srp_ecf: np.ndarray, arp_pos_coa: np.ndarray, arp_vel_coa: np.ndarray, image_plane: str = "SLANT"):
    """
    Computes SICD geometry parameters evaluated at the Scene Center Point (SCP).
    
    Returns:
        dict containing:
        - SlantRange (meters)
        - GroundRange (meters)
        - DopplerConeAng (degrees)
        - GrazeAng (degrees)
        - IncidenceAng (degrees)
        - AzimAng (degrees)
        - SlopeAng (degrees)
    """
    
    # 1. Slant Range and Line of Sight (LOS) Vector
    los_vec = srp_ecf - arp_pos_coa
    slant_range = float(np.linalg.norm(los_vec))
    u_los = los_vec / slant_range
    
    # 2. Doppler Cone Angle (angle between velocity and LOS)
    v_mag = np.linalg.norm(arp_vel_coa)
    u_vel = arp_vel_coa / v_mag
    cos_dca = np.clip(np.dot(u_vel, u_los), -1.0, 1.0)
    dca_deg = float(np.degrees(np.arccos(cos_dca)))
    
    # 3. Grazing and Incidence Angles relative to the ETP at SCP
    u_up = get_geodetic_up_vector(srp_ecf)
    sin_graze = np.clip(np.dot(-u_los, u_up), -1.0, 1.0)
    graze_deg = float(np.degrees(np.arcsin(sin_graze)))
    incidence_deg = 90.0 - graze_deg
    
    # 4. Ground Range (Approximation)
    ground_range = slant_range * np.cos(np.radians(graze_deg))
    
    # 5. Azimuth Angle (Heading from North to projected LOS)
    # Define True North and East vectors on the ETP
    z_ecf = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_north = z_ecf - np.dot(z_ecf, u_up) * u_up
    u_north /= np.linalg.norm(u_north)
    u_east = np.cross(u_north, u_up)
    
    # Project u_los (Sensor to SCP) onto the ETP
    u_los_gnd = u_los - np.dot(u_los, u_up) * u_up
    u_los_gnd /= np.linalg.norm(u_los_gnd)
    
    azim_rad = np.arctan2(np.dot(u_los_gnd, u_east), np.dot(u_los_gnd, u_north))
    azim_deg = float(np.degrees(azim_rad)) % 360.0
    
    # 6. Slope Angle (Angle between Image Plane and Ground Plane)
    if image_plane.upper() == "GROUND":
        slope_deg = 0.0
    else:
        # For SLANT plane, the normal is defined by the cross product of velocity and LOS
        u_spn = np.cross(u_vel, u_los)
        u_spn /= np.linalg.norm(u_spn)
        cos_slope = np.abs(np.dot(u_spn, u_up))
        slope_deg = float(np.degrees(np.arccos(np.clip(cos_slope, -1.0, 1.0))))
    
    return {
        "SlantRange": slant_range,
        "GroundRange": ground_range,
        "DopplerConeAng": dca_deg,
        "GrazeAng": graze_deg,
        "IncidenceAng": incidence_deg,
        "AzimAng": azim_deg,
        "SlopeAng": slope_deg
    }

def fit_arp_poly(cphd_xml: 'xml.etree.ElementTree.Element') -> dict:
    """
    Fits and extracts the ARP position polynomials from the CPHD PVP data.
    Note: A full implementation requires reading the actual PVP arrays (TxPos, TyPos, TzPos) 
    and fitting a polynomial over the dwell time. For now, this returns a placeholder
    to be populated once the PVP reader passes the arrays down.
    """
    return {
        "X": [0.0],
        "Y": [0.0],
        "Z": [0.0]
    }
