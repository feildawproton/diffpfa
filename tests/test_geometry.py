import numpy as np
import pytest
from diffpfa.sicd_geometry import (
    cartesian_to_geodetic,
    get_geodetic_up_vector,
    compute_scp_geometry,
    fit_arp_poly
)

def test_cartesian_to_geodetic():
    # Test on equator at prime meridian (HAE = 0)
    ecf_equator = np.array([6378137.0, 0.0, 0.0])
    lat, lon, hae = cartesian_to_geodetic(ecf_equator)
    assert np.isclose(lat, 0.0, atol=1e-8)
    assert np.isclose(lon, 0.0, atol=1e-8)
    assert np.isclose(hae, 0.0, atol=1e-4)

    # Test North Pole
    b_axis = 6356752.314245
    ecf_pole = np.array([0.0, 0.0, b_axis])
    lat_p, lon_p, hae_p = cartesian_to_geodetic(ecf_pole)
    assert np.isclose(lat_p, np.pi / 2.0, atol=1e-8)
    assert np.isclose(hae_p, 0.0, atol=1e-4)

def test_compute_scp_geometry_umbra_reference():
    # Known geometry values from Umbra-05 collection
    srp = np.array([-5912394.53125, 1100674.43847656, -2117715.57617188])
    arp = np.array([-6166621.24542219, 1464000.48221866, -2740434.88762259])
    arp_v = np.array([-2545.41039668, 2185.03260957, 6903.08403424])

    geom = compute_scp_geometry(srp, arp, arp_v, side_of_track="R", image_plane="SLANT")

    ref = {
        "DopplerConeAng": 60.853277,
        "GrazeAng": 41.475922,
        "IncidenceAng": 48.524078,
        "TwistAng": -35.336303,
        "SlopeAng": 52.323412,
        "AzimAng": 212.846326,
        "LayoverAng": 259.796213
    }

    for angle_name, ref_val in ref.items():
        computed_val = geom[angle_name]
        assert np.isclose(computed_val, ref_val, atol=1e-3), (
            f"{angle_name} mismatch: computed {computed_val}, expected {ref_val}"
        )

def test_fit_arp_poly_residuals():
    # Synthetic circular/elliptical satellite arc
    t = np.linspace(0, 1.0, 100)
    R_orb = 7000e3
    omega = 7.5e3 / R_orb
    x = R_orb * np.cos(omega * t)
    y = R_orb * np.sin(omega * t)
    z = np.full_like(t, 500e3)
    
    pvp = {
        "TxTime": t,
        "TxPos": np.column_stack([x, y, z]),
        "RcvPos": np.column_stack([x, y, z]),
        "TxVel": np.column_stack([-R_orb * omega * np.sin(omega * t), R_orb * omega * np.cos(omega * t), np.zeros_like(t)])
    }
    
    res = fit_arp_poly(pvp, deg=5)
    assert np.isclose(res["CollectDuration"], 1.0, atol=1e-6)
    assert np.isclose(res["SCPTime"], 0.5, atol=1e-2)
    assert len(res["ARPPoly"]["X"]) == 6
    assert np.isclose(res["ARPPos_COA"][2], 500e3, atol=1e-3)
