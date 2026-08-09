import os
import pytest
import numpy as np
from diffpfa.io import CPHDReader
from diffpfa.algo.pfa_engine import SPEED_OF_LIGHT
from diffpfa.algo.kspace import compute_look_vectors

SAMPLE_CPHD_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"

def get_grid_size(ch_data, cphd_meta):
    bw = cphd_meta.global_fx_max - cphd_meta.global_fx_min
    res_r = SPEED_OF_LIGHT / (2.0 * bw)
    
    P_vecs = compute_look_vectors(ch_data.pvp, device="cpu")
    arp = P_vecs.numpy()
    center_look = arp[len(arp)//2]
    center_look = center_look / np.linalg.norm(center_look)
    delta_look = arp[-1] - arp[0]
    delta_look = delta_look / np.linalg.norm(delta_look)
    sin_theta = np.linalg.norm(np.cross(center_look, delta_look))
    res_u = SPEED_OF_LIGHT / (2.0 * ch_data.fxc * sin_theta)
    
    u_min, u_max = -50.0, 50.0
    r_min, r_max = -50.0, 50.0
    
    N_u = int(np.ceil((u_max - u_min) / res_u * 1.5))
    N_r = int(np.ceil((r_max - r_min) / res_r * 1.5))
    
    M_u = int(np.ceil(N_u * 1.5))
    M_r = int(np.ceil(N_r * 1.5))
    return M_u, M_r

@pytest.mark.skipif(not os.path.exists(SAMPLE_CPHD_PATH), reason="CPHD file not found")
def test_grid_size_match():
    try:
        r_sarpy = CPHDReader(SAMPLE_CPHD_PATH, backend="sarpy")
        r_sarkit = CPHDReader(SAMPLE_CPHD_PATH, backend="sarkit")
    except ImportError:
        pytest.skip("Both backends required for match test")
        
    m_sarpy = r_sarpy.get_metadata()
    ch_sarpy = r_sarpy.read_channel(r_sarpy.get_channel_names()[0])
    
    m_sarkit = r_sarkit.get_metadata()
    ch_sarkit = r_sarkit.read_channel(r_sarkit.get_channel_names()[0])
    
    grid_sarpy = get_grid_size(ch_sarpy, m_sarpy)
    grid_sarkit = get_grid_size(ch_sarkit, m_sarkit)
    
    assert grid_sarpy == grid_sarkit
