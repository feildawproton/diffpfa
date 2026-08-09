import os
import pytest
from diffpfa.io import CPHDReader

SAMPLE_CPHD_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"

@pytest.mark.skipif(not os.path.exists(SAMPLE_CPHD_PATH), reason="CPHD file not found")
def test_metadata_match():
    try:
        r_sarpy = CPHDReader(SAMPLE_CPHD_PATH, backend="sarpy")
        r_sarkit = CPHDReader(SAMPLE_CPHD_PATH, backend="sarkit")
    except ImportError:
        pytest.skip("Both backends required for match test")
        
    m_sarpy = r_sarpy.get_metadata()
    m_sarkit = r_sarkit.get_metadata()
    
    assert m_sarpy.global_fx_min == m_sarkit.global_fx_min
    assert m_sarpy.global_fx_max == m_sarkit.global_fx_max
