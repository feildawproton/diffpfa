import os
import pytest
import numpy as np
from diffpfa.io import CPHDReader

SAMPLE_CPHD_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"

@pytest.mark.skipif(not os.path.exists(SAMPLE_CPHD_PATH), reason="CPHD file not found")
def test_txpos_match():
    try:
        r_sarpy = CPHDReader(SAMPLE_CPHD_PATH, backend="sarpy")
        r_sarkit = CPHDReader(SAMPLE_CPHD_PATH, backend="sarkit")
    except ImportError:
        pytest.skip("Both backends required for match test")
        
    ch_sarpy = r_sarpy.read_channel(r_sarpy.get_channel_names()[0])
    ch_sarkit = r_sarkit.read_channel(r_sarkit.get_channel_names()[0])
    
    tx_sarpy = ch_sarpy.pvp["TxPos"]
    tx_sarkit = ch_sarkit.pvp["TxPos"]
    
    assert np.allclose(tx_sarpy, tx_sarkit)
