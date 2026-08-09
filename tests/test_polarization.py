import os
import pytest
from diffpfa.io import CPHDReader

SAMPLE_CPHD_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"

@pytest.mark.skipif(not os.path.exists(SAMPLE_CPHD_PATH), reason="CPHD file not found")
@pytest.mark.parametrize("backend", ["sarpy", "sarkit"])
def test_polarization_not_unknown(backend):
    try:
        reader = CPHDReader(SAMPLE_CPHD_PATH, backend=backend)
    except ImportError:
        pytest.skip(f"Backend {backend} not available")
    
    with reader:
        channels = reader.get_channel_names()
        assert len(channels) > 0, "No channels found"
        
        for ch_name in channels:
            ch_data = reader.read_channel(ch_name)
            assert ch_data.tx_pol != "UNKNOWN", f"TxPol extraction failed for channel {ch_name} (backend {backend})"
            assert ch_data.rcv_pol != "UNKNOWN", f"RcvPol extraction failed for channel {ch_name} (backend {backend})"
