import os
import pytest
import torch
from diffpfa.io import CPHDReader

SAMPLE_CPHD_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"

@pytest.mark.skipif(not os.path.exists(SAMPLE_CPHD_PATH), reason="CPHD file not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory tests")
@pytest.mark.parametrize("backend", ["sarpy", "sarkit"])
def test_mem(backend):
    torch.cuda.empty_cache()
    try:
        r = CPHDReader(SAMPLE_CPHD_PATH, backend=backend)
    except ImportError:
        pytest.skip(f"Backend {backend} not available")
        
    m = r.get_metadata()
    ch = r.read_channel(r.get_channel_names()[0])
    sig = ch.signal.to("cuda")
    
    assert torch.cuda.memory_allocated() > 0
    del sig
    del ch
    r.close()
    torch.cuda.empty_cache()
