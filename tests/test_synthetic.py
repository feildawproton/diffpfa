import pytest
import torch
import numpy as np
from diffpfa.algo.pfa_engine import PFAEngine, PFAConfig
from diffpfa.io.base import BaseSICDWriter, SICDImagePayload

class MockSICDWriter(BaseSICDWriter):
    def __init__(self):
        self.payloads = []
        
    def write_sicd(self, output_path: str, payload: SICDImagePayload, cphd_meta):
        self.payloads.append(payload)
        return output_path
        
    def finalize(self):
        pass
        
    def close(self):
        pass

@pytest.mark.parametrize("mode", ["czt", "nufft", "hybrid"])
def test_synthetic_pfa_pipeline(synthetic_cphd_reader, pfa_device, mode):
    writer = MockSICDWriter()
    config = PFAConfig(
        mode=mode,
        device=pfa_device,
        num_subpatches=1
    )
    
    engine = PFAEngine(synthetic_cphd_reader, writer, config)
    engine.run()
    
    assert len(writer.payloads) == 1
    payload = writer.payloads[0]
    img = payload.complex_image
    
    # Check that a strong point target formed
    mag = torch.abs(img)
    max_val = mag.max().item()
    mean_val = mag.mean().item()
    
    # Signal-to-noise / peak-to-sidelobe ratio should be high for an ideal target
    assert max_val > 10 * mean_val
