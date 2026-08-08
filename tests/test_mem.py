from diffpfa.algo import PFAConfig, PFAEngine
from diffpfa.io.sarpy_cphd import SarpyCPHDReader
from diffpfa.io.sarkit_cphd import SarkitCPHDReader
import torch

cphd_path = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"

def test_mem(ReaderClass, name):
    torch.cuda.empty_cache()
    print(f"\n--- Testing {name} ---")
    r = ReaderClass(cphd_path)
    m = r.get_metadata()
    print(f"Memory after metadata: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    ch = r.read_channel(r.get_channel_names()[0])
    print(f"Memory after read_channel: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    sig = ch.signal.to("cuda")
    print(f"Memory after sig.to(cuda): {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    del sig
    del ch
    r.close()
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

test_mem(SarpyCPHDReader, "SARPY")
test_mem(SarkitCPHDReader, "SARKIT")
