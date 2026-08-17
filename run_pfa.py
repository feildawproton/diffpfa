import os
import sys
import time
import numpy as np
import scipy.signal as signal
import torch

from diffpfa.algo import PFAConfig, PFAEngine
from diffpfa.io import CPHDReader, SICDWriter


def run_pfa(cphd_path, mode):

    reader = CPHDReader(cphd_path)
    writer = SICDWriter()

    cfg_cztnufft = PFAConfig(
        mode=mode,
        image_area_mode="ImageArea",
        output_dir="workspace/output/"+mode+'/',
        device="cuda",
        num_subpatches=1
    )
    engine_cztnufft = PFAEngine(reader, writer, cfg_cztnufft)
    
    start_time = time.time()
    outs_cztnufft = engine_cztnufft.run()
    end_time = time.time()
    
    print(f"\n[TIMING] {mode}  Mode took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    cphd_path = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"
    #cphd_path = "/home/feildaw/data/2023-09-13-21-18-21_UMBRA-06_CPHD.cphd"
    #cphd_path = "/home/feildaw/data/2023-10-04-02-03-26_UMBRA-04_CPHD.cphd"
    #cphd_path = "/home/feildaw/data/2023-09-11-10-37-05_UMBRA-05_CPHD.cphd"
    
    run_pfa(cphd_path, "cztnufft")
    run_pfa(cphd_path, "nufft")

