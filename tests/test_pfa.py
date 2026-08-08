import os
import sys
import numpy as np
import scipy.signal as signal
import torch

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffpfa.algo import (
    PFAConfig,
    PFAEngine,
    align_and_combine_channels,
    compute_kspace,
    compute_look_components,
    compute_look_vectors,
    czt_1d_torch,
    nufft_2d_type1_torch,
)
from diffpfa.io import SarpyCPHDReader, SarpySICDWriter

SAMPLE_CPHD_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"


def test_look_components_linalg():
    P_vecs = torch.tensor([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]], dtype=torch.float64)
    uIAX = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    uIAY = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

    cos_th, sin_th = compute_look_components(P_vecs, uIAX, uIAY)

    # First vector along X -> cos=1, sin=0
    assert torch.isclose(cos_th[0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(sin_th[0], torch.tensor(0.0, dtype=torch.float64))

    # Second vector along Y -> cos=0, sin=1
    assert torch.isclose(cos_th[1], torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(sin_th[1], torch.tensor(1.0, dtype=torch.float64))


def test_czt_1d_accuracy():
    N, M = 64, 128
    fs = 1000.0
    f1, f2 = 100.0, 300.0
    t = np.arange(N) / fs
    x_np = np.sin(2 * np.pi * 200 * t) + np.cos(2 * np.pi * 250 * t)
    x_t = torch.from_numpy(x_np).cdouble()

    # scipy czt
    w = np.exp(-1j * 2 * np.pi * (f2 - f1) / (M * fs))
    a = np.exp(1j * 2 * np.pi * f1 / fs)
    res_sp = signal.czt(x_np, M, w, a)

    # torch czt
    df_sp = (f2 - f1) / M
    f2_endpoint = f1 + df_sp * (M - 1)
    res_th = czt_1d_torch(
        x_t, 
        M=M, 
        r_min=-f1, 
        r_max=-f2_endpoint, 
        k_step=torch.tensor(1.0/fs, dtype=torch.float64), 
        k_start=torch.tensor(0.0, dtype=torch.float64), 
        dim=-1
    ).numpy()

    max_diff = np.max(np.abs(res_sp - res_th))
    assert max_diff < 1e-6


def test_cphd_reader():
    if not os.path.exists(SAMPLE_CPHD_PATH):
        print(f"Skipping: {SAMPLE_CPHD_PATH} not found")
        return

    with SarpyCPHDReader(SAMPLE_CPHD_PATH) as reader:
        meta = reader.get_metadata()
        channels = reader.get_channel_names()

        assert meta.domain_type == "FX"
        assert len(channels) > 0

        ch_data = reader.read_channel(channels[0])
        assert isinstance(ch_data.signal, torch.Tensor)
        assert ch_data.signal.ndim == 2
        assert "SRPPos" in ch_data.pvp


import time

def test_pfa_pipeline_czt(tmp_path):
    if not os.path.exists(SAMPLE_CPHD_PATH):
        print(f"Skipping: {SAMPLE_CPHD_PATH} not found")
        return

    reader = SarpyCPHDReader(SAMPLE_CPHD_PATH)
    writer = SarpySICDWriter()

    # Test CZT mode
    cfg_czt = PFAConfig(
        mode="czt",
        image_area_mode="ImageArea",
        #custom_image_area=(-10.0, 10.0, -10.0, 10.0),
        output_dir=str(tmp_path / "out_czt"),
        device="cuda",
        num_subpatches=2
    )
    engine_czt = PFAEngine(reader, writer, cfg_czt)
    
    start_time = time.time()
    outs_czt = engine_czt.run()
    end_time = time.time()
    
    print(f"\n[TIMING] CZT Mode took {end_time - start_time:.2f} seconds")

    assert len(outs_czt) > 0
    for f in outs_czt:
        assert os.path.exists(f)
        assert os.path.getsize(f) > 0

def test_pfa_pipeline_nufft(tmp_path):
    if not os.path.exists(SAMPLE_CPHD_PATH):
        print(f"Skipping: {SAMPLE_CPHD_PATH} not found")
        return

    reader = SarpyCPHDReader(SAMPLE_CPHD_PATH)
    writer = SarpySICDWriter()

    # Test NUFFT mode
    cfg_nufft = PFAConfig(
        mode="nufft",
        image_area_mode="ImageArea",
        #custom_image_area=(-10.0, 10.0, -10.0, 10.0),
        output_dir=str(tmp_path / "out_nufft"),
        device="cuda",
        num_subpatches=2
    )
    engine_nufft = PFAEngine(reader, writer, cfg_nufft)
    
    start_time = time.time()
    outs_nufft = engine_nufft.run()
    end_time = time.time()
    
    print(f"\n[TIMING] NUFFT Mode took {end_time - start_time:.2f} seconds")

    assert len(outs_nufft) > 0
    for f in outs_nufft:
        assert os.path.exists(f)
        assert os.path.getsize(f) > 0

def test_pfa_pipeline_hybrid(tmp_path):
    if not os.path.exists(SAMPLE_CPHD_PATH):
        print(f"Skipping: {SAMPLE_CPHD_PATH} not found")
        return

    reader = SarpyCPHDReader(SAMPLE_CPHD_PATH)
    writer = SarpySICDWriter()

    cfg_hybrid = PFAConfig(
        mode="hybrid",
        image_area_mode="ImageArea",
        output_dir=str(tmp_path / "out_hybrid"),
        device="cuda",
        num_subpatches=2
    )
    engine_hybrid = PFAEngine(reader, writer, cfg_hybrid)
    
    start_time = time.time()
    outs_hybrid = engine_hybrid.run()
    end_time = time.time()
    
    print(f"\n[TIMING] Hybrid Mode took {end_time - start_time:.2f} seconds")

    assert len(outs_hybrid) > 0
    for f in outs_hybrid:
        assert os.path.exists(f)
        assert os.path.getsize(f) > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-v", __file__]))
