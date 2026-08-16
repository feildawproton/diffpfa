import os
import sys
import time
import numpy as np
import scipy.signal as signal
import torch

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffpfa.algo import PFAConfig, PFAEngine, align_and_combine_channels

from diffpfa.algo.channel.geometry_channel import compute_kspace, compute_look_vectors
from diffpfa.algo.channel.patch.geometry_patch import compute_look_components
from diffpfa.algo.channel.patch.czt_torch import czt_1d_torch
from diffpfa.algo.channel.patch.nufft_torch import nufft_2d_type1_torch

from diffpfa.io import CPHDReader, SICDWriter

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
        pytest.skip("Test file not found")

    with CPHDReader(SAMPLE_CPHD_PATH) as reader:
        meta = reader.get_metadata()
        channels = reader.get_channel_names()

        assert meta.domain_type == "FX"
        assert len(channels) > 0

        ch_data = reader.read_channel(channels[0])
        assert isinstance(ch_data.signal, torch.Tensor)
        assert ch_data.signal.ndim == 2
        assert "SRPPos" in ch_data.pvp




def test_pfa_pipeline_nufft(tmp_path):
    if not os.path.exists(SAMPLE_CPHD_PATH):
        pytest.skip("Test file not found")

    reader = CPHDReader(SAMPLE_CPHD_PATH)
    writer = SICDWriter()

    # Test NUFFT mode
    cfg_nufft = PFAConfig(
        mode="nufft",
        image_area_mode="ImageArea",
        #custom_image_area=(-10.0, 10.0, -10.0, 10.0),
        output_dir="output/out_nufft",
        device="cuda",
        num_subpatches=1
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

def test_pfa_pipeline_cztnufft(tmp_path):
    if not os.path.exists(SAMPLE_CPHD_PATH):
        pytest.skip("Test file not found")

    reader = CPHDReader(SAMPLE_CPHD_PATH)
    writer = SICDWriter()

    cfg_cztnufft = PFAConfig(
        mode="cztnufft",
        image_area_mode="ImageArea",
        output_dir="output/out_cztnufft",
        device="cuda",
        num_subpatches=1
    )
    engine_cztnufft = PFAEngine(reader, writer, cfg_cztnufft)
    
    start_time = time.time()
    outs_cztnufft = engine_cztnufft.run()
    end_time = time.time()
    
    print(f"\n[TIMING] CZTNUFFT Mode took {end_time - start_time:.2f} seconds")

    assert len(outs_cztnufft) > 0
    for f in outs_cztnufft:
        assert os.path.exists(f)
        assert os.path.getsize(f) > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-v", __file__]))
