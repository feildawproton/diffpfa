import pytest
import torch
import numpy as np
from diffpfa.IFA.PFA import pfa_per_polar, run_diffpfa_tensor, _apply_ifft_and_deconv
from diffpfa.IFA.channel.pfa_channel import process_cztnufft
from diffpfa.types import CPHDMetadata

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def test_differentiable_pixel_to_signal_dense():
    """Certify that gradients flow from formed image pixels back to input signal and are dense (>95%)."""
    N_pulses, N_samples = 16, 32
    N_u, N_r = 16, 16
    L_u, L_r = 50.0, 50.0
    fxc = 9.6e9
    
    cot_theta = torch.linspace(-0.001, 0.001, N_pulses, device=DEV, dtype=torch.float64)
    Kr_radial = torch.linspace(63.0, 65.0, N_samples, device=DEV, dtype=torch.float64).unsqueeze(0).expand(N_pulses, N_samples)
    Ku_radial = cot_theta.unsqueeze(1) * Kr_radial

    sig = torch.randn(N_pulses, N_samples, dtype=torch.complex64, device=DEV, requires_grad=True)

    grid = process_cztnufft(
        signal=sig,
        fxc=fxc,
        pvp={},
        Ku=Ku_radial,
        Kr=Kr_radial,
        N_u=N_u,
        N_r=N_r,
        L_u=L_u,
        L_r=L_r,
        k_ctr_u=0.0,
        k_ctr_r=64.0,
        batch_size=N_r,
        device=DEV
    )
    img = _apply_ifft_and_deconv(grid, N_u, N_r, DEV)

    # Compute loss on image magnitude
    loss = torch.sum(torch.abs(img)**2)
    loss.backward()

    assert sig.grad is not None, "Gradient was not populated"
    assert sig.grad.norm().item() > 0.0, "Gradient norm is zero"
    assert not torch.isnan(sig.grad).any(), "Gradient contains NaN"
    assert (sig.grad != 0).float().mean().item() > 0.95, "Gradient is too sparse (<95% non-zero)"


def test_run_diffpfa_tensor_entrypoint():
    """Certify that run_diffpfa_tensor preserves autograd graph directly from pfa_per_polar."""
    npulse, ns = 16, 32
    fc, bw = 9.6e9, 100e6
    R = 10000.0
    th = np.linspace(-0.01, 0.01, npulse)
    pos = np.stack([R * np.sin(th), -R * np.cos(th), np.zeros_like(th)], 1)
    srp = np.zeros(3)

    pvp = {
        "SRPPos": np.tile(srp, (npulse, 1)),
        "TxPos": pos,
        "RcvPos": pos,
        "TxVel": np.tile([7500.0, 0, 0], (npulse, 1)),
        "RcvVel": np.tile([7500.0, 0, 0], (npulse, 1)),
        "SC0": np.full(npulse, fc - bw / 2),
        "SCSS": np.full(npulse, bw / ns),
        "RcvTime": np.arange(npulse) / 1000.0,
        "TxTime": np.arange(npulse) / 1000.0
    }
    meta = CPHDMetadata(
        domain_type="FX", sgn=-1, global_fx_min=fc - bw / 2, global_fx_max=fc + bw / 2,
        iarp_ecf=srp, uIAX=np.array([1.0, 0, 0]), uIAY=np.array([0, 1.0, 0]),
        ref_ch_id="0", image_area=None, extended_area=None, collection_start=None,
        radar_mode="SPOTLIGHT", classification="U", srp_ecf=srp,
        arp_pos_coa=pos[npulse // 2], arp_vel_coa=np.array([7500.0, 0, 0]),
        side_of_track="R", line_spacing=None, sample_spacing=None, raw_meta=None
    )

    sig = torch.randn(npulse, ns, dtype=torch.complex64, device=DEV, requires_grad=True)
    img_t, bw_r, bw_u, N_r, N_u, _ = run_diffpfa_tensor(sig, pvp, meta, fc, L=20.0, device=DEV)

    assert isinstance(img_t, torch.Tensor)
    assert img_t.requires_grad

    loss = torch.sum(torch.abs(img_t)**2)
    loss.backward()

    assert sig.grad is not None
    assert sig.grad.norm().item() > 0.0
    assert not torch.isnan(sig.grad).any()


def test_torch_chain_adjoint_consistency():
    """Certify linear operator adjoint consistency: <Ax, y> = <x, A*y> to high precision."""
    npulse, ns = 16, 24
    fc, bw = 9.6e9, 60e6
    R = 10000.0
    th = np.linspace(-0.005, 0.005, npulse)
    pos = np.stack([R * np.sin(th), -R * np.cos(th), np.zeros_like(th)], 1)
    srp = np.zeros(3)

    pvp = {
        "SRPPos": np.tile(srp, (npulse, 1)),
        "TxPos": pos,
        "RcvPos": pos,
        "TxVel": np.tile([7500.0, 0, 0], (npulse, 1)),
        "RcvVel": np.tile([7500.0, 0, 0], (npulse, 1)),
        "SC0": np.full(npulse, fc - bw / 2),
        "SCSS": np.full(npulse, bw / ns),
        "RcvTime": np.arange(npulse) / 1000.0,
        "TxTime": np.arange(npulse) / 1000.0
    }
    meta = CPHDMetadata(
        domain_type="FX", sgn=-1, global_fx_min=fc - bw / 2, global_fx_max=fc + bw / 2,
        iarp_ecf=srp, uIAX=np.array([1.0, 0, 0]), uIAY=np.array([0, 1.0, 0]),
        ref_ch_id="0", image_area=None, extended_area=None, collection_start=None,
        radar_mode="SPOTLIGHT", classification="U", srp_ecf=srp,
        arp_pos_coa=pos[npulse // 2], arp_vel_coa=np.array([7500.0, 0, 0]),
        side_of_track="R", line_spacing=None, sample_spacing=None, raw_meta=None
    )

    a = torch.randn(npulse, ns, dtype=torch.complex64, device=DEV)
    with torch.no_grad():
        img_a, *_ = run_diffpfa_tensor(a, pvp, meta, fc, L=15.0, device=DEV)
        w = torch.randn(img_a.shape, dtype=torch.complex64, device=DEV)

    x = a.clone().requires_grad_(True)
    img_x, *_ = run_diffpfa_tensor(x, pvp, meta, fc, L=15.0, device=DEV)
    (torch.conj(w) * img_x).real.sum().backward()

    with torch.no_grad():
        lhs = (torch.conj(w) * img_a).sum()
        rhs = (torch.conj(x.grad) * a).sum()
    rel_err = abs(lhs - rhs) / abs(lhs)
    assert rel_err < 1e-4, f"Adjoint error too large: {rel_err}"
