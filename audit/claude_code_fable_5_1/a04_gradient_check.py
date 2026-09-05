"""
a04_gradient_check.py -- Is the image differentiable w.r.t. the signal, and is the gradient right?

Run from repo root:
    /home/feildaw/mypyenv/bin/python audit/claude_code_fable_5_1/a04_gradient_check.py

Three questions, answered separately:
 (1) API level: does pfa_per_polar itself carry a gradient?  (It takes / returns numpy.)
 (2) Kernel level: build the exact torch chain pfa_per_polar executes
        signal -> phase corr -> process_cztnufft -> accumulate -> _apply_ifft_and_deconv
     with a leaf tensor as input, check requires_grad propagates, and compare the analytic
     gradient of a real loss against central finite differences (torch.autograd.gradcheck in
     complex128, and a hand-rolled directional derivative check in the complex64 path the
     library actually uses).
 (3) Linearity: the whole chain is linear in the signal, so J s must equal image(s) exactly and
     the adjoint must satisfy <J s, y> = <s, J^H y>.  A wrong adjoint (e.g. a missing conj in a
     custom op) shows up here even when gradcheck's tolerances are loose.
Nothing is asserted; everything is printed and saved.
"""
import os, sys, json, math
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from diffpfa.IFA.PFA import pfa_per_polar, _apply_ifft_and_deconv
from diffpfa.IFA.channel.pfa_channel import process_cztnufft
from diffpfa.IFA.kspace import compute_kspace
from diffpfa.constants import SPEED_OF_LIGHT as C
from scipy.fft import next_fast_len

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a02_forward_model_vs_exact import make_collection

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def torch_chain(sig_t, pvp, meta, fxc, L, oversample=1.25, batch_size=64, dtype=torch.complex64):
    """Exactly what pfa_per_polar does for one channel, but with a torch tensor in / torch tensor out."""
    dev = sig_t.device
    Ku, Kr = compute_kspace(pvp, meta.uIAX, meta.uIAY, sig_t.shape[1], "FX", device=dev)
    gku_min, gku_max = Ku.min().item(), Ku.max().item()
    gkr_min, gkr_max = Kr.min().item(), Kr.max().item()
    gku_ctr, gkr_ctr = (gku_min + gku_max) / 2, (gkr_min + gkr_max) / 2
    bw_u, bw_r = gku_max - gku_min, gkr_max - gkr_min
    du, dr = 1 / (bw_u * oversample), 1 / (bw_r * oversample)
    N_u, N_r = next_fast_len(int(np.round(L / du))), next_fast_len(int(np.round(L / dr)))
    tau = pvp["RcvTime"] - pvp["RcvTime"]
    fc_global = (meta.global_fx_min + meta.global_fx_max) / 2
    phase = -2 * torch.pi * (fc_global - fxc) * torch.as_tensor(tau, dtype=torch.float64, device=dev)
    sig = sig_t * torch.exp(1j * phase).unsqueeze(1).to(sig_t.dtype)
    grid = process_cztnufft(signal=sig, fxc=fxc, pvp=pvp, Ku=Ku, Kr=Kr, N_u=N_u, N_r=N_r, L_u=L, L_r=L,
                            k_ctr_u=gku_ctr, k_ctr_r=gkr_ctr, batch_size=batch_size, device=dev)
    comb = grid.clone()
    img = _apply_ifft_and_deconv(comb, N_u, N_r, dev)
    return img


def main():
    out = {}
    fc, bw = 9.6e9, 600e6
    # small case so finite differences are affordable
    sig_np, pvp, meta, *_ = make_collection(fc, bw, ns=24, npulse=16, span_deg=2.0, targets=[(1.0, -2.0, 1.0)])
    L = 12.0

    # ---------------------------------------------------------------- (1) API level
    print("[1] API-level differentiability of pfa_per_polar")
    print("    signature takes List[np.ndarray] and returns np.ndarray (img_cpu = combined_img.cpu().numpy()).")
    print("    -> no autograd graph can cross this boundary; the caller must use the torch kernels directly.")
    out["api_differentiable"] = False

    # ---------------------------------------------------------------- (2) kernel level
    print("[2] kernel-level: requires_grad propagation and gradcheck")
    for dtype, name in [(torch.complex64, "complex64 (library path)"), (torch.complex128, "complex128")]:
        s = torch.tensor(sig_np, dtype=dtype, device=DEV, requires_grad=True)
        img = torch_chain(s, pvp, meta, fc, L, dtype=dtype)
        print(f"    {name}: image {tuple(img.shape)} dtype {img.dtype} requires_grad={img.requires_grad} grad_fn={type(img.grad_fn).__name__ if img.grad_fn else None}")
        loss = (img.abs() ** 2).sum()
        loss.backward()
        g = s.grad
        print(f"      d(sum|img|^2)/d(signal): finite={torch.isfinite(g).all().item()}  |g|max={g.abs().max().item():.4g}  nonzero fraction={(g.abs()>0).float().mean().item():.3f}")
        out[f"requires_grad_{name}"] = bool(img.requires_grad)

    # gradcheck in double precision on a real-valued loss with a random projection (keeps it cheap)
    torch.manual_seed(1)
    s64 = torch.tensor(sig_np, dtype=torch.complex128, device=DEV, requires_grad=True)
    y = torch.randn(1, dtype=torch.complex128, device=DEV)  # placeholder to size w below
    with torch.no_grad():
        img0 = torch_chain(s64, pvp, meta, fc, L, dtype=torch.complex128)
    w = torch.randn(img0.shape, dtype=torch.complex128, device=DEV)

    def f(x):
        im = torch_chain(x, pvp, meta, fc, L, dtype=torch.complex128)
        return (im * w).real.sum() + (im.abs() ** 2).sum() * 1e-3

    ok = torch.autograd.gradcheck(f, (s64,), eps=1e-6, atol=1e-4, rtol=1e-3, fast_mode=True)
    print(f"    torch.autograd.gradcheck (complex128, fast_mode): {ok}")
    out["gradcheck_complex128"] = bool(ok)

    # hand-rolled directional derivative in the library's complex64 path
    s32 = torch.tensor(sig_np, dtype=torch.complex64, device=DEV, requires_grad=True)
    w32 = w.to(torch.complex64)
    def f32(x):
        im = torch_chain(x, pvp, meta, fc, L, dtype=torch.complex64)
        return (im * w32).real.sum()
    val = f32(s32); val.backward()
    g32 = s32.grad.detach().clone()
    torch.manual_seed(2)
    d = torch.randn_like(s32); d = d / d.abs().max()
    # torch convention: for real f of complex z, grad = df/dRe + j df/dIm ; directional derivative along d = Re<grad, d> ... = sum(Re(conj(grad) * d))
    analytic = torch.sum((g32.conj() * d).real).item()
    rel = []
    for eps in (1e-1, 1e-2, 1e-3):
        with torch.no_grad():
            fp = f32(s32 + eps * d).item(); fm = f32(s32 - eps * d).item()
        fd = (fp - fm) / (2 * eps)
        rel.append((eps, fd, analytic, abs(fd - analytic) / max(abs(analytic), 1e-12)))
    print("    complex64 directional derivative (central FD vs autograd):")
    for eps, fd, an, r in rel:
        print(f"      eps={eps:.0e}  FD={fd:+.6e}  autograd={an:+.6e}  rel err={r:.2e}")
    out["complex64_directional"] = [dict(eps=e, fd=fd, autograd=an, rel_err=r) for e, fd, an, r in rel]

    # ---------------------------------------------------------------- (3) linearity + adjoint test
    print("[3] linearity and adjoint consistency (complex128)")
    with torch.no_grad():
        a = torch.randn(sig_np.shape, dtype=torch.complex128, device=DEV)
        b = torch.randn(sig_np.shape, dtype=torch.complex128, device=DEV)
        Ja = torch_chain(a, pvp, meta, fc, L, dtype=torch.complex128)
        Jb = torch_chain(b, pvp, meta, fc, L, dtype=torch.complex128)
        Jab = torch_chain(0.3 * a + 0.7j * b, pvp, meta, fc, L, dtype=torch.complex128)
        lin = (Jab - (0.3 * Ja + 0.7j * Jb)).abs().max().item() / Jab.abs().max().item()
    print(f"    linearity residual |J(0.3a+0.7jb) - (0.3Ja+0.7jJb)| / max = {lin:.2e}")
    # adjoint via autograd: for f(x) = Re<y, Jx> = Re sum(conj(y) * Jx), grad_x = J^H y (torch returns conj-Wirtinger*2? check numerically)
    yv = torch.randn(Ja.shape, dtype=torch.complex128, device=DEV)
    x = a.clone().requires_grad_(True)
    fval = (torch.conj(yv) * torch_chain(x, pvp, meta, fc, L, dtype=torch.complex128)).real.sum()
    fval.backward()
    JHy = x.grad
    lhs = (torch.conj(yv) * Ja).sum()          # <y, J a>
    rhs = (torch.conj(JHy) * a).sum()          # <J^H y, a>
    print(f"    <y,Ja> = {lhs.item():.6e}\n    <J^H y,a> = {rhs.item():.6e}\n    |diff|/|lhs| = {abs(lhs.item()-rhs.item())/abs(lhs.item()):.2e}")
    out["linearity_resid"] = lin
    out["adjoint_rel_diff"] = abs(lhs.item() - rhs.item()) / abs(lhs.item())

    # ---------------------------------------------------------------- (4) inference_mode / no_grad hazards in the codebase
    print("[4] static scan for gradient-killing constructs in diffpfa/ (excluding metadata-only code paths)")
    import re, glob
    pat = re.compile(r"\.detach\(|\.item\(|torch\.no_grad|inference_mode|\.numpy\(|from_numpy|\.cpu\(\)")
    for p in sorted(glob.glob("diffpfa/**/*.py", recursive=True)) + ["run_pfa.py"]:
        for i, line in enumerate(open(p), 1):
            if pat.search(line):
                print(f"    {p}:{i}: {line.strip()}")
    with open(os.path.join(HERE, "out", "a04_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
