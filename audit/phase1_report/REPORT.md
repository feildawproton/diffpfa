# Phase 1: Independent Audit Report

## 1. Mathematical Correctness Issues

### A. Subband Phase Alignment
In `diffpfa/algo/channel_combine.py`, the `align_and_combine_channels` function attempts to calculate the phase offset between channels by calculating the zero-lag cross-correlation (inner product) in K-space:
```python
cross_corr = torch.sum(ref_img * torch.conj(curr_img))
delta_phi = torch.angle(cross_corr)
```
By Plancherel's theorem, this inner product is mathematically identical to the spatial domain inner product. However, stepped-chirp subbands occupy *disjoint* frequency supports in K-space (e.g. Band 1: 9.875 - 10.125 GHz, Band 2: 10.125 - 10.375 GHz). Because they are orthogonal functions, their inner product evaluates to exactly zero. Taking `torch.angle()` of 0 (or computational noise) results in random phase rotations, which completely ruins the coherent combination and prevents the expected halving of the spatial impulse response width.

### B. Kaiser-Bessel Deconvolution
The mathematical formulation for the spatial-domain deconvolution of the Kaiser-Bessel window is incorrect across `nufft_1d_type1_torch`, `nufft_2d_type1_torch`, and `pfa_engine.py`.
The analytic Fourier transform of the Kaiser-Bessel window $I_0(\beta \sqrt{1 - (2k/J)^2})$ is proportional to $\sinh(z)/z$ where $z = \sqrt{\beta^2 - (\pi J x)^2}$. 
The implemented code incorrectly divides by $I_0(\sqrt{\beta^2 - (\pi J x)^2})$ or even by the spatial-domain Kaiser-Bessel shape itself. This distorts the spatial amplitude of the formed image.

## 2. Software Bugs
- **Missing Import:** `tests/test_pfa.py` crashes because it uses `time.time()` but lacks `import time`.
- **Simulation Crash:** `simulation/simulate_stepped_chirp.py` attempts to read debug channel payloads from the writer. However, in `diffpfa/algo/pfa_engine.py`, when `combine_in_kspace=True` (the default for `num_subpatches=1`), it explicitly skips writing these debug channels. This causes a `StopIteration` error in the simulation script.

## 3. Performance / Memory Bottlenecks
- **Memory Spikes during Channel Combination:** In `align_and_combine_channels`, all aligned channel tensors are accumulated into a list (`aligned_images.append(curr_aligned)`), and the combined image is created using `.clone()`. For multi-gigabyte complex K-space grids, storing all channels simultaneously in GPU VRAM will result in Out-Of-Memory (OOM) errors. This should be a lazy or in-place accumulation.
