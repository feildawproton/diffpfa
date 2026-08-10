# Auditor Report (Phases 1 & 2)

**Auditor:** Antigravity (Gemini 3.1 Pro)
**Date:** 2026-08-09
**Role:** Mathematical and Performance Audit


## Phase 1: Independent Audit (Mathematical & Performance Correctness)

### 1. Mathematical Correctness Issues

**A. Subband Phase Alignment (Critical)**
In `diffpfa/algo/channel_combine.py`, `align_and_combine_channels` calculates the phase offset by taking the zero-lag cross-correlation (inner product) in K-space:
`cross_corr = torch.sum(ref_img * torch.conj(curr_img))`
While Parseval's theorem equating the inner products is correct, for *disjoint* stepped-chirp subbands, this inner product evaluates to exactly zero. Taking `torch.angle(0)` yields computational noise, which applies a random phase rotation and breaks coherent combination. I verified this mathematically in a test script.

**B. Kaiser-Bessel Deconvolution (High)**
The spatial-domain deconvolution of the Kaiser-Bessel window is incorrect across `nufft_1d_type1_torch`, `nufft_2d_type1_torch`, and `pfa_engine.py`. 
The analytic Fourier transform of the Kaiser-Bessel window $I_0(\beta \sqrt{1 - (2k/J)^2})$ is proportional to $\sinh(z)/z$ where $z = \sqrt{\beta^2 - (\pi J x)^2}$. The code incorrectly divides by the $I_0$ function, distorting the spatial amplitude.

### 2. Software Bugs
- **Missing Import:** `tests/test_pfa.py` crashes due to a missing `import time`.
- **Simulation Crash:** `simulation/simulate_stepped_chirp.py` expects debug channel SICDs to be generated to measure resolution. However, in `pfa_engine.py`, when `combine_in_kspace=True`, writing these debug channels is skipped, causing a `StopIteration` crash.

### 3. Memory Bottleneck
- **VRAM Spikes During Channel Combination:** `align_and_combine_channels` stores all aligned channels in a list (`aligned_images.append(curr_aligned)`), and `PFAEngine.run()` holds all K-space grids in `channel_images`. For multi-channel large datasets, storing $2N$ full grids simultaneously will cause VRAM OOM errors. 

---

## Phase 2: Reconciliation with PI's Performance Audit

### 1. Intra-Channel PCIe Transfer & Compute Overlap
**PI's Claim:** Overlapping H2D transfers with cuFFT compute cuts pipeline duration by 56%.
**Status:** The current code in `process_channel_czt_nufft` still contains the dual-stream CUDA logic for *intra-channel* chunked H2D copies. However, as noted, the PI loads all channels simultaneously into host memory upfront in `PFAEngine.run()` using a `ThreadPoolExecutor`. The documentation needs an update to reflect the current multi-channel memory loading strategy versus the intra-channel streaming optimization.

### 2. K-Space Sub-Channel Combination
**PI's Claim:** Combining channels in K-space avoids redundant 2D IFFTs, achieves in-place alignment (using Parseval's theorem), and sidesteps memory allocations.
**Status:** 
- **Math Discrepancy:** As noted in Phase 1, the Parseval's theorem cross-correlation approach is mathematically flawed for disjoint stepped-chirp subbands and results in phase noise.
- **Memory Discrepancy:** The code does not perform in-place accumulation. It accumulates the full K-space grids into lists (`channel_images` and `aligned_images`), meaning large memory allocations were not sidestepped, but rather moved to K-space.
- **Test Discrepancy:** This optimization explicitly skips the debug outputs relied upon by `simulate_stepped_chirp.py`.

The documentation regarding these K-space optimizations needs to be updated or the codebase patched to reflect true in-place accumulation and correct disjoint-band alignment.
