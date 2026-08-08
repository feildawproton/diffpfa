# diffpfa Audit Report

## Executive Summary
This audit evaluated the `diffpfa` PyTorch-based PFA processor, focusing on mathematical correctness (primary), performance (secondary), and general bugs/gotchas (tertiary). The core mathematical engine is structurally sound and effectively implements PyTorch-based differentiable PFA. However, the evaluation identified a fundamental spatial-variance bug in the CZT implementation, performance bottlenecks due to improper GPU cache handling, and misconceptions in the TODO items regarding NUFFT oversampling.

---

## 1. Mathematical Correctness (Primary Focus)

### 1.1 The `ku_center` Error (Critical Fix)
**Issue:** The project’s `README.md` lists a TODO: "Address the `ku_center` error natively... to reduce strict reliance on subpatch scaling."
In `process_channel_czt`, the cross-range (K_u) resampling uses a fixed start and step spatial frequency derived *only* from the center range bin (`N_samples // 2`):
```python
Ku_center_bin = Ku[:, N_samples // 2]
ku_start = Ku_center_bin[0].unsqueeze(0)
ku_step = ((Ku_center_bin[-1] - Ku_center_bin[0]) / max(N_pulses - 1, 1)).unsqueeze(0)
```
This forces all range bins to share the same cross-range K-space spacing, which causes spatial scaling errors that worsen at the range edges (wavefront curvature). 

**Correction Plan:** 
Since $K_u = K_r \cot \theta$ (or more accurately, $K_u = K_r (P_U / P_R)$), the ratio $K_u / K_r$ is constant for a given pulse (slow-time) across all fast-time samples. We can completely eliminate the `ku_center` error without relying on subpatches by dynamically passing a unique `ku_start` and `ku_step` for *each* resampled Cartesian range bin during the cross-range CZT. 

The `czt_1d_torch` algorithm natively supports batched `k_start` and `k_step`.
Instead of the current implementation, calculate:
```python
# During Cross-Range CZT in pfa_engine.py:
cot_theta = Ku[:, N_samples//2] / Kr[:, N_samples//2]

for b in range(0, M_r, czt_batch_size):
    # Kr_cart is the new uniform Cartesian range grid 
    Kr_cart_b = k_out_start_r + torch.arange(b, b + czt_batch_size, device=self.device) * dK_r
    
    # Calculate exact ku_start and ku_step for each column!
    batch_ku_start = (Kr_cart_b * cot_theta[0]).unsqueeze(1)
    batch_ku_step = (Kr_cart_b * ((cot_theta[-1] - cot_theta[0]) / max(N_pulses - 1, 1))).unsqueeze(1)
    
    batch_out_t = czt_resample_kspace_1d(
        range_batch_t, 
        k_start=batch_ku_start, 
        k_step=batch_ku_step,
        ...
    )
```

### 1.2 NUFFT Oversampling Misunderstanding
**Issue:** The `README.md` TODO asks to "Remove the hardcoded oversample factor (1.5) in the NUFFT algorithm; derive output image grid size directly using LineSpacing...".
**Correction Plan:** This is a misunderstanding of NUFFT mathematics. The 1.5 `oversample` factor determines the size of the *intermediate* padded K-space grid (`M_u`, `M_r`) prior to the IFFT, which prevents circular convolution aliasing caused by the Kaiser-Bessel kernel's roll-off. It does not dictate the output image size (`grid_size_u`, `grid_size_r`). The `oversample` factor should remain (at least 1.5x to 2x).

However, the output target sizes (`N_u`, `N_r` in `_determine_spatial_bounds`) should indeed utilize the CPHD `ReferenceGeometry` if available, rather than natively estimating from bandwidth. Update `_determine_spatial_bounds` to check `getattr(cphd_meta.raw_meta, "ReferenceGeometry", None)` and use its `LineSpacing` and `SampleSpacing` if present.

---

## 2. Performance (Secondary Focus)

### 2.1 Destructive GPU Memory Management
**Issue:** In `pfa_engine.py` (lines 253-255, 288-289, 434-435, 442-444, 464-465), there are repeated calls to `torch.cuda.empty_cache()` inside the core tight processing loops. 
```python
del sig_patch
del Kr
torch.cuda.empty_cache()
```
**Correction Plan:** Remove all `torch.cuda.empty_cache()` calls within the processing loops. `empty_cache()` forces full GPU synchronization and flushes the PyTorch memory allocator cache, making subsequent tensor allocations extremely slow. Merely deleting the tensors (`del sig_patch`) is sufficient for PyTorch's caching allocator to reuse the VRAM immediately for the next batch. 

### 2.2 NUFFT Bottleneck
**Issue:** The PyTorch native NUFFT loops in `diffpfa/algo/nufft_torch.py` (specifically `nufft_2d_type1_torch`) rely on standard Python `for` loops and `grid.index_put_()` with `accumulate=True` over sliding windows (`J x J`). This is severely bottlenecked by PyTorch's CPU overhead for small dispatch operations.
**Correction Plan:** While `torch.compile` helps, creating a custom Triton kernel for the 2D non-uniform scattering would yield a 10x-50x speedup for the NUFFT mode.

---

## 3. Bugs and Gotchas (Tertiary Focus)

### 3.1 Test Suite Failure (`test_mem.py`)
**Issue:** The testing suite is crashing at collection time because `tests/test_mem.py` executes code in the global scope (line 26: `test_mem(SarpyCPHDReader, "SARPY")`) that attempts to load a hardcoded external file (`/home/feildaw/data/...`).
**Correction Plan:** Wrap test executions inside `pytest` test functions and use fixtures for files. For example, `def test_mem_sarpy(): ...`. Do not run processing functions in the global module scope of a test file.

### 3.2 Size Discrepancy between sarpy and sarkit SICDs
**Issue:** A TODO mentions size discrepancies between `sarkit` and `sarpy` SICD outputs.
**Correction Plan:** Both engines write exact `RE32F_IM32F` (complex64) pixel types with matching dimensions. Discrepancies in byte footprint arise because NITF files group data into blocked segments (e.g., 512x512 sub-blocks) and append varying degrees of XML metadata strings. `sarkit` and `sarpy` utilize different default XML layout spacing and different default block sizing strategies. This is expected behavior and not an algorithmic bug.

### 3.3 Complex Arithmetic Promotion Warning
**Issue:** In `channel_combine.py`: `curr_aligned = curr_img * torch.exp(1j * delta_phi)`. 
**Correction Plan:** While recent PyTorch handles the implicit conversion of native Python `complex` (`1j`) against a PyTorch tensor, to ensure strict type promotion (especially under `torch.compile`), it is safer to write: `torch.exp(1j * delta_phi.to(torch.complex64))` or similar native torch complex constructors.
