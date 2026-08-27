# diffpfa Audit Report

> **Auditor**: Claude Opus 4.6 (via Google Antigravity)  
> **Date**: 2026-08-27  
> **Scope**: Full codebase audit of `diffpfa/` — mathematical correctness, conciseness, and bugs  
> **Status**: Initial findings (pre-merge with other auditor)  
> **Verification scripts**: All located in `audit/claude_opus_4.6/`

---

## 1. Executive Summary

`diffpfa` is a PyTorch-based Polar Format Algorithm (PFA) SAR image formation processor. It reads CPHD files via `sarkit`, computes K-space geometry, resamples polar K-space to a Cartesian grid using CZT (range) and NUFFT (cross-range), then applies IFFT with Kaiser-Bessel deconvolution to produce SICD-U images.

**Overall assessment**: The core mathematical pipeline is **correct and well-architected**. The CZT, NUFFT gridding, K-space geometry, and multi-channel coherent combination are all mathematically sound. An end-to-end point target test produces a focused image with peak-to-sidelobe ratio >500:1, with the peak at the exact expected pixel. I found **no blocking bugs** but identified several issues of varying severity.

---

## 2. Findings Summary

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| F1 | Medium | Bug | Geodetic altitude formula unstable at geographic poles |
| F2 | Low | Math | Missing I0(beta) normalization in KB deconvolution (constant scale factor) |
| F3 | Medium | Conciseness | Stale test suite referencing removed `diffpfa.io` and `diffpfa.algo` modules |
| F4 | Low | Conciseness | Dead code in `xtra_ideas.py` |
| F5 | Low | Performance | NUFFT kernel evaluates one wasted point outside KB support |
| F6 | Info | Math | IFFT normalization deliberately cancelled (unnormalized DFT output) |
| F7 | Info | Conciseness | SICD UVectECF hardcoded to identity axes, not actual IAX/IAY |
| F8 | Info | Robustness | No domain type support beyond "FX" |
| F9 | Info | Math | Image corners use flat-earth approximation |

---

## 3. Detailed Findings

### F1: Geodetic Altitude Unstable at Poles (Medium)
**File**: `diffpfa/IFP.py` lines 15-27

The `_cartesian_to_geodetic` function computes altitude as:
```python
alt = p / np.cos(lat) - n
```

At the geographic poles, `cos(lat) -> 0`, causing `p/cos(lat)` to diverge. My test confirmed: at the North Pole (0, 0, b), the computed altitude is **-6,399,594 m** instead of ~0 m.

**Impact**: Only affects SICD metadata (LLH coordinates). The image formation math is unaffected. Would produce incorrect altitude in the NITF header for polar-region collects.

**Fix**: Use the Bowring/iterative method, or the standard alternative:
```python
alt = x[2] / np.sin(lat) - n * (1 - e2)  # for |lat| > 45 degrees
```

---

### F2: Missing I0(beta) Normalization in Deconvolution (Low)
**File**: `diffpfa/IFA/PFA.py` lines 27-39

The Kaiser-Bessel deconvolution computes:
```python
deconv = I0(sqrt(beta^2 - (pi*J*u)^2))
```

The standard formula normalizes by `I0(beta)`:
```python
deconv = I0(sqrt(beta^2 - (pi*J*u)^2)) / I0(beta)
```

With beta=13.9086, `I0(beta) ~ 118,509`. This means the image amplitude is scaled by a constant factor of ~1/118,509.

**Impact**: The image shape and point spread function are correct (verified by end-to-end test). Only the absolute amplitude scale is affected. For uncalibrated imagery (which SICD-U is), this is cosmetic. For radiometric calibration, this would need correction.

---

### F3: Stale Test Suite (Medium)
**Files**: `tests/` directory

All test files import from modules that no longer exist in the codebase:
- `diffpfa.io.base` -> `ModuleNotFoundError`
- `diffpfa.algo` -> `ModuleNotFoundError`  
- `diffpfa.algo.pfa_engine` -> `ModuleNotFoundError`
- `diffpfa.io.sarkit_sicd` -> `ModuleNotFoundError`

The current codebase uses `diffpfa.IFP`, `diffpfa.IFA.PFA`, etc. The tests appear to be from an earlier architecture that had separate `io` and `algo` subpackages with a `PFAEngine`/`PFAConfig` pattern and a reader/writer abstraction layer.

**Impact**: No existing test can be run against the current codebase. `conftest.py`, `test_pfa.py`, `test_synthetic.py`, and `test_schema.py` are all broken. Similarly, the `simulation/` scripts import from the old module structure and cannot run.

---

### F4: Dead Code (Low)
**File**: `diffpfa/IFA/channel/xtra_ideas.py`

Contains two unused functions (`_get_image_plane_vectors` and `_apply_scp_shift`) that are never imported anywhere. These appear to be design sketches for future patch-based processing.

---

### F5: Wasted NUFFT Kernel Evaluation (Low)
**File**: `diffpfa/IFA/channel/nufft_torch.py` lines 80-87

The j-offset loop uses `j_offsets = [-3, -2, -1, 0, 1, 2]` (for J=6). For non-integer grid positions, `jx=-3` always produces a distance > J/2 = 3.0, causing the KB kernel to return 0. This wastes one iteration per point.

**Impact**: Performance only. The kernel zeroing ensures correctness despite the wasted computation.

---

### F6: IFFT Normalization Cancellation (Info)
**File**: `diffpfa/IFA/PFA.py` lines 15-22

```python
img = torch.fft.ifft2(grid)    # divides by M_u*M_r (default "backward" norm)
img.mul_(M_u * M_r)            # cancels the normalization
```

This effectively computes an unnormalized DFT sum. Combined with F2, the absolute scale is incorrect but consistent. The approach is valid for image formation where relative magnitudes matter.

---

### F7: Hardcoded UVectECF in SICD Output (Info)
**File**: `diffpfa/IFP.py` lines 218-222

The Row and Col unit vectors in the SICD Grid metadata are hardcoded:
```python
Row: UVectECF = (1,0,0)
Col: UVectECF = (0,1,0)
```

These should be derived from `cphd_meta.uIAX` and `cphd_meta.uIAY`.

**Impact**: SICD metadata consumers (e.g., geolocation tools) would get incorrect grid orientation information. The pixel data itself is unaffected.

---

### F8: Only FX Domain Supported (Info)
**File**: `diffpfa/IFA/kspace.py` lines 61-68

`_compute_fasttime_frequencies` raises `NotImplementedError` for any domain type other than "FX". Fine for the stated use case but worth noting.

---

### F9: Flat-Earth Image Corners (Info)
**File**: `diffpfa/IFP.py` lines 186-207

Image corner lat/lon uses a flat-earth linear approximation. For a 100m image at 45 deg latitude, the error is ~0.001 deg. Acceptable for typical SAR scene sizes.

---

## 4. Mathematical Correctness Summary

### 4.1 CZT (Chirp Z-Transform) - CORRECT
**File**: `diffpfa/IFA/channel/czt_torch.py`

- Bluestein decomposition correctly implemented
- Pre-chirp, convolution kernel, and post-chirp phases all correct
- Verified against brute-force DFT: relative error < 1e-12 (F64)
- Batched mode (per-pulse varying k_start/k_step) is correct
- CZT resampling (double-CZT with conjugation) correctly implements forward + inverse transform
- Normalization by N_spatial is correct

### 4.2 K-space Geometry - CORRECT
**File**: `diffpfa/IFA/kspace.py`

- Look vector computation: P = SRP - 0.5*(Tx + Rx) is correct for monostatic/bistatic
- Direction cosines use 3D magnitude (not 2D projection), giving true cos^2+sin^2 <= 1
- K-space mapping: K_u = (2F/c)*cos(theta), K_r = (2F/c)*sin(theta) is standard PFA
- Fast-time frequencies F(n,k) = SC0[n] + k*SCSS[n] match CPHD spec

### 4.3 NUFFT Gridding - CORRECT
**File**: `diffpfa/IFA/channel/nufft_torch.py`

- Kaiser-Bessel kernel: symmetric, peaks at 0, zero outside support, correctly normalized
- Grid index computation maps k_center to grid center (M/2)
- dK scaling is correct: dK = grid_size / (M * L_x)
- Factored K-space form (cot_theta * Kr_base) is mathematically equivalent

### 4.4 RVP Deskew - CORRECT
**File**: `diffpfa/IFA/channel/pfa_channel.py` lines 13-30

- Phase = pi*F^2/gamma matches the standard RVP correction formula
- Correctly skipped when TxFMRate is absent or zero

### 4.5 Multi-Channel Phase Alignment - CORRECT
**File**: `diffpfa/IFA/PFA.py` lines 136-142

- Phase correction: exp(-j*2*pi*(f_global - f_channel)*delta_tau) is correct
- Reference channel subtraction (tau = RcvTime - ref_RcvTime) correctly implemented

### 4.6 Rotation Detection and Axis Swap - CORRECT
**File**: `diffpfa/IFA/PFA.py` lines 78-91

- Detects when cross-range and range axes are swapped
- Correctly swaps K-space arrays, spatial bounds, pixel spacing, and transposes output

---

## 5. Conciseness Assessment

The codebase is **reasonably concise** for the problem domain:

- **Good**: Separable CZT (range) + NUFFT (cross-range) avoids a full 2D NUFFT
- **Good**: In-place accumulation minimizes GPU memory
- **Good**: Per-channel cleanup with `del` + `empty_cache()` enables large dataset processing
- **Mild concern**: SICD XML construction in `_write_sicd` is ~190 lines of boilerplate

---

## 6. Verification Scripts

All verification scripts are in `audit/`:

| Script | Tests | Results |
|--------|-------|---------|
| `test_czt_correctness.py` | CZT vs brute-force, batched, round-trip, kernel, sign | 5/5 PASS |
| `test_nufft_correctness.py` | KB kernel, gridding, placement, deconv | 5/5 PASS |
| `test_kspace_geometry.py` | K-space, look vectors, RVP, phase correction | 7/7 PASS |
| `test_deep_investigations.py` | Sign convention, scaling, end-to-end | 7/7 PASS |
| `test_edge_cases.py` | Single pulse/sample, geodetic, CZT M=1, corners | 8/8 PASS |

**Total: 32/32 tests pass**

---

## 7. Recommendations (Priority Order)

1. **Fix geodetic altitude** at poles using the standard dual-formula approach (F1)
2. **Update or remove stale tests** — the entire `tests/` directory is non-functional (F3)
3. **Add I0(beta) normalization** to deconvolution for calibration correctness (F2)
4. **Populate SICD UVectECF** from actual image plane vectors (F7)
5. **Clean up dead code** in `xtra_ideas.py` (F4)
6. **Consider** adding end-to-end tests that exercise the current `IFAProcessor` interface
