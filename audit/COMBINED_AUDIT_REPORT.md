# Combined Audit Report — diffpfa

> **Auditors**:  
> - Claude Opus 4.6 (via Google Antigravity)  
> - Gemini 3.1 Pro (via Google Antigravity)  
> **Date**: 2026-08-27  
> **Scope**: Full codebase audit — mathematical correctness, conciseness, and bugs

---

## 1. Executive Summary

`diffpfa` is a PyTorch-based Polar Format Algorithm (PFA) SAR image formation processor that reads CPHD files, computes K-space geometry, resamples polar K-space to a Cartesian grid using CZT (range) + NUFFT (cross-range), and applies IFFT with Kaiser-Bessel deconvolution to produce SICD-U images.

Both auditors independently concluded that the **core mathematical architecture is correct and well-designed**. The hybrid CZT/NUFFT approach is efficient, the in-place GPU memory management is best-practice, and the asymmetric 1D deconvolution (only on the NUFFT axis) is mathematically justified.

However, we identified **one critical bug** (RVP deskew), **one significant metadata bug** (SICD grid vectors), and several lower-severity issues. The critical RVP bug only triggers on stretch-processed data (when `TxFMRate` is nonzero) — matched-filter data is unaffected.

---

## 2. Consolidated Findings

### Priority 1 — Critical / High

| ID | Finding | Auditor | Severity |
|----|---------|---------|----------|
| **C1** | RVP deskew uses absolute RF frequency instead of baseband video frequency | Gemini 3.1 Pro (confirmed by Claude Opus 4.6) | 🔴 Critical |
| **C2** | SICD UVectECF hardcoded to identity axes, not actual uIAX/uIAY | Both auditors independently | 🟠 High |

### Priority 2 — Medium

| ID | Finding | Auditor | Severity |
|----|---------|---------|----------|
| **M1** | Geodetic altitude formula diverges at geographic poles | Claude Opus 4.6 | 🟡 Medium |
| **M2** | Entire `tests/` directory references nonexistent modules (`diffpfa.io`, `diffpfa.algo`) | Claude Opus 4.6 | 🟡 Medium |

### Priority 3 — Low / Informational

| ID | Finding | Auditor | Severity |
|----|---------|---------|----------|
| **L1** | Missing I₀(β) normalization in KB deconvolution (constant scale factor ~118,509×) | Claude Opus 4.6 | 🟢 Low |
| **L2** | Dead code in `xtra_ideas.py` | Claude Opus 4.6 | 🟢 Low |
| **L3** | NUFFT kernel evaluates one wasted point outside KB support per gridding point | Claude Opus 4.6 | 🟢 Low |
| **L4** | IFFT normalization deliberately cancelled (unnormalized DFT output) | Claude Opus 4.6 | 🟢 Info |
| **L5** | Only FX domain type supported (TOA raises NotImplementedError) | Claude Opus 4.6 | 🟢 Info |
| **L6** | Image corners use flat-earth approximation | Claude Opus 4.6 | 🟢 Info |
| **L7** | Direct ground-plane PFA inherently produces skewed IPRs vs. slant-plane reference | Gemini 3.1 Pro | 🟢 Info (not a bug) |

---

## 3. Critical Finding Details

### C1: RVP Deskew Uses Absolute RF Frequency 🔴

**File**: `diffpfa/IFA/channel/pfa_channel.py`, lines 26–27  
**Found by**: Gemini 3.1 Pro | **Confirmed by**: Claude Opus 4.6 (independent mathematical derivation + numerical verification)

**The bug:**
```python
# Current code (WRONG for stretch-processed data)
F_hz = sc0.unsqueeze(1) + scss.unsqueeze(1) * k_idx.unsqueeze(0)
rvp_phase = torch.pi * (F_hz ** 2) / gamma.unsqueeze(1)
```

**The problem:** In stretch processing, the RVP to be cancelled is π·f_v²/γ where f_v is the *baseband video frequency* (f_v = F_hz − F_center), NOT the absolute RF frequency F_hz (~10 GHz).

Expanding (F_center + f_v)² = F_center² + 2·F_center·f_v + f_v², the erroneous term 2·F_center·f_v introduces a **linear phase ramp** in frequency, which is equivalent to a **range shift** in the spatial domain:

```
Δrange = c · F_center / (2γ)
```

For typical values (F_center=10 GHz, γ=5×10¹² Hz/s): **Δrange ≈ 300 km**. This completely destroys the image.

**Evidence**: Gemini's `sim_ipr_rvp.py` produces a completely defocused image with grating lobes across the entire range extent. Claude's `verify_gemini_rvp_finding.py` confirms the linear phase slope matches the predicted 2π·F_center/γ to machine precision.

**Mitigating factor**: The code only triggers RVP deskew when `TxFMRate` is present and nonzero in the PVP data. Matched-filter processed data (TxFMRate absent or zero) is **not affected**.

**Fix** (requires passing `fxc` to `_deskew_rvp`):
```python
f_v = F_hz - fxc   # baseband video frequency
rvp_phase = torch.pi * (f_v ** 2) / gamma.unsqueeze(1)
```

---

### C2: Hardcoded SICD Grid Unit Vectors 🟠

**File**: `diffpfa/IFP.py`, lines 219–222  
**Found by**: Both auditors independently

**The bug:**
```python
if dir_name == "Row":
    sub(uv, "X", "1.0"); sub(uv, "Y", "0.0"); sub(uv, "Z", "0.0")
else:
    sub(uv, "X", "0.0"); sub(uv, "Y", "1.0"); sub(uv, "Z", "0.0")
```

**Impact**: The image pixels are correctly oriented to uIAX/uIAY during formation, but the SICD metadata tells downstream exploitation tools the grid is aligned to X/Y axes. This causes geolocation tools and viewers to project the image onto the wrong geographic axes, producing heavily sheared/skewed imagery. As Gemini noted, this is the primary cause of any observed "skewed IPR" behavior when comparing against reference tools.

**Fix:**
```python
if dir_name == "Row":
    sub(uv, "X", str(cphd_meta.uIAX[0]))
    sub(uv, "Y", str(cphd_meta.uIAX[1]))
    sub(uv, "Z", str(cphd_meta.uIAX[2]))
else:
    sub(uv, "X", str(cphd_meta.uIAY[0]))
    sub(uv, "Y", str(cphd_meta.uIAY[1]))
    sub(uv, "Z", str(cphd_meta.uIAY[2]))
```

---

### M1: Geodetic Altitude Unstable at Poles 🟡

**File**: `diffpfa/IFP.py`, lines 15–27  
**Found by**: Claude Opus 4.6

The altitude formula `alt = p / cos(lat) - N` diverges at the poles where cos(lat) → 0. Verified: North Pole altitude computes as **−6,399,594 m** instead of ~0 m. Equatorial positions are accurate.

**Impact**: Only affects NITF header LLH coordinates for polar-region collects. Image formation is unaffected.

---

### M2: Stale Test Suite 🟡

**File**: `tests/` directory  
**Found by**: Claude Opus 4.6

All test files import from `diffpfa.io`, `diffpfa.algo`, `diffpfa.io.base`, etc., which no longer exist. The codebase was restructured to use `diffpfa.IFP`/`diffpfa.IFA` but the tests were not updated. No test can currently execute.

---

## 4. Mathematical Correctness — Agreed Assessment

Both auditors independently verified the following components and agree on their correctness:

| Component | Verdict | Notes |
|-----------|---------|-------|
| CZT (Chirp Z-Transform) | ✅ Correct | Bluestein decomposition, batched mode, resampling all verified to 1e-12 relative error |
| K-space geometry | ✅ Correct | Look vectors, direction cosines, frequency mapping all match PFA standard formulation |
| NUFFT gridding | ✅ Correct | Kaiser-Bessel kernel properties, grid index computation, dK scaling all verified |
| 1D-only deconvolution | ✅ Correct | Both auditors independently confirmed: only the NUFFT axis (cross-range) uses KB gridding, so only that axis needs deconvolution. "Remarkably concise and clever" — Gemini |
| Multi-channel phase alignment | ✅ Correct | exp(−j·2π·(f_global − f_ch)·Δτ) is standard |
| Rotation detection + axis swap | ✅ Correct | Properly detects and handles swapped range/cross-range axes |
| RVP deskew (matched filter data) | ✅ Correct | Correctly bypassed when TxFMRate is absent or zero |
| RVP deskew (stretch-processed data) | ❌ Bug C1 | Uses F_RF instead of f_video — see C1 above |
| Ground-plane IPR skew | ✅ Physically correct | Inherent consequence of direct ground-plane PFA; not a bug (Gemini analysis) |
| End-to-end point target | ✅ Correct | PSR > 500:1, peak at exact center pixel (Claude verification) |

---

## 5. Architecture & Conciseness — Agreed Assessment

Both auditors praise the architecture:

- **Hybrid CZT + NUFFT** avoids a costly 2D NUFFT by exploiting the separability of polar-to-Cartesian resampling
- **In-place operations** (`mul_`, `div_`, `add_`) and eager `del` + `empty_cache()` minimize GPU VRAM footprint
- **Threaded I/O** with `ThreadPoolExecutor` decouples disk reads from GPU processing
- **Sequential channel accumulation** enables processing datasets larger than GPU memory
- SICD XML generation is ~190 lines of boilerplate (minor conciseness concern, not actionable)

---

## 6. Verification Artifacts

### Claude Opus 4.6 (`audit/claude_opus_4.6/`)
| Script | Tests | Result |
|--------|-------|--------|
| `test_czt_correctness.py` | CZT vs brute-force, batched, round-trip, kernel symmetry, sign convention | 5/5 PASS |
| `test_nufft_correctness.py` | KB kernel properties, gridding, placement, deconv formula, deconv dimension | 5/5 PASS |
| `test_kspace_geometry.py` | K-space broadside/cross-range, look vectors, direction cosines, RVP, phase correction, frequencies | 7/7 PASS |
| `test_deep_investigations.py` | CZT sign convention, KB distance, dK scaling, CZT normalization, cot_theta, J-offset, end-to-end | 7/7 PASS |
| `test_edge_cases.py` | Single pulse/sample, geodetic poles/origin, CZT M=1, IFFT scale, corners, rotation | 8/8 PASS |
| `verify_gemini_rvp_finding.py` | Independent cross-verification of Gemini's RVP bug | Confirmed |

### Gemini 3.1 Pro (`audit/gemini_3.1_pro_findings/`)
| Script | Purpose | Key Result |
|--------|---------|------------|
| `sim_ipr.py` | Ground-plane IPR (no RVP) | Clean focused point target ✅ |
| `sim_ipr_rvp.py` | IPR with RVP bug active | Complete defocus, grating lobes ❌ |
| `check_ipr_metrics.py` | Quantitative peak comparison ± RVP | Confirms shift/defocus |
| `check_skewness.py` | Ground vs slant plane 2nd-moment analysis | Confirms inherent ground-plane skew |
| `check_skewness_rotated.py` | Skewness with rotated flight path | Confirms skew scales with heading misalignment |
| `check_skew.py` | K-space support visualization | Shows parallelogram-shaped K-space on ground plane |

---

## 7. Recommended Actions (Priority Order)

1. **🔴 Fix RVP deskew** — Change `F_hz` to `f_v = F_hz - fxc` in `_deskew_rvp`. Requires threading `fxc` through to `process_cztnufft` → `_deskew_rvp`. (C1)

2. **🟠 Fix SICD UVectECF** — Propagate `cphd_meta.uIAX`/`uIAY` into the SICD Grid metadata instead of hardcoding identity axes. (C2)

3. **🟡 Fix geodetic altitude** — Use dual-formula approach for altitude (switch to Z/sin(lat) near poles). (M1)

4. **🟡 Update test suite** — Either update tests to import from current `diffpfa.IFP`/`diffpfa.IFA` module structure, or remove the stale tests and replace with new ones. (M2)

5. **🟢 Add I₀(β) normalization** to deconvolution for radiometric calibration correctness. (L1)

6. **🟢 Clean up** dead code in `xtra_ideas.py`. (L2)
