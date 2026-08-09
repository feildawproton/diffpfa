# Phase 3 Audit Report: Cross-Auditor Reconciliation

**Auditor:** Antigravity (Claude Opus 4.6)  
**Date:** 2026-08-09  
**Scope:** Reconciliation with other auditor's findings in `audit/ag_audit/Phase_1_and_2_Audit_Report.md`

---

## 1. Other Auditor's Findings — Agreement/Disagreement

### 1.1 [AGREE — ADOPTED] Sarpy SRP `get_array()` Bug

**Other auditor's finding:** `sarpy_cphd.py` line 65 calls `rg.SRP.get_array()`, but `SRPType` has no `get_array()` method. Should be `rg.SRP.ECF.get_array()`.

**My assessment:** ✅ **AGREE**. I independently verified this by inspecting the sarpy API:
- `SRPType` has attributes: `ECF`, `IAC`, but no `get_array()`.
- `ECF` is an `XYZType` which has `get_array()`.
- The correct call is `rg.SRP.ECF.get_array()`.

**Severity upgrade:** This is more serious than it initially appears. The bug is silently masked because the line uses a conditional guard:
```python
srp_ecf = rg.SRP.get_array() if rg and getattr(rg, "SRP", None) else None
```
If `rg` exists and has `SRP`, the `get_array()` call will throw `AttributeError`. Since this isn't wrapped in try/except, it will **crash** when using the sarpy backend with any real CPHD file that has `ReferenceGeometry`. This should be **CRITICAL** since it prevents the sarpy backend from working for slant plane processing.

**Combined with:** My Phase 2, §2.2 (sarpy reader incomplete) and Phase 1, H-3 (silent error handling in sarpy writer).

---

### 1.2 [AGREE — ADOPTED] SICD XML Schema Non-Compliance

**Other auditor's finding:** The sarkit SICD writer produces XML that fails schema validation. Missing mandatory tags include `Classification`, `FullImage` sub-elements, `SCPPixel`, `EarthModel`, `TimeCOAPoly`, and datetime formatting issues.

**My assessment:** ✅ **AGREE**. I reviewed the sarkit SICD writer (`sarkit_sicd.py`) and confirmed:

1. **`FullImage` (line 59):** Written as `<FullImage>0</FullImage>` but per SICD 1.3.0, it should contain `<NumRows>` and `<NumCols>` sub-elements.
2. **Missing `Classification`:** Not present in `CollectionInfo`. Required by SICD spec.
3. **Missing `SCPPixel`:** Not present in `ImageData`. Should specify the row/col pixel of the Scene Center Point.
4. **Missing `EarthModel`:** Not present under `GeoData`. Required to be `WGS_84`.
5. **Missing `TimeCOAPoly`:** Required in the `Grid` section to specify the center-of-aperture time polynomial.

**Severity:** HIGH — While the NITF files may still render in tolerant viewers, they are technically non-compliant with the SICD 1.3.0 standard. Strict validators will reject them.

**Note:** The sarpy writer (`sarpy_sicd.py`) likely avoids these issues because sarpy's `SICDType` auto-populates many derived fields. But the sarkit writer constructs raw XML and must include all mandatory fields explicitly.

---

### 1.3 [AGREE — ALREADY COVERED] §3.2 Slant Plane Fix Incomplete (Sarpy Crash)

**Other auditor's finding:** The PI's §3.2 fix is incomplete because the sarpy SRP extraction crashes (related to finding 1.1 above).

**My assessment:** ✅ **AGREE**. This is a consequence of finding 1.1. While the structural fix (adding fields to `CPHDMetadata`, using abstraction in `pfa_engine.py`) is correct, the sarpy implementation of the extraction is broken.

**Already covered by:** My finding 1.1 above.

---

### 1.4 [DISAGREE] §4.1 Code Duplication Verified as "Fixed"

**Other auditor's finding:** The code duplication fix is verified as complete — `_deskew_rvp`, `_get_image_plane_vectors`, and `_apply_scp_shift` were extracted properly.

**My assessment:** ⚠️ **PARTIALLY DISAGREE**. While the three methods were indeed extracted, **only `process_channel_czt()` actually calls `_apply_scp_shift()`**. The `process_channel_hybrid()` (lines 363-370) and `process_channel_nufft()` (lines 520-527) both inline the SCP shift logic instead of calling the shared method. This means the duplication fix is incomplete — 2 out of 3 modes still have the duplicated code.

**Evidence:**
- CZT mode (line 216): `sig_patch, P_patch = self._apply_scp_shift(...)` ✅
- Hybrid mode (lines 363-370): Inline `P_patch = P_vecs_orig + u_c * uIAX + ...` ❌
- NUFFT mode (lines 520-527): Inline `P_patch = P_vecs_orig + u_c * uIAX + ...` ❌

---

### 1.5 [DISAGREE] §4.9 LineSpacing/SampleSpacing Verified as "Fixed"

**Other auditor's finding:** Both readers now extract `LineSpacing` and `SampleSpacing`.

**My assessment:** ❌ **DISAGREE**. The sarkit reader correctly extracts these (lines 89-90), but the sarpy reader does NOT. I verified by searching `sarpy_cphd.py` for any reference to `spacing`, `LineSpacing`, or `SampleSpacing` — none found. The `CPHDMetadata` construction in the sarpy reader (lines 69-85) leaves `line_spacing` and `sample_spacing` at their `None` defaults.

---

## 2. Findings Already Covered by My Reports

The following findings from the other auditor were already covered in my Phase 1 or Phase 2 reports:

| Other Auditor Finding | My Equivalent Finding | Notes |
|---|---|---|
| Core math tests pass | Phase 1, §9 Test Suite Assessment | Same conclusion |
| §3.1 Polarization fixed | Phase 2, §1 line 1 | Same verification |
| §3.3 Import fixed | Phase 2, §1 line 3 | Same verification |
| §3.4 CLI fixed | Phase 2, §1 line 4 | Same verification |
| §3.5 ku_center fixed | Phase 2, §1 line 5 | Same verification |
| §4.4 ImageArea fixed | Phase 2, §1 line 7 | Same verification |
| §4.5 NUFFT scatter fixed | Phase 2, §1 line 9 | Same verification |
| §4.6 torch.compile fixed | Phase 2, §1 line 10 | Same verification |
| §4.8 empty_cache fixed | Phase 2, §1 line 8 | Same verification |
| §6 Test anti-patterns fixed | Phase 2, §1 line 12 | Same verification |

---

## 3. Consolidated Unresolved Issues (All Phases Combined)

This is the master list of all unresolved issues across all three phases and both auditors.

### CRITICAL

| ID | Description | Source | Status |
|---|---|---|---|
| **C-1** | Stale `Ku_sample`/`Kr_sample` variable → wrong bandwidth in output SICD metadata | Phase 1, my finding | **UNRESOLVED** |
| **C-2** | NUFFT mode ignores `global_k_ctr_u`/`global_k_ctr_r` → step-chirp grid misalignment | Phase 1, my finding | **UNRESOLVED** |
| **C-3** | In-place slice assignment (`image_2d[...] = patch_img`) breaks PyTorch autograd | Phase 1, my finding | **UNRESOLVED** |
| **C-4** | Sarpy SRP extraction crash (`rg.SRP.get_array()` → should be `rg.SRP.ECF.get_array()`) | Other auditor | **UNRESOLVED** |

### HIGH

| ID | Description | Source | Status |
|---|---|---|---|
| **H-1** | SICD XML schema non-compliance (sarkit writer missing mandatory tags) | Other auditor | **UNRESOLVED** |
| **H-2** | 2D NUFFT deconvolution missing `1/I0(beta)` normalization → cross-mode gain mismatch | Phase 1, my finding | **UNRESOLVED** |
| **H-3** | Silent default assumptions in sarkit CPHD reader (zero bandwidth, arbitrary coordinate frame) | Phase 1, my finding | **UNRESOLVED** |
| **H-4** | Silent `except Exception` in sarpy SICD writer (Null Island, wrong datetime) | Phase 1, my finding | **UNRESOLVED** |

### MEDIUM

| ID | Description | Source | Status |
|---|---|---|---|
| **M-1** | Sarpy reader missing `line_spacing`/`sample_spacing` (PI claimed fixed, not actually fixed) | Phase 2, my finding | **UNRESOLVED** |
| **M-2** | SCP shift code still duplicated in hybrid/NUFFT modes (PI claimed fixed, partially fixed) | Phase 2, my finding | **UNRESOLVED** |
| **M-3** | `_determine_spatial_bounds` fallback uses dimensionally questionable formula | Phase 1, my finding | **UNRESOLVED** |
| **M-4** | Spherical earth approximation for SICD image corners | Phase 1, my finding | **UNRESOLVED** |
| **M-5** | No `__del__` finalizer for sarkit CPHD reader file handle | Phase 1, my finding | **UNRESOLVED** |

### LOW

| ID | Description | Source | Status |
|---|---|---|---|
| **L-1** | `debug_res.py` imports non-existent `SimWriter` class | Phase 1, my finding | **UNRESOLVED** |
| **L-2** | `test_pfa.py` still uses `return` instead of `pytest.skip()` for missing data | Phase 1, my finding | **UNRESOLVED** |
| **L-3** | `SPEED_OF_LIGHT` defined in multiple locations | Phase 1, my finding | **UNRESOLVED** |
| **L-4** | Sarkit writer swap file committed to audit directory | Other auditor, observed | **TRIVIAL** |

---

## 4. Final Summary

| Category | Count |
|---|---|
| Issues verified as fixed by PI | 15 |
| Issues partially fixed by PI | 2 |
| New critical issues (not in original audit) | 4 |
| New high issues (not in original audit) | 4 |
| New medium issues | 5 |
| New low issues | 4 |
| **Total unresolved** | **17** |

### Assessment
The PI has done excellent work addressing the original audit's architectural and usability issues. The remaining unresolved issues fall into two categories:

1. **Multi-channel step-chirp correctness** (C-1, C-2): These bugs only manifest with multi-channel data and were not caught by the original audit or the existing tests. They are the most urgent remaining issues.

2. **Standards compliance and error handling** (C-4, H-1, H-3, H-4): The I/O layer needs hardening — the sarpy SRP crash, SICD schema compliance, and silent error defaults could cause production failures.

Both auditors agree on the overall assessment: the core mathematical engine is sound, the architecture is well-designed, and most of the original audit's critical findings have been properly addressed.
