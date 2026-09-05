# diffpfa — combined audit report

**Combined by:** Claude Code (Claude Fable 5.1), 2026-09-05, from two independent audits of commit `b1717dd`:

| | audit A | audit B |
|---|---|---|
| auditor | Claude Code / Claude Fable 5.1 | Agy / Gemini 3.8 Flash |
| directory | `audit/claude_code_fable_5_1/` | `audit/agy_with_gemini_3.8_flash/` |
| report | `REPORT.md` | `DIFFPFA_INDEPENDENT_AUDIT_REPORT.md` |
| scripts | `a01`–`a15`, `proposed_tests/` | 38 scripts, `test_audit_regressions.py` |

Both audits ran with `/home/feildaw/mypyenv/bin/python` (torch 2.6.0+cu124, sarkit 1.8.0) on the same six Umbra CPHD/SICD pairs. Neither modified the repository. Where the two audits disagreed, the disagreement was settled by a new measurement (`a15_cross_check_other_audit.py`); those cases are listed explicitly in §4. Every number below carries the script that produced it. Run scripts from the repository root.

---

## 1. Executive summary

**The image formation is correct** (both audits, independently). Audit A measured the formed image against an exact analytic reference at the real products' 5 km scale: residual −55 dB, exact localisation, analytic impulse response at every position, gradients passing `gradcheck` with an exact adjoint. Audit B confirmed `gradcheck` on each kernel and end-to-end gradient flow. On the gold collection (2025-10-26 UMBRA-08) diffpfa's pixels register to the vendor's with 0.00 m offset across the scene.

**Two problems matter for the project's goals** and were found by both audits:

1. **The inter-channel phase rotation `exp(−j2π(f_c,global − f_xc)·τ)` must be removed.** CPHD channels are SRP-referenced with RF frequency labels; the term is not a remodulation but an uncontrolled constant phase per sub-band. Both audits showed the shipped simulation cannot see it (its defaults give exactly 0, 0, −60 000 cycles) and that a fractional-cycle delay collapses sub-band coherence (A: 0.999→0.611; B: 0.999→0.348). Audit A's receiver-level simulation (`a14`) shows the CPHD compensation removes every carrier term before the data reach diffpfa, gives the diagnostic to run on real multi-step data, and shows that a non-compliant producer is repaired by a *measured* per-channel phase, not by the formula.
2. **The SICD metadata misdescribes a correct image**, and the two audits found complementary halves: B found the ground-plane `ImageArea` being used as slant bounds (wrong footprint and aspect) and the σ₀ radiometric ratio; A found the time-origin/COA cluster, the PFA polynomial variables, `Grid/Type`, and the k-space centre, and supplied a drop-in that takes sarkit's `sicdcheck` from 13 findings to 0. Both found `ImpRespWid` 12.9 % too wide, `KCtr` = 0, and the axis-aligned image corners.

**Latent defects on paths no local data exercises** (audit A): rotated-axes branch swaps Row/Col metadata (GROUND only), `SGN` never applied, non-symmetric image areas re-centred on the SRP, CZT resampler gain ∝ 1/`SCSS` (unequal sub-band weights), dead RVP branch, `SIGNAL` flag ignored, CZT precision by accidental promotion.

**Differentiability at the API** (both): `pfa_per_polar` is numpy-in/numpy-out; the torch kernels are fully differentiable; a tensor-native entry point is needed for the exploit generator.

---

## 2. Verified correct (union of both audits)

| property | measurement | evidence |
|---|---|---|
| image vs ideal polar→Cartesian reference, 40 m scene | −59 dB RMS rel. peak | A: a02 |
| same, single target swept to 97 % of half-extent | −53.5 … −49.8 dB; amplitude 1.0000 to 90 %, 0.9635 at 97 % | A: a03 |
| same at 5 km scale, spaceborne geometry | −55.7 … −54.1 dB; IRW and PSLR equal to reference | A: a13 |
| target localisation (synthetic) | 0.0000 m | A: a02/a03; B: `tests/test_pfa_coherence.py` |
| impulse response | PSLR −13.30 dB (analytic −13.26); IRW = reference at all positions | A: a03 |
| Kaiser–Bessel deconvolution (edge gain 29×) | no position-dependent error | A: a03 |
| CZT vs direct sum | 1.1×10⁻¹⁴ (double) | B; `tests/test_czt_nufft.py` |
| gradient w.r.t. signal (torch chain) | `gradcheck` complex128 True; adjoint identity 4×10⁻¹⁵; linearity 2×10⁻¹⁵; complex64 FD rel. err 2×10⁻⁵ | A: a04; B: `gradcheck_pipeline.py`, `test_differentiability.py` |
| gradient density pixel→signal | 100 % non-zero; GD reduces a pixel loss 29.0→20.5 in 10 steps | B: `demo_pixel_to_signal_gradient.py`, `verify_optimization_convergence.py` |
| CZT working precision | all FFTs receive complex128 (promotion from float64 `k_start/k_step`) | A: a11 |
| `compute_scp_geometry` vs CPHD `ReferenceGeometry`, 6 collections | ≤ 1×10⁻⁵ °, ≤ 1×10⁻⁴ m | A: a10 |
| slant axes vs vendor, 6 collections incl. left-looking | ≤ 0.003° | A: a10 |
| degree-5 `ARPPoly` fit residual | 0.000 mm (A), < 5 nm synthetic (B) | A: a10; B |
| polar-angle zero of Row axis = CPHD `ReferenceTime` | 0.000 ms | A: a08 |
| PFA corner defocus, ground scatterers, 6 geometries | ≤ 0.005 cycles | A: a10 |
| pixel spectrum baseband, support = declared BW, flat (uniform); vendor identical | centroid ≤ 0.004 cyc/m | A: a07 |
| gold-pair pixel registration, 7 chips | 0.00 m; log-magnitude NCC 0.53–0.81 | A: a12 |
| runtime, 4346×3999 CPHD | 5.8 s wall, 0.80 GB GPU | A: a07 |
| shipped test suite | 8 pass | both |
| `SIGNAL≠1` vectors in Umbra data | all-zero rows (benign) | A: a01 |
| SICD version choice 1.3.0 | sarkit has no default; vendor uses 1.2.1 (2023) and 1.3.0 (2025); keep 1.3.0 | B: `check_sarkit_xsd.py`, `check_umbra_namespaces.py` |

---

## 3. Findings (merged, ranked by impact on trusting a result)

### C1 — CRITICAL — Inter-channel phase rotation corrupts stepped-chirp coherence (A F1 ≡ B F1)
`diffpfa/IFA/PFA.py:142-147`:
```
τ_n = RcvTime_n − RcvTime_ref,n ;  φ_n = −2π (f_c,global − f_xc,m) τ_n ;  s_m[n,:] ·= e^{jφ_n}
```
**Maths.** CPHD DIDD §1.4, §4: the compensated FX-domain phase is `φ(fx) = SGN·2π·fx·ΔTOA` with `fx` the RF frequency; the SRP echo is set to the same constant phase in every vector of every channel. No LO/carrier term survives, so there is nothing to remove. For a burst channel `τ_n` is constant over *n*, hence the term is a constant `exp(−j2π(f_c − f_m)·m·Δτ)` per channel, harmless iff `(f_c − f_m)·m·Δτ ∈ ℤ`. The shipped simulation has `(f_c − f_m) ∈ {+200, 0, −200} MHz`, `Δτ = 150 µs` → 0, 0, −60 000 cycles exactly.
**Evidence.**

| Δτ | rotation cycles (ch 2) | coherence ON | coherence OFF | source |
|---|---|---|---|---|
| 150 µs | −60 000.000 | 0.9986 | 0.9986 | A a06/a14, B |
| 150.00125 µs | (+0.25 on ch 0 basis) | **0.3476**, targets shifted 0.20 m | 0.9986 | B `test_real_sim_patch.py` |
| 137.3217 µs | −54 928.68 | **0.6109**, PSLR −5.2 dB | 0.9988 | A a06/a14 |
| 100.00025 µs | −40 000.10 | 0.957, IRW 0.238 m | 0.9994 | A a14 |
| 150 µs ± 5 µs PRI jitter | pulse-varying | 0.832, IRW **0.300 m** | 0.9985 | A a14 |

`a14` additionally: SRP phase after CPHD compensation ≤ 3×10⁻¹⁷ rad in every channel although the raw demodulated data carried up to 1.58 rad; compliant sub-bands imaged alone show SRP phases −0.02°, +0.05°, −0.02°; a non-compliant producer with residual (0°, 115.2°, 115.2°) gives 0.600 uncorrected, **0.052** with the shipped rotation, 0.9988 with the measured per-channel phase. Figure `claude_code_fable_5_1/out/a14_range_cuts.png`.
**Fix.** Delete lines 142–147. Before processing real multi-step data run `a14` experiment [3] (image each step alone with a common spacing, compare the phase of a dominant scatterer); equal phases → nothing to do; constant offsets → apply the measured offsets. Tests: A `proposed_tests/…::test_subband_coherence_independent_of_burst_timing`; B `test_audit_regressions.py`.

### C2 — HIGH — SICD metadata cluster (A F2–F4, B F4, F5, F7)
All in `diffpfa/IFP.py::_write_sicd` and `sicd_geometry.py`. sarkit `sicdcheck` reports 13 error/warning lines on every product; the gold vendor file reports 2 (its own). Audit A's `a08_metadata_fix_demo.py::corrected_metadata()` fixes everything below and brings sicdcheck to **0** while remaining schema-valid (2023-09-11 product).

| field | written | required | reference | evidence |
|---|---|---|---|---|
| `Grid/Row,Col/ImpRespWid` | `1/BW` | `k/BW`, k = 0.886 (0.8859 in sarkit and vendor) for uniform | DIDD Table 3-4; §4.14.6 | A a07: pixels give k = 0.879/0.885 (vendor control 0.879/0.883); declared/measured = 1.124/1.129. B: all six vendor files declare 0.8859 |
| `Grid/Row/KCtr` | 0 | spatial frequency at the DFT zero bin = k-space grid centre ≈ 64.04 | Table 3-4 | A a01/a07; B check on gold (63.9195) |
| `Grid/Col/KCtr` | 0 | grid centre `g_ku,ctr` (= +0.0707 for the 2025 product: aperture asymmetric, Kaz −0.463/+0.604); 0 only if the processor symmetrises | Table 3-4 | A (see §4, disagreement 3) |
| `Grid/Type` | `PLANE` | `RGAZIM` | §4.15.1 | A |
| `Grid/TimeCOAPoly` | 0 | SCP COA time | Table 3-4 note | A (root cause of 6 sicdcheck SCPCOA errors) |
| `SCPCOA/SCPTime`, `PFA/PolarAngRefTime`, `ARPPoly` origin | relative to first pulse | relative to `CollectStart` (CPHD `TxTime` already is) | Table 3-15; CPHD Table 2-2 | A a10: consumer `ARPPoly(SCPTime)` 32–58 m from truth |
| `PFA/PolarAngPoly` | in `(t − t_ref)` | in absolute time; must be 0 at `PolarAngRefTime` | Table 3-15 | A: product gives −0.0072 rad |
| `PFA/SpatialFreqSFPoly` | in time | in polar angle | Table 3-15 | A |
| COA time | mid-index pulse | time of the image-plane definition (CPHD `ReferenceTime`; a08 recovers it to 0.000 ms as the polar-angle zero) | §4.15.1 | A a08; B noted SCPCOA differs from gold but attributed it to accuracy (see §4) |
| `GeoData/ImageCorners` | lat/lon box | corner pixels projected to the SCP-height ground plane (R/Rdot) | Table 3-3; sarkit check | A a08 / a15: sarkit projection reproduces vendor ICPs to 1.4–2.2 m; slant-plane-point remedy is 1.1–2.0 km off (§4) |
| `Grid/Row,Col/WgtType` | absent | `UNIFORM` | Table 3-4 | A |
| `ImpRespBW` | max−min over all pulses | support at SCP `(2/c)(F_last − F_first)`; 1.2 % overstated | §4.14.6 | A a07 |

### C3 — HIGH — CPHD `ImageArea` (ground) used as slant-plane bounds; footprint and aspect wrong (B F2; A F17, F7)
`IFP.py:133-147`. `ImageArea/X1Y1,X2Y2` live on the CPHD reference surface spanned by `uIAX, uIAY`. Projecting the four ground corners onto the slant axes `(u_row, u_col)`: `r_k = ΔP_k·u_row`, `u_k = ΔP_k·u_col`. Gold collection: the 5000 m square → 3535 m (range) × 5535 m (azimuth); diffpfa's 5000 × 5000 slant box over-covers range 1.41× and clips 5 % of the azimuth extent. Vendor extents are 1.202× and 1.205× the projected spans on the gold file (B's α = 1.20). **Caveat (A a15 a):** across the five older collections the ratio runs 1.04–1.21, so α = 1.20 characterises Umbra SAR Processor 4.25.1, not a general rule; B's "within 0.2 % on all six" does not reproduce. **Interaction (A F7):** `pfa_per_polar` centres the image on the SRP whatever `u_min…r_max` say (a02 §4: SRP lands at the area-centre pixel), so non-symmetric bounds from any projection would be mis-registered until that is fixed. The gold recipe (A §6) also fixes the aperture (symmetric about CPHD `ReferenceTime`) and the range band (ground IRW = 1.000 m). Fix: B `reference_ground_to_slant.py` for the bounding box (drop its ICP part, see C2), plus the SRP-centring fix.

### C4 — HIGH (for the adversarial use) — `pfa_per_polar` is not differentiable at its boundary (A F9 ≡ B F3)
numpy in (`torch.from_numpy(...astype)`), numpy out (`.cpu().numpy()`); a tensor input raises `AttributeError`. Kernels are differentiable (§2). Fix: accept `List[torch.Tensor]` and add `return_tensor=True` (B), or expose a tensor-native chain (A `a04::torch_chain`). Keep `run_pfa.py`'s `inference_mode` out of the attack path.

### C5 — MEDIUM — Rotated-axes branch swaps Row/Col spacing and bandwidth (A F5)
`PFA.py:217-222`; GROUND mode only. Synthetic: returned `bw_range = 2.30` (true 4.00), `N_range = 120` (true 200). Real GROUND run (a09): XML `Row/ImpRespBW` 0.454 vs pixel support 0.642. Fix: make step 6 unconditional; return spacings not counts.

### C6 — MEDIUM — `Global/SGN` parsed, never applied (A F6)
`SGN = +1` data form a point-mirrored image (test `test_sgn_plus_one_is_honoured`). Fix: conjugate when `sgn = +1`, write `Grid/*/Sgn` accordingly.

### C7 — MEDIUM — CZT resampler gain `1/(L·Δk_in)` weights sub-bands by sample spacing (A F8)
`czt_torch.py:188`. Middle band sampled 2× denser → k-space weight 2.007×, range IRW 0.225→0.275 m, PSLR −13.3→−16.2 dB (a11 B). Fix: multiply by `|k_step|·r_step·N_spatial` instead of dividing by `N_spatial`.

### C8 — MEDIUM — Radiometric ratios (B F6, corrected by A a15 b)
`IFP.py:370-372`. DIDD §4.10.4: β₀ per slant-plane area, σ₀ per ground area, γ₀ per area normal to slant range:
```
σ₀ = β₀ · cos(SlopeAng)                      (code: β₀·cos(GrazeAng))
γ₀ = σ₀ / cos(IncidenceAng) = β₀·cos(SlopeAng)/cos(IncidenceAng)   (code: β₀·sin(GrazeAng); B's remedy β₀·sin(SlopeAng) also wrong)
```
Gold file: σ₀/β₀ = 0.696361 = cos(45.8642°); γ₀/β₀ = 0.984625 = cos(slope)/sin(graze); γ₀/σ₀ = 1.413957 = 1/cos(44.9896°). The whole block is otherwise a placeholder (`RCSSF = 1`, `β₀ = 1/(ΔrΔu)`); image amplitude ∝ `N_samples·N_pulses` must be divided out in any real calibration.

### C9 — LOW — Dead/latent code paths (A F11–F13)
RVP deskew keys on PVP `TxFMRate`, which CPHD 1.x does not define, and CPHD FX data is already compensated → dead; if triggered it shifts targets 0.8 m (a05). `SIGNAL≠1` vectors processed (benign here). CZT precision relies on type promotion; forced float32 → −21 dB error (a11 A).

### C10 — LOW — Test suite (A F14, B F9)
Mutation results (a05): `test_kaiser_bessel_kernel_properties` blind to kernel shape; `test_point_target_localization` blind to deconvolution removal, kernel change, amplitude/phase scaling, k-centre offset; `test_fit_arp_poly_residuals` blind to the time origin; schema tests pass with every C2 defect present. Both audits supply pytest suites: A `proposed_tests/test_diffpfa_audit.py` (12 tests; 7 fail today by design), B `test_audit_regressions.py` (4 tests, pass today; note B's `test_imprespwid_normative` tests a constant, not the code, and `test_radiometric_slope_angle` tests the vendor file, not diffpfa).

### C11 — INFO — B F8: `FPN` equals `IPN`
diffpfa computes the polar angle by orthogonal projection onto the slant plane, which is exactly the DIDD §4.15.1 construction when `FPN = IPN`; the metadata is self-consistent and the corner defocus is ≤ 0.005 cycles (A a10). Writing `FPN = geodetic up` (B's remedy) without also projecting along it in `compute_kspace` would make the PFA block inconsistent with the processing. Not a defect; a product-design option.

---

## 4. Where the audits disagreed, and how it was settled

1. **B: "projection with α = 1.20 matches gold dimensions across all six collections to within 0.2 %."** a15 a: ratios vendor/projected are 1.101, 1.214, 1.200, 1.169, 1.112, 1.202 (row) and 1.041, 1.151, 1.135, 1.103, 1.064, 1.205 (col). Holds for the gold file and one older row axis only. The projection itself is correct and valuable; the constant is processor-specific.
2. **B's ICP remedy** (geodetic of `SCP + Δr·u_row + Δu·u_col`): applied to the vendor's own grid it lands 1149–1989 m from the vendor's ICPs, because the slant-plane corner points sit ±1.1–1.9 km above/below the SCP height. sarkit's R/Rdot ground-plane projection (A a08) reproduces the vendor's ICPs to 1.4–2.2 m. Use the ground projection.
3. **B: `Grid/Col/KCtr = 0`.** The DFT zero bin corresponds to the k-space grid centre actually used; diffpfa's full-aperture grid is centred at `(Kaz1+Kaz2)/2 = +0.0707` cyc/m on the 2025 product. 0 is right only after symmetrising the aperture (which the vendor does).
4. **B's γ₀ remedy `β₀·sin(SlopeAng)`.** Gold file gives γ₀/β₀ = 0.9846 = cos(slope)/cos(incidence), not sin(slope) = 0.7177. σ₀ = β₀·cos(slope) is confirmed.
5. **B: "SCPCOA angles match gold to < 0.05°".** a15 d: DopplerCone −0.058°, Twist −0.065°, Azimuth −0.087°, SlantRange +94 m. All explained by A F4 (COA at the mid-index pulse, 0.6117 s, instead of the image-plane time 0.6991 s). The geometry formulas themselves are exact (A a10).
6. **B F8 (FPN)** — see C11; A classes it as a design option, not a defect.
7. **B's F1 figure 0.3476 vs A's 0.6109**: different Δτ (150.00125 µs puts the quarter cycle on a different channel than 137.3217 µs); both reproduce (a15 reran B's script). No conflict.
8. **A's coverage gaps closed by B:** the ground→slant `ImageArea` projection (C3) and the σ₀ ratio (C8). **B's coverage gaps closed by A:** all time-origin/COA/PFA-polynomial/`Grid/Type` items (C2), rotated branch, `SGN`, SRP-centring, resampler gain, RVP, `SIGNAL`, precision, the sicdcheck-verified drop-in, the pixel-level IRW measurement, the gold-pair registration, and the vendor processing recipe.

---

## 5. Concerns that are not defects (do not "fix")

- Baseband pixels with `KCtr ≈ 64`: SICD convention; do not add a carrier ramp to the pixels.
- `FPN = IPN` (C11).
- NUFFT without grid oversampling: −50 dB or better everywhere (A a03).
- Full asymmetric aperture and full band vs the vendor's symmetric/trimmed choice: a product decision; only the metadata must follow (`Col/KCtr`).
- Ideal-PFA vs back-projection references differ by design (A a02).

---

## 6. Recommendations in priority order (with dependencies)

1. **Delete `PFA.py:142-147`** (C1). Adopt A `a14` as the stepped-chirp simulation (non-integer default Δτ) and the coherence test. No dependencies.
2. **Port `a08::corrected_metadata()` into `_write_sicd`** (C2); add a sicdcheck-based test. Independent of 1.
3. **Fix SRP-centring (A F7) then adopt the ground→slant framing** (C3), with the margin as a parameter (1.20 reproduces the gold product); optionally the gold aperture/band recipe (A §6). Depends on the centring fix for `SCPPixel` to be right.
4. **Apply `SGN` (C6), normalise the CZT resampler (C7), fix σ₀/γ₀ (C8).** One-liners.
5. **Tensor-native entry point** (C4). Add A's gradient test and B's density test to CI.
6. **Fix or remove the rotated branch / GROUND mode** (C5).
7. **Delete RVP, mask `SIGNAL≠1`, pin float64 in the CZT** (C9).
8. **Adopt both proposed test suites; keep A `a05_test_bite.py` to prove new tests can fail** (C10).

---

## 7. Appendix A — the maths in one place

Notation: `c` speed of light, `fx` RF frequency, `P̂_n` unit vector Tx/Rx-midpoint → SRP at pulse *n*, `u_row` (range) and `u_col` (azimuth) slant-plane unit vectors, `L_u, L_r` image extents, `N_u, N_r` grid sizes.

- **k-space mapping** (`kspace.py`): `K_u = (2fx/c)(P̂·u_col)`, `K_r = (2fx/c)(P̂·u_row)`; polar angle `PLR = atan2(K_u, K_r)`; `KSF = |P̂_in-plane| = √((P̂·u_col)² + (P̂·u_row)²)`.
- **Image**: `I(x) = Σ_K S(K) e^{+j2πK·x}` over the Cartesian grid `K = K_ctr + m/L`; the IFFT demodulates by `K_ctr`, so the pixel spectrum is baseband and `KCtr := K_ctr`.
- **CZT resampler** (`czt_torch.py`): pass 1 `S(r_m) = Σ_n x_n e^{j2π r_m k_n}`, `r_m ∈ [−L/2, L/2]`; pass 2 `X(k') = (1/N_sp) Σ_m S(r_m) e^{−j2π k' r_m} ≈ x(k′)/(L·Δk_in)`. Correct normalisation `Δk_in·r_step·Σ_m` (C7).
- **KB gridding/deconvolution**: kernel `w(κ) = I₀(β√(1−(2κ/J)²))/I₀(β)`, `J = 6`, `β = 13.9086`; transform `W(ξ) = sinh(√(β²−(πJξ)²))/√(β²−(πJξ)²)`, normalised by `W(0)`, `ξ ∈ [−½, ½)`.
- **Impulse response** (uniform): `IRW = 0.886/BW` (half power), `PSLR = −13.26 dB`; `1/BW` is the Rayleigh width.
- **Phase rotation cycles**: `N = (f_c,global − f_xc,m)·m·Δτ`; harmless iff `N ∈ ℤ`. Shipped defaults: 0, 0, −60 000.
- **Time origins**: CPHD `TxTime`, `RcvTime` are relative to `CollectionStart`; SICD `ARPPoly(t)`, `PolarAngPoly(t)`, `SCPTime`, `PolarAngRefTime`, `TimeCOAPoly(0,0)` all use that same origin; `PolarAngPoly(PolarAngRefTime) = 0`.
- **Ground→slant framing**: corners `ΔP_k = x_k u_IAX + y_k u_IAY`; `r_k = ΔP_k·u_row`, `u_k = ΔP_k·u_col`; extents `[min r_k, max r_k]·α`, `[min u_k, max u_k]·α`; gold: α = 1.20; range band such that `0.8859/BW_r / cos(graze) = 1.0 m`; aperture `|t − t_ref| ≤ min(t_ref − t_first, t_last − t_ref)`; `SS·BW = 0.8`.
- **Image corners**: `x_row = (row − SCPRow)·SS_row`, `y_col = (col − SCPCol)·SS_col`; project with the R/Rdot contour to the plane through the SCP with normal `up(SCP)` (sarkit `image_to_ground_plane`), then ECF→LLH.
- **Radiometry**: `σ₀ = β₀ cos(SlopeAng)`, `γ₀ = σ₀ / cos(IncidenceAng)`.
- **SCPCOA/geometry**: `compute_scp_geometry` (DIDD §5.3 formulas) verified against the CPHD producer to 1×10⁻⁵ ° when evaluated at the same time.

## 8. Appendix B — script index (both audits)

**Audit A (`audit/claude_code_fable_5_1/`)**: a01 inventory; a02 forward model vs exact references; a03 IPR vs position; a04 gradients; a05 mutation harness; a06 subband phase; a07 real-data run + pixel IRW; a08 metadata drop-in + sicdcheck 13→0; a09 GROUND mode; a10 geometry/ARPPoly/time origin/defocus; a11 precision + sub-band gain; a12 gold-pair registration; a13 full-scale vs ideal; a14 receiver-level stepped-chirp simulation; a15 cross-check of audit B; `proposed_tests/test_diffpfa_audit.py`.

**Audit B (`audit/agy_with_gemini_3.8_flash/`)**: `test_real_sim_patch.py` (phase rotation bite); `reference_ground_to_slant.py`, `verify_ground_to_slant_projection.py` (framing); `gradcheck_pipeline.py`, `test_differentiability.py`, `demo_pixel_to_signal_gradient.py`, `verify_optimization_convergence.py` (gradients); `check_radiometric_math.py`, `find_radiometrics.py` (σ₀); `check_fpn_up.py` (FPN); `check_sarkit_xsd.py`, `check_umbra_namespaces.py` (SICD version); `test_audit_regressions.py`.
