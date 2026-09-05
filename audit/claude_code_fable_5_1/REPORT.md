# diffpfa audit report

**Auditor:** Claude Code, running Claude Fable 5.1 (Anthropic)
**Date:** 2026-09-05
**Code audited:** `diffpfa` at commit `b1717dd` (2026-08-29), working tree unmodified
**Environment:** `/home/feildaw/mypyenv` (Python 3.12.3, torch 2.6.0+cu124, numpy 2.4.1, sarkit 1.8.0, scipy 1.17.1), RTX 3070 Laptop 8 GiB, WSL2
**Rules followed:** no file under the repository was modified; everything the audit produced lives in `audit/claude_code_fable_5_1/`; `PI/` was not read; the handoff in `audit/AUDIT_HANDOFF.md` was used for method only.

All numbers below were produced by the scripts named in brackets. Every script runs from the repository root with `/home/feildaw/mypyenv/bin/python <script>`; results are also saved as JSON under `out/`.

---

## 0. Executive summary

The image-formation mathematics is correct. Against an exact analytic reference (the scene's k-space sampled on the very Cartesian grid the code uses, masked to the polar support, inverse-FFT'd), diffpfa's image agrees to −55 dB RMS relative to the peak at the real products' 5 km scale, with exact target localisation, the analytic uniform-aperture impulse response (−13.3 dB first sidelobe) at every position in the image, and gradients with respect to the signal that pass `torch.autograd.gradcheck` and satisfy the adjoint identity to 4×10⁻¹⁵. On the gold collection (2025-10-26 UMBRA-08), diffpfa's pixels register to the vendor's pixels with 0.00 m offset at seven chips spanning the 4 km scene.

Three classes of problems were found:

1. **A physics defect that matters precisely for the target use case (stepped chirp).** The per-channel phase rotation `exp(−j2π(f_c,global − f_xc)·τ)` has no basis in the CPHD signal model and destroys inter-sub-band coherence whenever `(f_c,global − f_xc)·τ` is not an integer number of cycles. The shipped simulation cannot see this because its default parameters make the rotation an exact identity (0, 0 and −60 000.000 cycles). With a non-integer delay the complex coherence with the monolithic reference drops from 0.9988 to 0.6109 and the first sidelobe rises from −13.3 to −5.2 dB; with realistic PRI jitter the range IPR widens from 0.225 to 0.300 m. A receiver-level simulation that models the local oscillator explicitly (a14) shows that CPHD compensation removes every carrier term before the data reach diffpfa, and that a non-compliant producer's residual phase is repaired by a *measured* per-channel phase, not by the formula. (§3, F1)
2. **SICD metadata that misdescribes a correct image.** `ImpRespWid` is 12.9 % too large, `KCtr` is 0 instead of ≈64 cycles/m, every time-referenced field is measured from the first pulse instead of `CollectStart` (32–58 m position error for a consumer evaluating `ARPPoly` at `SCPTime`), `TimeCOAPoly` is 0, the PFA polynomials use the wrong independent variables, `Grid/Type` is `PLANE`, and the image corners are a lat/lon box 5.8 km from the truth. sarkit's `sicdcheck` reports 13 error/warning lines on every product. A drop-in reference implementation (`a08_metadata_fix_demo.py::corrected_metadata`) brings that to **0** while keeping the XML schema-valid. (F2–F4)
3. **Latent defects on code paths no local data exercises:** the rotated-axes branch swaps Row/Col sample spacing and bandwidth (GROUND mode only), the CPHD `SGN` is parsed but never applied, non-symmetric image areas are silently re-centred on the SRP, and the CZT resampler's gain depends on the fast-time sample spacing so sub-bands with different `SCSS` are weighted unequally. (F5–F8)

For the adversarial use: differentiability with respect to the signal is intact and numerically right inside the torch kernels, but `pfa_per_polar` itself takes and returns numpy, so the exploit generator must call the kernels directly (a template is given). (F9)

A reproducible reverse-engineering of what the vendor did on the gold file is in §6: a symmetric aperture about the CPHD reference time, range bandwidth trimmed to give exactly 1.000 m ground-range resolution, 1.25× oversampling, and a 4249 m × 6671 m slant-plane footprint.

---

## 1. What the project does (as audited)

For each polarisation group of a CPHD, `IFAProcessor.run()` reads the channels, builds slant-plane axes from the CPHD reference geometry (`u_row` = unit vector ARP→SRP, `u_col` = velocity component orthogonal to it, negated for left-looking), and calls `pfa_per_polar`, which:

- maps each sample `(n, k)` to spatial frequency `K = (2F/c)·P̂_n` projected onto `(u_col, u_row)`, with `P̂_n` the unit vector from the Tx/Rx-midpoint to the SRP (`kspace.py`);
- resamples every pulse along range with a two-pass chirp-Z transform onto a uniform `K_r` grid of step `1/L_r` centred on `(K_r,min + K_r,max)/2` (`czt_torch.py`);
- grids along cross-range with a Type-1 NUFFT (Kaiser–Bessel, `J = 6`, `β = 13.9086`, no grid oversampling) using the factorisation `K_u = cot θ_n · K_r` (`nufft_torch.py`, `pfa_channel.py`);
- accumulates channels on one grid, inverse-FFTs, and divides by the kernel's closed-form transform along cross-range (`PFA.py`).

The image is therefore an RGAZIM slant-plane PFA image whose pixel spectrum is baseband (the IFFT implicitly demodulates by the grid centre). `_write_sicd` then writes SICD 1.3 XML and the pixels through sarkit.

---

## 2. What is verified correct

| Property | Measurement | Script |
|---|---|---|
| Image vs ideal polar→Cartesian reference, 40 m scene, 3 targets | residual −59.1 dB RMS rel. peak; fitted gain phase −0.0002° | a02 |
| Same, single target swept to 97 % of half-extent along u and along r | residual −53.5 … −49.8 dB; peak amplitude 1.0000 out to 90 %, 0.9635 at 97 % | a03 |
| Same at the products' scale (5 km, k ≈ 64 cyc/m, R = 570 km), targets at centre and corners | residual −55.7 … −54.1 dB; IRW and PSLR identical to the reference to the last digit | a13 |
| Target localisation (synthetic) | error 0.0000 m at all tested positions | a02, a03 |
| IPR shape | PSLR −13.30 dB (analytic uniform: −13.26); IRW equals reference at every position; along r the CZT is exact | a03 |
| Kaiser–Bessel deconvolution | correct: no position-dependent amplitude or width error despite edge gain 1/W(½) = 29.2 | a03 |
| Matched-filter (back-projection) reference | −40.7 dB — the difference from the ideal-PFA reference is the expected polar-Jacobian/support difference, not an error | a02 |
| Gradient w.r.t. signal (torch chain) | `gradcheck` complex128 **True**; linearity residual 2.1×10⁻¹⁵; adjoint ⟨y,Ja⟩ = ⟨Jᴴy,a⟩ to 3.95×10⁻¹⁵; complex64 directional derivative vs central FD rel. err 2×10⁻⁵ | a04 |
| CZT working precision | all 32 `torch.fft.fft` calls inside the CZT receive complex128 (type promotion from float64 `k_start/k_step`) | a11 A |
| `compute_scp_geometry` vs the CPHD producer's own `ReferenceGeometry` angles, 6 collections | ≤ 1×10⁻⁵ °, slant range ≤ 1×10⁻⁴ m | a10 |
| Slant-plane Row/Col unit vectors vs vendor SICD, 6 collections incl. one left-looking | ≤ 0.003° | a10 |
| Corrected SCPCOA vs sarkit's independent implementation | angles agree to 3×10⁻⁶ ° | a08 |
| Polar-angle zero of the product's Row axis vs CPHD `ReferenceTime` | 0.000 ms | a08 |
| Degree-5 `ARPPoly` fit residual on real 8229-vector trajectory | 0.000 mm | a10 |
| PFA space-variant defocus for ground scatterers at the ImageArea corners, 6 real geometries | ≤ 0.005 cycles peak-to-peak (slant focus plane is adequate) | a10 |
| Pixel spectrum: baseband, support equals declared `ImpRespBW`, flat in-band (uniform) — same as vendor | centroid ≤ 0.004 cyc/m; support/declared 0.988 (Row), 0.999 (Col) | a07 |
| Gold pair registration: diffpfa vs vendor pixels at 7 chips across the scene | offset 0.00 m (row and col) everywhere; log-magnitude NCC 0.53–0.81 | a12 |
| Real-data run, 2023-09-11 (4346 × 3999 CPHD) | 5.8 s wall, 4.15 s processing, 0.80 GB peak GPU allocation | a07 |
| Shipped test suite | 8 passed | pytest |
| `SIGNAL` PVP flag ignored — benign on Umbra data | all flagged vectors (431/13152, 200/4439, 1/4346) are all-zero rows | a01 |

Figure: `out/a12_scp_chip_vendor_vs_diffpfa.png` (768² vendor-pixel chip at the SCP, vendor left, diffpfa resampled to the vendor grid right).

---

## 3. Findings

Severity is by impact on trusting a diffpfa result for the project's stated goals (slant-plane image fidelity for stepped-chirp, multi-polarisation CPHD; SICD metadata trustworthiness; signal-gradient correctness).

### F1 — HIGH — Inter-channel "differential carrier phase" rotation is unphysical and breaks sub-band coherence
`PFA.py:142-147` multiplies channel *m* by `exp(−j2π(f_c,global − f_xc,m)·τ_n)`, `τ_n = RcvTime_n − RcvTime_ref,n`.

*Why it is wrong.* CPHD DIDD §1.4 (pp. 9–11) and §4 (p. 35): the compensated FX-domain signal has phase `φ(fx) = SGN·fx·ΔTOA` with `fx` the **RF** frequency, and "the phase of the SRP signal is set to zero by the compensation processing … the SRP phase is set to the same constant value for all vectors". No local-oscillator or carrier term survives compensation, in any channel. The rotation therefore removes nothing; it multiplies channel *m* by the constant `exp(−j2π(f_c − f_xc,m)·m·Δτ)` (constant because `τ` is the same for every pulse of a burst channel), i.e. it applies an arbitrary relative phase between sub-bands.

*Measurement (a06).* The shipped simulation has `f_c − f_xc ∈ {+200, 0, −200} MHz`, `Δτ = 150 µs` → 0, 0, −60 000.000 cycles: the rotation is exactly the identity, so the simulation's 0.9986 coherence proves nothing about it. With `Δτ = 137.3217 µs` (−54 928.68 cycles on channel 2):

| | coherence with monolithic reference | range IRW | peak ratio |
|---|---|---|---|
| Δτ = 150 µs, rotation on or off | 0.998561 | 0.400 m | 0.9974 |
| Δτ = 137.3217 µs, rotation **on** (shipped) | **0.610858** | **0.600 m** | **0.6675** |
| Δτ = 137.3217 µs, rotation **off** (neutralised in-process) | 0.998793 | 0.400 m | 0.9974 |

Real stepped-chirp bursts will have arbitrary fractional cycles. The README's claim that this term "preserves coherence across moving platform subband bursts" is not supported by the simulation, which never exercised it.

*Improved simulation (a14, written at the developer's request).* `a14_stepped_chirp_simulation_v2.py` starts one step earlier than the shipped script, at the receiver: each step *m* is demodulated with its own local oscillator `exp(−j(2π f_m t + θ_mn))` using the worst-case LO convention (phase reset at every pulse), so the raw data carry a large channel- and pulse-dependent phase; the CPHD producer's compensation is then applied exactly as CPHD DIDD §1.4 defines it (zero the SRP echo phase using the known SRP range and receiver timing) and the channels are labelled with RF frequency. Results:

| experiment | result |
|---|---|
| [1] SRP phase in the compensated channels (raw data had up to 1.58 rad) | ≤ 3×10⁻¹⁷ rad in all channels: no carrier term survives, whatever the LO did |
| [2] 3 × 200 MHz vs monolithic, Δτ = 150 µs (−60 000.000 cycles) | rotation ON 0.9986 / OFF 0.9986 — identical, as in the shipped simulation |
| [2] Δτ = 137.3217 µs (−54 928.68 cycles) | ON: coherence **0.611**, PSLR **−5.2 dB**, peak 0.668 · OFF: 0.9988, −13.3 dB, 0.997 |
| [2] Δτ = 100.00025 µs (−40 000.10 cycles) | ON: 0.957, IRW 0.238 m, PSLR −11.6 dB · OFF: 0.9994, 0.225 m, −13.3 dB |
| [2] Δτ = 150 µs with ±5 µs PRI jitter (rotation now varies pulse to pulse) | ON: 0.832, IRW **0.300 m**, peak 0.704 · OFF: 0.9985, 0.225 m, 0.997 |
| [3] each compliant channel imaged alone, phase of the SRP scatterer | −0.02°, +0.05°, −0.02°: the inter-band phases already agree, there is nothing to rotate |
| [4] non-compliant producer leaving a constant residual phase per channel (0°, 115.2°, 115.2°) | no correction 0.600 · shipped rotation **0.052** (its model gives 0°, 0°, 115.2°) · per-channel phase *measured* as in [3] **0.9988** |

Figure: `out/a14_range_cuts.png` (range cut through the SRP scatterer for Δτ = 137.3217 µs: reference, rotation OFF, rotation ON).

Experiment [3] is the diagnostic to run on the real multi-step CPHD before deciding anything: image each step alone with a common pixel spacing, read the phase of one dominant scatterer in each, and compare. Equal phases mean the producer is compliant and no correction is needed; constant offsets mean a non-compliant producer and the offsets themselves are the correction; the formula in `PFA.py` is right only if the hardware's LO phase happens to equal `2π(f_c − f_m)·τ`, which in [4] it did for one channel out of three by coincidence.

*Fix.* Delete the rotation for CPHD-compliant input. If the high-side producer is known to leave a per-step carrier phase in (a non-compliance), it must be measured (overlapping band or reference scatterer, experiment [3]), not modelled from `RcvTime`. Proposed test: `proposed_tests/test_diffpfa_audit.py::test_subband_coherence_independent_of_burst_timing` (fails today).

### F2 — HIGH (metadata) — `ImpRespWid = 1/ImpRespBW` instead of the half-power width
`IFP.py:245`. DIDD Table 3-4 defines ImpRespWid as the half-power width; §4.14.6 (p. 162): `Rg_IRW = k_RG / Krg_IRBW`, `k_RG = 0.886` for uniform weighting. sarkit and the Umbra products use 0.8859.

*Measurement (a07).* Aperture→IPR measured from the pixels of the fresh 2023-09-11 product versus the vendor's product of the same collection as control:

| | declared k = IRW·BW | measured k | declared / measured IRW |
|---|---|---|---|
| vendor (control) Row / Col | 0.8857 / 0.8857 | 0.8785 / 0.8832 | 1.009 / 1.003 |
| diffpfa Row / Col | 1.0000 / 1.0000 | 0.8791 / 0.8852 | **1.124 / 1.129** |

A consumer sizing anything from `ImpRespWid` is off by 12.9 %. Also declare `WgtType/WindowName = UNIFORM`.

### F3 — HIGH (metadata) — `KCtr = 0` misdescribes the spectrum; `Grid/Type = PLANE`
`IFP.py:248,233`. DIDD Table 3-4: KCtr is the spatial frequency that "corresponds to the zero frequency of the DFT in the row direction". The pixels are baseband (a07: centroid −0.0009 cyc/m) but the k-space they came from is centred at `(K_r,min+K_r,max)/2 ≈ 64.04 cyc/m`; the vendor declares 63.92 for its band. Writing 0 asserts a carrier of 0 Hz and fails sarkit's `Krg within 0.5/SS of KCtr`. For Col, `KCtr` must be the grid centre `g_ku,ctr` actually used (−0.0002 for 2023-09-11 but **+0.0707** for the 2025 file, whose aperture is asymmetric: Kaz1/Kaz2 = −0.463/+0.604), not 0. §4.15.1 (p. 167): "The resulting PFA image is Grid Type = RGAZIM".

### F4 — HIGH (metadata) — Time origin, COA time, PFA polynomials, image corners
`sicd_geometry.py:190-203, 254-273`, `IFP.py:207-235`.

- `fit_arp_poly` and `compute_pfa_metadata` subtract `TxTime[0]`; CPHD `TxTime` is already relative to `CollectionStart` (CPHD DIDD Table 2-2). A SICD consumer evaluating `ARPPoly(SCPTime)` gets the ARP **32.5–57.6 m** from where it was at that absolute time (a10, all six collections).
- `TimeCOAPoly = 0.0`; DIDD Table 3-4: "Coefficient (0,0) is the SCP COA time". This single field is why sarkit rejects `SCPCOA/SCPTime, ARPPos, ARPVel, ARPAcc, SlantRange, GroundRange`.
- `PolarAngPoly` is fitted in `(t − t_ref)` and `SpatialFreqSFPoly` in time; Table 3-15 requires a polynomial in time since Collection Start and a polynomial in polar angle respectively. `PolarAngPoly(PolarAngRefTime) = −0.0072 rad` in the product (must be 0).
- SCPCOA is evaluated at the mid-index pulse but the image plane was defined at the CPHD `ReferenceTime`, so `PolarAngPoly(t_ref) ≠ 0` even before the variable error (0.0011 rad for 2025).
- `ImageCorners` are a lat/lon box centred on the SCP ignoring orientation and slant geometry; sarkit: ICP1 is 5810 m from the SCP+grid prediction (2025 product).

*Demonstration (a08).* `corrected_metadata()` recomputes all of the above from the product's own Row/Col vectors and the reference-channel PVPs. sicdcheck on the 2023-09-11 product: **13 → 0** error/warning lines; schema valid; recovered `t_ref` = CPHD `ReferenceTime` to 0.000 ms; `SCPCOA/ARPPos` = CPHD `ReferenceGeometry/ARPPos` to 0.000 m; corners from `sarkit.sicd.image_to_ground_plane`.

### F5 — MEDIUM (GROUND mode) — Rotated-axes branch swaps Row/Col sample spacing and bandwidth
`PFA.py:217-222`, `IFP.py:571-572`. After the internal `Ku/Kr` swap in step 1.1, internal "r" *is* the range axis, so step 6 must not swap again. When the branch is taken, the returned `(bw_range, N_range)` describe the azimuth axis.

*Measurement.* Synthetic (a03 §3): returned `bw_range = 2.3048` (true 3.9966), `N_range = 120` (true rows = 200) → `Row/SS` 0.333 written for a true 0.200. Real data, GROUND mode on 2023-09-11 (a09): branch taken; XML `Row/ImpRespBW = 0.4543` while the pixel spectrum along Row spans 0.642 cyc/m; `Col/ImpRespBW = 0.6881` while the pixels span 0.454. Image content is correct; metadata is swapped. `IFP` also derives `dr_range` from the *un-swapped* extents, so a non-square area would additionally mis-scale SS. SLANT mode never enters this branch (LOS is always `u_row`). Test: `test_rotated_axes_return_true_range_and_azimuth_parameters` (fails today).

### F6 — MEDIUM (latent) — `Global/SGN` parsed, never applied
`IFP.py:78` reads it; nothing uses it. A CPHD with `SGN = +1` produces a point-mirrored image (target at (+5, −5) m appears at (−5, +5) m; `test_sgn_plus_one_is_honoured`). Fix: conjugate the signal when `sgn = +1` (and write `Grid/Row|Col/Sgn` accordingly).

### F7 — MEDIUM (latent) — Image area extents are used only for size; the image is always centred on the SRP
The CZT evaluates space over `[−L/2, L/2]` and the IFFT origin is the k-space grid centre, so `u_min/u_max/r_min/r_max` only set `L`. With `u ∈ [0,40], r ∈ [−10,30]` the SRP lands at the area-centre pixel (60,100) instead of (0,50) (a02 §4). `SCPPixel = N//2` keeps the SICD self-consistent, but `ExtendedArea` or any non-symmetric `ImageArea` will not image the requested region. Fix: apply the linear phase `exp(+j2π(K_u·u_c + K_r·r_c))` to the grid (or shift the CZT window), and compute `SCPPixel` from `(u_c, r_c)`.

### F8 — MEDIUM (latent, multi-channel) — CZT resampler gain is `1/(L·Δk_in)`
`czt_torch.py:188`. The second pass divides by `N_spatial`; the correct inverse weight is `Δk_in·r_step` (∫ over space of the pulse's spatial signal). The resampled k-space is `x(k′)/(L·Δk_in)` — constant per channel, so single-channel images only pick up a scale (a02 §2: peak ∝ N_samples, independent of L), but sub-bands with different `SCSS` get different weights.

*Measurement (a11 B).* Three 200 MHz sub-bands, middle one sampled twice as densely: k-space amplitude ratio low/mid/high = 1.000 / **2.007** / 0.999; range IRW 0.225 → 0.275 m; PSLR −13.3 → −16.2 dB. Fix: multiply each pulse's output by `|k_step|·r_step·N_spatial` (≈ `|k_step|·L_r`) instead of dividing by `N_spatial`.

### F9 — MEDIUM (for the adversarial use) — `pfa_per_polar` is not differentiable at its API boundary
It ingests `List[np.ndarray]`, casts with `torch.from_numpy(...astype(np.complex64))`, and returns `.cpu().numpy()`. No autograd graph can cross it (a04 §1). The torch kernels it calls are fully differentiable (a04 §2–3), and the only `.item()` calls act on geometry, not signal. `a04_gradient_check.py::torch_chain` is a 20-line torch-in/torch-out template that reproduces the pipeline for one channel; the exploit generator should use an entry point like it (or `pfa_per_polar` should accept tensors and skip the numpy round-trip). `run_pfa.py` wraps in `inference_mode`, which is right for batch processing but must not be inherited by the attack path.

### F10 — LOW — `ImpRespBW` overstated by the polar trapezoid
`bw_r = K_r,max − K_r,min` over all pulses includes the `cos(θ)` variation across the aperture; the support at the SCP is `(2/c)(F_last − F_first)`. 2023-09-11: declared 0.6328 vs measured 0.6253 (1.2 %); vendor exact (1.000). Prefer the DIDD form `Krg_IRBW = N_S·Krg_SS` evaluated at the reference pulse.

### F11 — LOW — RVP deskew is dead code keyed on a PVP that does not exist in CPHD
`pfa_channel.py:19`: `"TxFMRate"` is not a CPHD 1.x PVP (CPHD DIDD Table 2-2; the only FM-rate parameters are the optional `TxWF/LFMRate` XML fields). CPHD FX data is already compensated ("the details of the transmitted waveform and the radar receiver timing are no longer needed", p. 10). The branch can never run on compliant data; if it did, a05 shows a bogus `γ` shifts targets by 0.8 m. Remove it (or move it behind an explicit, documented option).

### F12 — LOW — `SIGNAL` PVP not honoured
Vectors with `SIGNAL ≠ 1` are processed. On the Umbra files they are all-zero rows, so the effect is only wasted compute; CPHD allows arbitrary content in such vectors. Mask them.

### F13 — LOW — CZT precision depends on accidental type promotion
`czt_1d_torch` selects float32 for complex64 signals; its phases reach `2π·2500·64 ≈ 10⁶ rad`, where float32 resolves 0.06 rad. Today `k_start/k_step` arrive as float64 and promote everything to complex128 (a11 A). If a future edit casts them to `real_dtype`, the image degrades to −21 dB max error (a11 A, forced). Make the float64 choice explicit.

### F14 — LOW — Test suite cannot see most of the above
Mutation results (a05; each mutation applied at runtime in a subprocess, no source edits):

| test | injected defect | outcome |
|---|---|---|
| `test_czt_1d_vs_direct_sum` | post-chirp sign flipped | FAIL (good) |
| `test_kaiser_bessel_kernel_properties` | kernel replaced by rect window | **PASS** (blind) |
| `test_point_target_localization` | look-vector sign; `F/c`; RVP γ bogus; CZT conj dropped | FAIL (good) |
| `test_point_target_localization` | deconvolution removed; NN gridding kernel; amplitude ×10⁻³ and phase +90°; azimuth k-centre off by 0.5 cyc/m | **PASS** (blind to amplitude, phase, IPR shape) |
| `test_compute_scp_geometry_umbra_reference` | twist sign flipped | FAIL (good) |
| `test_fit_arp_poly_residuals` | `TxTime` shifted by +0.3 s | **PASS** (blind to F4) |
| `test_sicd_*_schema_validation` | (schema only) | passes with F2–F4 present |

`proposed_tests/test_diffpfa_audit.py` adds 12 tests: 5 lock in verified behaviour (pass today), 7 document F1, F5, F6, F7, F8 and the metadata (fail today, turn green with the fixes).

### F15 — LOW — Radiometric block is a placeholder, and its two trigonometric ratios are wrong
`RCSSFPoly = 1`, `BetaZeroSFPoly = 1/(Δr·Δu)`; the vendor writes 5×5 polynomials. Already on the roadmap. Note that image amplitude scales with `N_samples·N_pulses` (matched-filter gain, a02 §2), so any future calibration must divide that out.

*Added after reading the other audit (a15 b).* `IFP.py:370-372` writes `σ₀ = β₀·cos(graze)`, `γ₀ = β₀·sin(graze)`. DIDD §4.10.4 defines β₀ per unit slant-plane area, σ₀ per unit ground-plane area and γ₀ per unit area normal to the slant range, so `σ₀ = β₀·cos(SlopeAng)` and `γ₀ = σ₀ / cos(IncidenceAng) = β₀·cos(SlopeAng)/cos(IncidenceAng)`. The gold file confirms both to six digits: σ₀/β₀ = 0.696361 = cos(45.8642°); γ₀/β₀ = 0.984625 = cos(slope)/sin(graze); γ₀/σ₀ = 1.413957 = 1/cos(44.9896°). The other audit found the σ₀ error (credit: agy/Gemini F6) but its proposed `γ₀ = β₀·sin(slope)` (= 0.7177) is also wrong.

### F17 — MEDIUM — CPHD `ImageArea` is a ground-plane rectangle but is used as slant-plane bounds
*Found by the other audit (agy/Gemini F2); re-derived in a15 a.* `IFP.py:133-147` takes `ImageArea/X1Y1, X2Y2` (defined on the CPHD reference surface spanned by `uIAX, uIAY`) and uses them directly as the slant-plane `u,r` extents. For the gold collection the 5000 m ground square projects onto the vendor's slant axes to 3535 m (range) × 5535 m (azimuth), so diffpfa's 5000 × 5000 m slant box over-covers range by 1.41× and clips the azimuth corners of the requested ground area by 5 %. The vendor's extent is 1.202× / 1.205× the projected span in the gold file; across the five older collections the ratio varies from 1.04 to 1.21, so 1.20 is a property of Umbra SAR Processor 4.25.1, not a universal rule. Fix: project the four ground corners onto `(u_row, u_col)`, take the bounding box, apply the product's chosen margin (see §6 for the gold recipe). Interaction with F7: any non-symmetric bounds produced this way are silently re-centred on the SRP by the current `pfa_per_polar`, so F7 must be fixed first or the `SCPPixel` will be wrong.

### F16 — INFO — Miscellaneous metadata
`RadarMode/ModeType` hard-coded `SPOTLIGHT` (CPHD carries it); `CollectorName = "CZTPFA"` (CPHD: `Umbra-08`); `TxFrequencyProc/MaxProc = FxMax` although samples extend to `SC0 + (N−1)·SCSS` (0.1 MHz higher); `CollectDuration = TxTime[-1] − TxTime[0]` (should be from CollectStart); output filename split on `_CPHD` is Umbra-specific.

---

## 4. Drop-in material

- **`a08_metadata_fix_demo.py::corrected_metadata(xmltree, pvp)`** — returns a corrected copy of a product's XML: `Grid/Type`, `TimeCOAPoly`, `ImpRespWid`, `KCtr`, `WgtType`, `CollectDuration`, `TStartProc/TEndProc`, `ARPPoly` (absolute time, Tx/Rx-midpoint at mid time), `SCPCOA` (via `sarkit.sicd.compute_scp_coa`, cross-checked against `compute_scp_geometry`), `PolarAngRefTime` (root of the polar angle about the product's Row axis), `PolarAngPoly` (absolute time), `SpatialFreqSFPoly` (polar angle), `Krg/Kaz`, `IPN`, `ImageCorners` (sarkit ground-plane projection). Verified: sicdcheck 13 → 0, schema valid. It can be ported into `_write_sicd` almost line for line; the inputs it needs (`u_row`, `u_col`, `Ku`, `Kr`, PVPs, `N_r`, `N_u`, `SS`) are all available there.
- **`a04_gradient_check.py::torch_chain`** — tensor-in/tensor-out equivalent of `pfa_per_polar` for one channel (F9).
- **`proposed_tests/test_diffpfa_audit.py`** — 12 pytest tests (see F14). Run: `python -m pytest audit/claude_code_fable_5_1/proposed_tests -v`.
- **`a05_test_bite.py`** — reusable mutation harness for proving future tests can fail.

Fix recipes not implemented here (each a few lines): F1 delete `PFA.py:142-147`; F5 make step 6 of `pfa_per_polar` unconditional (`bw_range, bw_azm = bw_r, bw_u; N_range, N_azm = N_r, N_u`) and return the pixel spacings rather than counts; F6 `sig = sig.conj()` when `cphd_meta.sgn == +1`; F8 replace `/ float(N_spatial)` with `* (torch.abs(k_step_b) * r_step * N_spatial)`; F11 delete `_deskew_rvp`; F12 drop rows with `SIGNAL != 1` before `compute_kspace`.

---

## 5. Concerns that are not defects (do not "fix")

- **Baseband pixels with `KCtr ≈ 64`.** Both the vendor and diffpfa store demodulated pixels; that is the SICD convention (KCtr is metadata, not a phase ramp in the data). Do not add a carrier ramp to the pixels.
- **`FPN = IPN` (slant focus plane).** Unusual (the vendor uses the geodetic up vector) but self-consistent with how the polar angle is computed, and the resulting corner defocus is ≤ 0.005 cycles for all six geometries (a10). Fine for these collections.
- **No NUFFT grid oversampling (`oversample = 1.0`).** With `J = 6, β = 13.9086` the measured accuracy is −50 dB or better everywhere and amplitude error 3.7 % only in the outermost 3 % of the extent (a03). Adequate; oversampling would cost 2× memory for little.
- **Full asymmetric aperture and full band** (vs the vendor's symmetric/trimmed choice) give finer resolution than the vendor; that is a product-design choice, but the metadata must then carry `Col/KCtr = g_ku,ctr ≠ 0` (F3).
- **Ideal-PFA reference vs back-projection differ by 40 dB residual.** Expected: PFA's Cartesian mask vs the polar Jacobian. Not an error.
- **Image amplitude ∝ `N_samples·N_pulses`.** Natural matched-filter scaling; relevant only to F15.

---

## 6. What the vendor did on the gold file (2025-10-26 UMBRA-08), from its metadata

All statements are arithmetic on the vendor XML and the CPHD (a01 inventory; verification lines in the a01/a08 run logs):

| choice | evidence |
|---|---|
| Aperture symmetric about the CPHD `ReferenceTime` (0.69911 s) | `TStartProc/TEndProc = 0.16989/1.22701`, centre 0.69845 (0.7 ms off); `Kaz1 = −Kaz2 = 0.44970` exactly; `SCPTime = PolarAngRefTime = ReferenceTime` |
| Range band trimmed from the top so that **ground-range IRW = 1.000 m** | `MinProc = FxMin`; `MaxProc = 9675.21 MHz` (187.8 of 225.2 MHz); `ImpRespWid_row / cos(GrazeAng) = 0.9999999989` |
| Uniform weighting, k = 0.8859 | `ImpRespWid·ImpRespBW = 0.8859` both axes; `WgtType = UNIFORM`; flat in-band spectrum (a07-style measurement, std/mean 0.046) |
| Oversampling exactly 1.25 | `SS·ImpRespBW = 0.8000` both axes |
| `Row/KCtr = (Krg1+Krg2)/2 = 63.9195`, `Col/KCtr = 0`, `DeltaKCOAPoly = 0` | direct |
| Footprint 6655 × 7500 px = **4248.7 m (slant range) × 6671.1 m (azimuth)**, SCP at the centre pixel | `NumRows·SS`, `NumCols·SS`; ICPs in CPHD ground coordinates ≈ (±3000 m along uIAX, −3000…+3750 m along uIAY) — larger than the 5 km `ImageArea` and than the CPHD `ImageGrid` (6400 × 0.782 m) |
| Focus plane = geodetic up at the SCP | `FPN = (−0.0812, −0.6784, 0.7302)` = WGS-84 normal at the SCP LLH |
| Radiometric 5×5 polynomials, `ImageBeamComp = SV`, Antenna and ErrorStatistics blocks present | direct |

diffpfa's product for the same CPHD: identical `Row/Col` unit vectors (0.0000°), identical SCP, 0.00 m pixel registration (a12), full aperture (`Kaz = −0.463/+0.604`) and full band, so `ImpRespWid` should read 0.588 m × 0.830 m (0.8859/BW) against the vendor's 0.707 m × 0.985 m; footprint 5000 × 5000 m.

To reproduce the vendor's product design: keep pulses with `|t − t_ref| ≤ min(t_ref − t_first, t_last − t_ref)`; keep fast-time samples with `F ≤ F_first + (c/2)·0.8859·cos(graze)/1.0 m`; set the slant-plane image extent to 4249 m × 6671 m (its ground meaning is ≈ 6.0 km × 6.7 km). Whether the high-side processor follows the same rules cannot be determined from here; these are the rules that reproduce the gold file's XML to the last digit.

---

## 7. Recommendations, in priority order

1. **Remove the inter-channel phase rotation (F1)** and add `test_subband_coherence_independent_of_burst_timing`. Blocks trustworthy stepped-chirp imagery; nothing else depends on it. Consider replacing `simulation/stepped_chirp_simulation.py` with (or extending it by) the receiver-level model in a14, whose default Δτ is not an integer number of cycles. On the real multi-step data, run experiment [3] of a14 first.
2. **Port `corrected_metadata()` into `_write_sicd` (F2–F4, F10, F16)** and add a sicdcheck-based test (`test_product_passes_sarkit_consistency_and_declares_uniform_irw`). Independent of 1.
3. **Apply `SGN` (F6)** and **normalise the CZT resampler (F8)** — both one-liners, both latent for the target data (multi-step, possibly SGN=+1 producers). Do 8 before any radiometric work.
4. **Provide a tensor-native entry point (F9)** for the exploit generator; keep `pfa_per_polar` as the numpy wrapper. Add `test_torch_chain_is_differentiable_and_adjoint_consistent` to CI.
5. **Fix or remove the rotated-axes branch and GROUND mode (F5)**; if kept, return spacings not counts.
6. **Honour non-symmetric image areas (F7)** — needed before the roadmap's patch/SCP-shift work, which is the same phase-shift machinery.
7. **Delete RVP deskew (F11), mask `SIGNAL ≠ 1` (F12), pin float64 in the CZT (F13).**
8. **Strengthen tests (F14):** adopt the proposed tests; keep a05 to prove new tests can fail.

---

## 8. Hypotheses I formed and then disproved (stress-tested claims)

1. *"The NUFFT without grid oversampling broadens the IPR 2× off-centre."* My first width measurement (linear interpolation on 1.25×-sampled data) said so (a02, third target). With 16× FFT upsampling and a control on the ideal reference, widths and PSLR are identical to the reference at every position (a03). The instrument was wrong, not the code.
2. *"The CZT resampler's `1/(L·Δk_in)` gain makes image amplitude depend on the image extent."* Predicted 1.6× between L = 20 and 40 m; measured 1.0000 (a02 §2). The extra k-space gain is exactly cancelled by the `N_r ∝ L` bins summed by the IFFT. The gain does depend on `Δk_in` (F8 stands).
3. *"The slant focus plane defocuses ground scatterers at the image edge."* Order-of-magnitude estimate suggested ~0.3 cycles; exact computation from the PVPs gives ≤ 0.005 cycles (a10).
4. *"The CZT runs in float32 and injects degrees of phase noise."* It would, but type promotion makes it complex128 (a11 A). Recorded as fragility (F13), not defect.
5. *"Vendor `TStartProc = 2·t_ref − TEndProc`."* Off by 1.3 ms; the aperture centre matches `t_ref` to 0.7 ms. Accepted as "symmetric about `t_ref`" at that tolerance.
6. *"δτ = 137 µs will expose F1."* It did not: 200 MHz × any integer number of µs is an integer number of cycles. 137.3217 µs did. Recorded because the shipped simulation fell into the same trap.
7. *First run of a14 reported 115° inter-band phase on a compliant channel and a "measured" fix that did nothing.* Both were my simulation's fault: the rotation-OFF switch used the global centre frequency instead of the one `pfa_per_polar` derives from the channels it is given, so imaging one channel alone re-enabled the rotation. Fixed and re-run; the compliant phases are 0.0°, and the measured fix restores 0.9988.

---

## 9. Scope, limits, and what was not done

- The 2 GB gold CPHD was not re-processed by me; the analysed product is `workspace/output/2025-10-26-05-00-15_UMBRA-08_SICDU_V_V.nitf`, written 2026-08-29 14:32, after the audited commit (13:58), so it reflects the audited code.
- GROUND mode was examined only far enough to establish F5; ATR consumes slant-plane imagery.
- Older vendor SICDs (2023) were used only as measurement controls (their k = 0.8857 and geometry are independently valid), never as image-formation references.
- No performance audit beyond recording one run.
- TOA-domain CPHD, bistatic collections, and `PI/` were out of scope.
- The stepped-chirp and multi-polarisation code paths were tested only synthetically; no such real data exists here.

---

## Appendix A — Script index

| script | purpose | key outputs |
|---|---|---|
| a01_inventory.py | metadata of all 6 CPHD / vendor SICD / diffpfa products, SIGNAL flags | `a01_inventory.json` |
| a02_forward_model_vs_exact.py | image vs ideal-PFA and back-projection references; amplitude scaling; asymmetric area | `out/a02_results.json` |
| a03_ipr_vs_position.py | IRW/PSLR/amplitude vs position with 16× upsampling and control; deconv gain; rotated branch | `out/a03_results.json` |
| a04_gradient_check.py | API vs kernel differentiability, gradcheck, adjoint, static scan | `out/a04_results.json` |
| a05_test_bite.py | mutation testing of shipped tests | `out/a05_results.json` |
| a06_subband_phase_correction.py | F1 measurement with the shipped simulation | `out/a06_results.json` |
| a07_real_data_run_and_irw.py | fresh SLANT run, sicdcheck, pixel-measured BW/IRW with vendor control | `out/run_*/`, `out/a07_*.json` |
| a08_metadata_fix_demo.py | `corrected_metadata()` + sicdcheck 13→0 | `out/*_corrected.xml`, `out/a08_*.json` |
| a09_ground_mode_real_data.py | GROUND run, rotated branch on real data | `out/run_ground_*/`, `out/a09_*.json` |
| a10_geometry_and_focus.py | geometry vs CPHD/vendor, ARPPoly, time origin, corner defocus | `out/a10_results.json` |
| a11_precision_and_multichannel_gain.py | CZT dtype/precision; unequal-SCSS sub-band weighting | `out/a11_results.json` |
| a12_gold_pair_registration.py | pixel registration diffpfa vs vendor gold | `out/a12_results.json`, PNG |
| a13_full_scale_vs_ideal.py | ideal-reference comparison at 5 km scale | stdout |
| a14_stepped_chirp_simulation_v2.py | receiver-level stepped-chirp simulation (explicit LO, CPHD compensation); rotation ON/OFF sweep; inter-band phase diagnostic; non-compliant producer and measured fix | `out/a14_results.json`, `out/a14_range_cuts.png` |
| a15_cross_check_other_audit.py | independent re-derivation of the agy/Gemini audit's checkable claims (1.20× framing on all 6 collections, radiometric ratios, ICP remedy vs gold, SCPCOA agreement) | `out/a15_results.json` |
| proposed_tests/test_diffpfa_audit.py | 12 regression tests (5 pass, 7 expected-fail today) | pytest |

## Appendix B — Normative references used
NGA.STND.0024-1 v1.5 SICD DIDD: Table 3-4 (pp. 27–30), Table 3-15 (pp. 74–75), §4.4 (pp. 86–89), §4.14.6 (pp. 161–164), §4.15 (pp. 165–172). NGA.STND.0068-1 v1.1.0 CPHD DIDD: §1.4 (pp. 9–11), Table 2-2 (p. 25), §4 (p. 35), §8.1 (pp. 115–116). Schema `schemas/SICD_schema_V1.3.0_2021_11_30.xsd`. sarkit 1.8.0 `verification/_sicd_consistency.py` (KAPFAC = 0.8859).
