# Verification of the remediation (commit `405b657`)

**Verifier:** Claude Code / Claude Fable 5.1, 2026-09-05
**Reference:** `audit/REMEDIATION_RESPONSE.md`, `audit/COMBINED_AUDIT_REPORT.md`
**Method:** every measurement script of the original audit (`a02`–`a14`) was re-run unchanged against the remediated code, all three test suites were run, all six regenerated products were passed through `sicdcheck`, and two new targeted checks (`a16_remediation_gaps.py`) were written for behaviours that no local data exercises. My proposed tests were confirmed unmodified before trusting their green result.

## 1. Verdict

The remediation is real and substantially complete. **Eight of the eight recommendations are implemented and independently confirmed**; the image-formation core is unchanged in accuracy; all six regenerated products pass `sicdcheck` with zero findings; the gold product still registers to the vendor's pixels to 0.00 m with the new framing. **Two gaps remain**, both in `_write_sicd`, both invisible on the local single-channel, SRP-centred data, and one of them matters for the project's stepped-chirp target: the k-space metadata is computed from the first channel of a polarisation group instead of the combined grid. Neither affects pixel values.

## 2. Confirmed fixed (independent measurement)

| item | before | after | evidence |
|---|---|---|---|
| C1 phase rotation | coherence 0.611 at Δτ = 137.3217 µs; 0.832 with PRI jitter | 0.9988 / 0.9985; rotation ON ≡ OFF at every Δτ | a06, a14 §2 |
| C2 metadata: sicdcheck | 13 error/warning lines on every product | **0** on all six regenerated products and on a fresh run | a07, `sicdcheck workspace/output/*.nitf` |
| ImpRespWid | k = 1.000 (12.9 % too wide) | k = 0.8859; declared/measured IRW = 0.996 (Row), 1.000 (Col) | a07 |
| KCtr (single channel) | 0 | 64.044 (Row), grid centre (Col; +0.0707 on 2025) | product summary |
| SCPTime / COA | mid-index pulse, first-pulse origin | 0.6991 s on 2025 = CPHD ReferenceTime; Grid/Type RGAZIM | product summary |
| ICPs | lat/lon box | ground projection (sicdcheck ICP checks pass) | a07 |
| C3 framing | 5000 × 5000 m slant box | 4242 × 6643 m on 2025 (vendor 4249 × 6671: 0.2 % / 0.4 %) | product summary |
| F7 SRP centring | SRP at area-centre pixel | SRP at (u=0, r=0) for u∈[0,40], r∈[−10,30] | a02 §4 |
| C5 rotated branch | Row/Col BW and SS swapped | consistent; GROUND product Row BW 0.6881 = k-extent along Row | a03 §3, a09 |
| C6 SGN | ignored | conjugated when +1 (test green) | proposed tests |
| C7 resampler gain | sub-band weight ∝ 1/SCSS (2.007×) | 1.010 (equal to the equal-sampling case) | a11 B |
| C8 radiometry | σ₀/β₀ = cos(graze) | σ₀/β₀ = 0.6964 = cos(slope) on 2025 (vendor 0.696361); γ₀ = σ₀/cos(inc) | product summary, code |
| C4 tensor entry | numpy only | `run_diffpfa_tensor` / `return_tensor`; gradcheck True, adjoint 5×10⁻¹⁵ | a04, `tests/test_differentiability.py` |
| C9 RVP / SIGNAL / float64 | dead branch; flags ignored; promotion-dependent | removed; masked; forced-float32 run now −53 dB from double (was −21 dB) | a11 A |
| image accuracy unchanged | −59 dB vs ideal, PSLR −13.30, exact localisation | identical | a02, a03 |
| gold registration | 0.00 m | 0.00 m at all 7 chips on the regenerated 8000 × 8910 product | a12 |
| simulation default | Δτ = 150 µs (integer cycles) | 137.3217 µs | `simulation/stepped_chirp_simulation.py:27` |
| tests | 8 | 13 in `tests/` + 12 + 4 audit tests, all green | pytest |

## 3. Remaining gaps

### G1 — MEDIUM (stepped-chirp) — k-space metadata computed from the first channel only
`IFP.py:283-327`: `_write_sicd` recomputes `Ku, Kr` from `ref_pvp = channel_pvps[0]` and `num_samples = channel_signals[0].shape[1]`, then writes `KCtr = (min+max)/2`, `Krg1/2`, `Kaz1/2` from that channel. `pfa_per_polar` formed the image on the grid centred at `(gk_min+gk_max)/2` over **all** channels of the group. For one channel the two coincide (which is why every local product is clean); for a stepped-chirp group they do not.

Measurement (a16 §1, three synthetic 200 MHz sub-bands, realistic spaceborne geometry, written with the remediated `_write_sicd`):

| field | grid actually used | written |
|---|---|---|
| Row/KCtr | 64.0415 | **62.7073** (centre of sub-band 0) |
| Krg1 / Krg2 | 62.0426 / 66.0405 | 62.0426 / **63.3720** |
| Row/ImpRespBW | 3.9979 | 3.9979 (correct: passed in from `pfa_per_polar`) |

`sicdcheck`: `[Error] Row IPR bandwidth supported by Krg`, `[Error] Col IPR bandwidth supported by Kaz`. Control with one full-band channel: 0 findings. Pixel data are unaffected; the XML would describe the sub-band-0 spectrum as the image spectrum.

*Fix.* Return `gku_ctr, gkr_ctr, gku_min/max, gkr_min/max` from `pfa_per_polar` (they are already computed there) and use them for `KCtr`, `Krg1/2`, `Kaz1/2`; alternatively compute `Ku, Kr` in `_write_sicd` over every channel of the group. The polar-angle and KSF polynomial fits from the reference channel are fine. Add a multi-channel `_write_sicd` + `sicdcheck` test (a16 §1 is one).

### G2 — LOW (latent) — `SCPPixel` still `(N//2, N//2)` after the SRP-centring fix
`IFP.py:239-241, 517-518`. The image is now centred on the framing-box centre `(u_c, r_c)`, so the SRP sits at pixel `(N_r/2 − r_c/Δr, N_u/2 − u_c/Δu)`, but `SCPPixel` and the ICP projection assume the centre pixel. a16 §2: with `u∈[0,40], r∈[−10,30]` the SRP is at (row 50, col 0) while the XML says (100, 60): 10 m range, 20 m azimuth mis-registration. With the new SLANT framing `(u_c, r_c) ≠ 0` only when IARP ≠ SRP or the ImageArea is not centred on the IARP; true for none of the six local CPHDs (a12 confirms 0.00 m on the gold product), so this is latent. *Fix.* `scp_row = round(N_r/2 − r_c/Δr)`, `scp_col = round(N_u/2 − u_c/Δu)`, pass `(u_c, r_c)` from `_determine_spatial_bounds` into `_write_sicd`, and use the same pixel for the ICP projection.

## 4. Notes, not defects

- `diffpfa/sicd_geometry.py` (`compute_scp_geometry`, `fit_arp_poly`, `compute_pfa_metadata`) is no longer used by the pipeline; only `tests/test_geometry.py` imports it. Either delete it (and retarget the geometry test at the product's SCPCOA, which sarkit now computes) or keep it as a reference implementation, but it should not be mistaken for production code. `a10`'s "consumer ARP(SCPTime) 32–58 m" line now measures this dead function, not the product.
- The tensor path returns complex128 images (`_apply_ifft_and_deconv` divides a complex64 image by a float64 vector); for the 8000 × 8910 gold product that is 1.1 GB instead of 0.57 GB. Cast `deconv` to the image's real dtype if memory matters.
- Absolute image scale: the resampler normalisation is now amplitude-preserving in k-space, so the image peak of a unit scatterer is independent of `SCSS` (good) but proportional to the range extent `L_r` (a02 §2: 0.5 / 1.0 / 2.0 for L = 20 / 40 / 80 m). Before, it was independent of `L` and proportional to `N_samples`. Neither is a calibrated scale; `BetaZeroSFPoly = 1/(ΔrΔu)` tracks neither. Relevant only when the roadmap's radiometric calibration is done.
- GROUND mode still produces three `sicdcheck` warnings (Row OSR 1.02 < 1.1 because the CPHD's suggested ground spacing is used; `TxFrequencyProc` bounds vs the sample grid). Low priority per project goals.
- The `REMEDIATION_RESPONSE.md` wording differs from the code in two places, harmlessly: `TimeCOAPoly` is set to the zero of the polar angle about the Row axis (correct), not "the instant when the look vector is orthogonal to velocity"; `ARPPoly` is fitted in absolute time (correct), not "(t − t_mid)". The README's "exactly reproducing vendor slant extents" holds for the gold processor version only (ratios 1.04–1.21 on the 2023 collections, a15).

## 5. Scripts

`a16_remediation_gaps.py` (new; G1 and G2), plus re-runs of `a02`, `a03`, `a04`, `a06`, `a07`, `a09`, `a10`, `a11`, `a12`, `a14`; outputs refreshed under `out/`.
