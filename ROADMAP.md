# diffpfa Roadmap and Future Concepts

This document serves to capture ideas, mathematical concepts, and planned features for the `diffpfa` processing pipeline.

## 2. Patch-Based Processing (SCP Shift)

Processing exceedingly large scenes may require breaking the image into smaller tiles or "patches" to fit within GPU VRAM limits. This requires phase-shifting the data's Scene Center Point (SCP) to the center of each patch.

*Note: Doing this prior to K-space combination currently breaks the shared Cartesian grid assumption. The architecture will need to be adapted to accommodate per-patch K-spaces.*

**Proposed Logic:**
```python
def _apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY):
    """
    Shifts the phase center of the signal to a new scene center point (u_c, r_c).
    """
    P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
    
    R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
    R_patch = torch.linalg.norm(P_patch, dim=-1)
    dR = R_patch - R_orig
    
    # Phase correction for the differential range
    phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
    corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
    
    sig_patch = signal * corr_term.to(signal.dtype)
    return sig_patch, P_patch
```

## 3. Radiometric Calibration
`diffpfa` now generates a mathematically balanced, relative `<Radiometric>` block. Because absolute system transmit powers and range losses are not currently extracted from the CPHD, the `RCSSFPoly` (Radar Cross Section Scale Factor) defaults to a flat `1.0`.

However, the geometric scale factors are correctly balanced using the pixel area and grazing angles:
- `BetaZeroSFPoly` ($\beta^0$): `1.0 / (RowSpacing * ColSpacing)`
- `SigmaZeroSFPoly` ($\sigma^0$): `BetaZeroSFPoly * cos(GrazeAng)`
- `GammaZeroSFPoly` ($\gamma^0$): `BetaZeroSFPoly * sin(GrazeAng)`

This prevents strict downstream ATR apps from crashing and ensures that the physical ratio of power between the Slant, Ground, and LOS planes is completely preserved!

**Backlog (Absolute Calibration):**
- Extract transmit power, antenna patterns, and receiver noise from the CPHD to calculate the absolute `RCSSFPoly` gain, replacing the `1.0` placeholder and converting the pixel values into true square-meters ($m^2$).

## 4. Full IPR Metadata Generation
To fully support downstream exploitation algorithms (like CLEAN and autofocus), the output SICD must mathematically describe the space-variant 2D Impulse Response (IPR) of the image. 

Currently, `diffpfa` forms the image but does not write the detailed K-space and windowing metadata into the XML. 

**Plan:**
1. **Waveform Abstraction (The "Rect" Assumption)**: Regardless of the original waveform (LFM, Phase-coded), we assume the CPHD data has been perfectly pulse-compressed/matched-filtered and equalized into a flat spectrum (a "rect"). Therefore, the exact waveform does not need to be passed down; the IPR is exclusively defined by the K-space bandwidth limits and our applied digital weighting functions.
2. **Write Grid Parameters**: Populate `Grid/Row/WgtFunct` and `Grid/Col/WgtFunct` with the exact discrete arrays of the digital windows applied (e.g., Kaiser-Bessel) and write the nominal `ImpRespBW` limits.
3. **Write PFA Parameters**: Populate the `<PFA>` block with the base polar annulus limits (`Krg1/2`, `Kaz1/2`).
4. **Calculate Space-Variant Polynomials**: Generate and write `<PolarAngPoly>` and `<SpatialFreqSFPoly>` inside the `<PFA>` block. These tell downstream tools exactly how the K-space annulus rotates and scales across the image, allowing them to perfectly reconstruct the space-variant 2D IPR at any pixel coordinate.

## 5. Complete Geometric Metadata for Geolocation
- `SlantRange`, `GroundRange`, `DopplerConeAng`, `GrazeAng`, `IncidenceAng`, `AzimAng`, and `SlopeAng` have been mathematically implemented for the Scene Center Point at Center of Aperture (COA).

**Backlog / Future Implementation:**
- **TwistAng & LayoverAng**: Requires deriving the projection of the cross-range vector and vertical structure lean relative to the chosen image plane.
- **ARPPoly & ARPAcc**: Requires extracting the full Position Velocity Parameter (PVP) history (TxPos, TyPos, TzPos) across the dwell time and performing a polynomial fit, rather than just extracting the single COA point. Currently hardcoded to `0.0`.

## 6. Optimization: PyTorch Memory Strategy
The `pfa_per_polar` loop creates massive intermediate tensors (CZT and NUFFT grids) that quickly exceed VRAM limits on wideband, multi-channel datasets. 

Currently, `torch.cuda.empty_cache()` is called unconditionally at the end of each channel loop. While this heavily degrades processing speed (by forcing a CUDA synchronization and flushing the PyTorch caching allocator), it is **intentionally left in place** to prevent memory fragmentation and ensure stability in production without OOM crashes.

**Future Optimization:**
Wrap the channel loop in a `try...except torch.cuda.OutOfMemoryError:` block. If PyTorch's caching allocator fails to find a contiguous block, catch the exception, explicitly call `empty_cache()`, and retry the channel. This would deliver the speed of an untampered cache while retaining the OOM safety net.

## 7. Dependency Consolidation (sarkit vs sarpy)
The core processor correctly utilizes `sarkit` as its primary I/O backend for strict standard compliance and reading/writing CPHD/SICD files. However, the `tools/` directory (specifically for visualization and PNG conversion) still relies on the older `sarpy` library to access its highly optimized density remapper. 

**Plan:**
Eventually consolidate the visualization stack. This could mean either migrating the density remapper logic into our own lightweight utility, or waiting for `sarkit` to implement an equivalent visualization module, allowing us to drop the dual `sarpy`/`sarkit` dependency entirely.
