# diffpfa Roadmap and Future Concepts

This document captures implemented capabilities, mathematical concepts, and planned features for the `diffpfa` processing pipeline.

---

## Completed Capabilities

### 4. Radiometric Calibration (Relative)
---

## Active Roadmap & Future Concepts

### 1. Patch-Based Processing (SCP Shift)
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

### 2. Absolute Radiometric Calibration
- Currently, populates mathematically balanced relative `<Radiometric>` scale factor polynomials (`BetaZeroSFPoly`, `SigmaZeroSFPoly`, `GammaZeroSFPoly`) preserving power ratios across Slant, Ground, and LOS planes.
- Extract transmit power, antenna gain patterns, and receiver noise floor from the CPHD to compute absolute `RCSSFPoly` gain, replacing the uncalibrated `1.0` placeholder to convert complex pixel values into physical radar cross section square-meters ($m^2$).

### 3. Optimization: On-Demand PyTorch Memory Strategy
Currently, `torch.cuda.empty_cache()` is called unconditionally at the end of each channel loop to prevent memory fragmentation and ensure OOM-safety.

**Future Optimization:**
Wrap the channel loop in a `try...except torch.cuda.OutOfMemoryError:` block. If PyTorch's caching allocator fails to find a contiguous block, catch the exception, explicitly call `empty_cache()`, and retry the channel. This delivers the speed of an untampered cache while retaining the OOM safety net.

### 4. Dependency Consolidation (sarkit vs sarpy)
The core processor utilizes `sarkit` as its primary I/O backend for standard compliance. However, `tools/convert2png.py` relies on `sarpy` for its optimized density remapper.

**Plan:**
Consolidate visualization utilities into a standalone, lightweight pure-Python module to remove the secondary `sarpy` dependency.
