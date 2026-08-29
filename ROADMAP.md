# diffpfa Roadmap and Future Concepts

This document tracks upcoming architectural improvements and future milestones for the `diffpfa` processing pipeline.

---

## 1. Patch-Based Processing (SCP Shift)
Processing exceedingly large scenes may require breaking the image into smaller tiles or "patches" to fit within GPU VRAM limits. This requires phase-shifting the data's Scene Center Point (SCP) to the center of each patch.

*Note: Doing this prior to K-space combination breaks the single global Cartesian grid assumption. The architecture will need to be adapted to accommodate per-patch K-spaces.*

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

---

## 2. Absolute Radiometric Calibration
Currently, `diffpfa` generates mathematically balanced, relative `<Radiometric>` scale factor polynomials (`BetaZeroSFPoly`, `SigmaZeroSFPoly`, `GammaZeroSFPoly`), preserving power ratios across Slant, Ground, and LOS planes.

**Future Goal:**
Extract transmit power, antenna gain patterns, and receiver noise floor from the CPHD header/PVP to compute absolute `RCSSFPoly` gain, replacing the uncalibrated `1.0` placeholder to convert complex pixel values into physical radar cross section square-meters ($m^2$).

