# diffpfa Roadmap and Future Concepts

This document serves to capture ideas, mathematical concepts, and planned features for the `diffpfa` processing pipeline.

## 1. Slant-Plane Processing

Currently, the image formation process projects data onto the ground plane. However, forming images directly on the slant plane is mathematically simpler (avoids ground-projection skew) and often desired by analysts. 

To form images on the slant plane, the unit vectors for the image area axes (`uIAX`, `uIAY`) must be derived from the Line-of-Sight (LoS) vector and the sensor velocity vector.

**Proposed Logic:**
```python
def _get_image_plane_vectors(cphd_meta: CPHDMetadata, image_plane: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_plane == "Slant":
        srp = cphd_meta.srp_ecf
        arp = cphd_meta.arp_pos_coa
        arp_v = cphd_meta.arp_vel_coa
        
        # Line of sight vector
        p_vec = srp - arp
        u_row = p_vec / np.linalg.norm(p_vec)
        
        # Cross-range vector (orthogonal to LoS and velocity)
        u_v = arp_v / np.linalg.norm(arp_v)
        u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
        u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
        
        uIAX = torch.as_tensor(u_col, dtype=torch.float64, device=device)
        uIAY = torch.as_tensor(u_row, dtype=torch.float64, device=device)
    else:
        uIAX = torch.as_tensor(cphd_meta.uIAX, dtype=torch.float64, device=device)
        uIAY = torch.as_tensor(cphd_meta.uIAY, dtype=torch.float64, device=device)
    return uIAX, uIAY
```

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
