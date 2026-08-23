def _get_image_plane_vectors(cphd_meta: CPHDMetadata, image_plane: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_plane == "Slant":
        srp = cphd_meta.srp_ecf
        arp = cphd_meta.arp_pos_coa
        arp_v = cphd_meta.arp_vel_coa
        p_vec = srp - arp
        u_row = p_vec / np.linalg.norm(p_vec)
        u_v = arp_v / np.linalg.norm(arp_v)
        u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
        u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
        uIAX = torch.as_tensor(u_col, dtype=torch.float64, device=device)
        uIAY = torch.as_tensor(u_row, dtype=torch.float64, device=device)
    else:
        uIAX = torch.as_tensor(cphd_meta.uIAX, dtype=torch.float64, device=device)
        uIAY = torch.as_tensor(cphd_meta.uIAY, dtype=torch.float64, device=device)
    return uIAX, uIAY

def _apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY):
    """
    if we were to process this in patches we would need this
    with something like:

    signal = _deskew_rvp(signal, pvp, N_samples, device)

    P_vecs = _compute_look_vectors(pvp, device=device)
    F_hz = _compute_fasttime_frequencies(pvp, N_samples, domain_type, device=device)
    
    uIAX, uIAY = _get_image_plane_vectors(cphd_meta, image_plane, device)

    u_c = (u_min + u_max) / 2.0
    r_c = (r_min + r_max) / 2.0 
    sig_patch, P_patch = _apply_scp_shift(signal, F_hz, P_vecs_orig, u_c, r_c, uIAX, uIAY) # why
    
    cos_theta, sin_theta = _compute_look_components(P_vecs, uIAX, uIAY)
    F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
    Ku = F_cpm * cos_theta.unsqueeze(1)
    Kr = F_cpm * sin_theta.unsqueeze(1)
    
    """
    P_patch = P_vecs_orig + u_c * uIAX + r_c * uIAY
    R_orig = torch.linalg.norm(P_vecs_orig, dim=-1)
    R_patch = torch.linalg.norm(P_patch, dim=-1)
    dR = R_patch - R_orig
    phi_corr = (4.0 * torch.pi / SPEED_OF_LIGHT) * F_hz * dR.unsqueeze(1)
    corr_term = torch.exp(torch.complex(torch.zeros_like(phi_corr), phi_corr))
    sig_patch = signal * corr_term.to(signal.dtype)
    return sig_patch, P_patch

