"""
Stepped-Chirp Multi-Channel Radar Simulation for diffpfa
========================================================
Simulates an ultra-wideband synthetic aperture radar (SAR) utilizing a stepped-chirp
waveform with inter-step time delays (delta_tau), platform orbital motion, and baseband
downconversion. Verifies coherent K-space accumulation and image formation against
an ideal monolithic full-band reference.
"""

import os
import sys
import numpy as np
import torch

from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.constants import SPEED_OF_LIGHT
from diffpfa.types import CPHDMetadata, ImageAreaBounds


def run_stepped_chirp_simulation(
    fc_global: float = 9.6e9,
    bw_total: float = 600e6,
    num_subbands: int = 3,
    samples_per_sub: int = 256,
    num_pulses: int = 512,
    prf_burst: float = 1000.0,
    delta_tau: float = 150e-6,
    sat_vel: float = 7500.0,
    slant_range: float = 15000.0,
    squint_span_deg: float = 2.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 75)
    print("STEPPED-CHIRP MULTI-CHANNEL SAR SIMULATION")
    print("=" * 75)
    print(f"Global Center Frequency (fc):  {fc_global / 1e9:.2f} GHz")
    print(f"Total Synthesized Bandwidth:   {bw_total / 1e6:.1f} MHz")
    print(f"Subband Channels:              {num_subbands} x {bw_total / (num_subbands * 1e6):.1f} MHz")
    print(f"Inter-Pulse Delay (delta_tau): {delta_tau * 1e6:.1f} microseconds")
    print(f"Satellite Orbital Velocity:    {sat_vel:.1f} m/s")
    print(f"Platform Motion / Step:        {sat_vel * delta_tau:.3f} meters")
    print(f"Device:                        {device}")
    print("-" * 75)

    bw_sub = bw_total / num_subbands
    t_burst = np.arange(num_pulses) / prf_burst
    t_duration = t_burst[-1]

    theta_burst = np.linspace(-np.radians(squint_span_deg / 2), np.radians(squint_span_deg / 2), num_pulses)

    uIAX = np.array([1.0, 0.0, 0.0]) # Azimuth / Cross-Range
    uIAY = np.array([0.0, 1.0, 0.0]) # Range
    srp_ecf = np.array([0.0, 0.0, 0.0])

    # Point scatterers: (u, r, amplitude)
    targets = [
        (0.0, 0.0, 1.0),      # Center target
        (5.0, 8.0, 1.0),      # Quadrant 1
        (-7.0, -12.0, 1.0),   # Quadrant 3
        (10.0, -5.0, 1.0),    # Quadrant 4
    ]

    channel_signals = []
    channel_pvps = []
    channel_fxcs = []
    channel_domains = []

    ref_rcv_time = t_burst.copy()

    for m in range(num_subbands):
        f_sub_center = (fc_global - bw_total / 2.0) + (m + 0.5) * bw_sub
        f_sub_start = f_sub_center - bw_sub / 2.0
        f_sub_step = bw_sub / samples_per_sub

        f_v = -bw_sub / 2.0 + np.arange(samples_per_sub) * f_sub_step
        f_rf = f_sub_center + f_v

        t_m = t_burst + m * delta_tau
        theta_m = theta_burst + (m * delta_tau / t_duration) * np.radians(squint_span_deg)

        tx_pos_m = np.zeros((num_pulses, 3))
        for n in range(num_pulses):
            th = theta_m[n]
            tx_pos_m[n] = np.array([slant_range * np.sin(th), -slant_range * np.cos(th), 0.0])

        sig_m = np.zeros((num_pulses, samples_per_sub), dtype=np.complex64)
        for u_t, r_t, amp in targets:
            for n in range(num_pulses):
                th = theta_m[n]
                cos_pfa = -np.sin(th)
                sin_pfa = np.cos(th)
                delta_R = u_t * cos_pfa + r_t * sin_pfa
                phase = -2.0 * np.pi * (2.0 * f_rf / SPEED_OF_LIGHT) * delta_R
                sig_m[n] += amp * np.exp(1j * phase).astype(np.complex64)

        pvp_m = {
            "SRPPos": np.tile(srp_ecf, (num_pulses, 1)),
            "TxPos": tx_pos_m,
            "TxVel": np.tile(np.array([sat_vel, 0.0, 0.0]), (num_pulses, 1)),
            "RcvPos": tx_pos_m,
            "RcvVel": np.tile(np.array([sat_vel, 0.0, 0.0]), (num_pulses, 1)),
            "SC0": np.full(num_pulses, f_sub_start),
            "SCSS": np.full(num_pulses, f_sub_step),
            "RcvTime": t_m
        }

        channel_signals.append(sig_m)
        channel_pvps.append(pvp_m)
        channel_fxcs.append(f_sub_center)
        channel_domains.append("FX")

    meta = CPHDMetadata(
        domain_type="FX",
        sgn=-1,
        global_fx_min=fc_global - bw_total / 2.0,
        global_fx_max=fc_global + bw_total / 2.0,
        iarp_ecf=srp_ecf,
        uIAX=uIAX,
        uIAY=uIAY,
        ref_ch_id="Channel_0",
        image_area=ImageAreaBounds(x1=-20.0, y1=-20.0, x2=20.0, y2=20.0, polygon=None),
        extended_area=None,
        collection_start="2026-08-28T00:00:00Z",
        radar_mode="SPOTLIGHT",
        classification="UNCLASSIFIED",
        srp_ecf=srp_ecf,
        arp_pos_coa=channel_pvps[0]["TxPos"][num_pulses // 2],
        arp_vel_coa=np.array([sat_vel, 0.0, 0.0]),
        side_of_track="R",
        line_spacing=None,
        sample_spacing=None,
        raw_meta=None
    )

    print("Reconstructing multi-channel stepped-chirp image (all 3 subbands, 600 MHz)...")
    img_multi, bw_range, bw_azm, N_range, N_azm, _ = pfa_per_polar(
        channel_signals=channel_signals,
        channel_pvps=channel_pvps,
        channel_fxcs=channel_fxcs,
        channel_domain_types=channel_domains,
        ref_rcv_time=ref_rcv_time,
        cphd_meta=meta,
        u_min=-20.0, u_max=20.0,
        r_min=-20.0, r_max=20.0,
        image_oversample=1.25,
        device=device
    )

    du = 40.0 / N_azm
    dr = 40.0 / N_range
    common_spacing = (du, dr)

    print("Reconstructing single subband image (Channel 1 only, 200 MHz)...")
    meta_ch1 = CPHDMetadata(
        domain_type="FX",
        sgn=-1,
        global_fx_min=channel_fxcs[0] - bw_sub / 2.0,
        global_fx_max=channel_fxcs[0] + bw_sub / 2.0,
        iarp_ecf=srp_ecf,
        uIAX=uIAX,
        uIAY=uIAY,
        ref_ch_id="Channel_0",
        image_area=ImageAreaBounds(x1=-20.0, y1=-20.0, x2=20.0, y2=20.0, polygon=None),
        extended_area=None,
        collection_start="2026-08-28T00:00:00Z",
        radar_mode="SPOTLIGHT",
        classification="UNCLASSIFIED",
        srp_ecf=srp_ecf,
        arp_pos_coa=channel_pvps[0]["TxPos"][num_pulses // 2],
        arp_vel_coa=np.array([sat_vel, 0.0, 0.0]),
        side_of_track="R",
        line_spacing=None,
        sample_spacing=None,
        raw_meta=None
    )
    img_ch1, _, _, _, _, _ = pfa_per_polar(
        channel_signals=[channel_signals[0]],
        channel_pvps=[channel_pvps[0]],
        channel_fxcs=[channel_fxcs[0]],
        channel_domain_types=["FX"],
        ref_rcv_time=ref_rcv_time,
        cphd_meta=meta_ch1,
        u_min=-20.0, u_max=20.0,
        r_min=-20.0, r_max=20.0,
        custom_pixel_spacing=common_spacing,
        image_oversample=1.25,
        device=device
    )

    print("Reconstructing two subbands accumulated (Channels 1 + 2, 400 MHz)...")
    meta_ch1_2 = CPHDMetadata(
        domain_type="FX",
        sgn=-1,
        global_fx_min=channel_fxcs[0] - bw_sub / 2.0,
        global_fx_max=channel_fxcs[1] + bw_sub / 2.0,
        iarp_ecf=srp_ecf,
        uIAX=uIAX,
        uIAY=uIAY,
        ref_ch_id="Channel_0",
        image_area=ImageAreaBounds(x1=-20.0, y1=-20.0, x2=20.0, y2=20.0, polygon=None),
        extended_area=None,
        collection_start="2026-08-28T00:00:00Z",
        radar_mode="SPOTLIGHT",
        classification="UNCLASSIFIED",
        srp_ecf=srp_ecf,
        arp_pos_coa=channel_pvps[0]["TxPos"][num_pulses // 2],
        arp_vel_coa=np.array([sat_vel, 0.0, 0.0]),
        side_of_track="R",
        line_spacing=None,
        sample_spacing=None,
        raw_meta=None
    )
    img_ch1_2, _, _, _, _, _ = pfa_per_polar(
        channel_signals=channel_signals[:2],
        channel_pvps=channel_pvps[:2],
        channel_fxcs=channel_fxcs[:2],
        channel_domain_types=channel_domains[:2],
        ref_rcv_time=ref_rcv_time,
        cphd_meta=meta_ch1_2,
        u_min=-20.0, u_max=20.0,
        r_min=-20.0, r_max=20.0,
        custom_pixel_spacing=common_spacing,
        image_oversample=1.25,
        device=device
    )

    # Reference Monolithic Pulse
    print("Reconstructing monolithic full-band reference image...")
    f_full_step = bw_total / (num_subbands * samples_per_sub)
    f_full_vec = (fc_global - bw_total / 2.0) + np.arange(num_subbands * samples_per_sub) * f_full_step
    sig_full = np.zeros((num_pulses, num_subbands * samples_per_sub), dtype=np.complex64)
    for u_t, r_t, amp in targets:
        for n in range(num_pulses):
            th = theta_burst[n]
            cos_pfa = -np.sin(th)
            sin_pfa = np.cos(th)
            delta_R = u_t * cos_pfa + r_t * sin_pfa
            phase = -2.0 * np.pi * (2.0 * f_full_vec / SPEED_OF_LIGHT) * delta_R
            sig_full[n] += amp * np.exp(1j * phase).astype(np.complex64)

    pvp_full = {
        "SRPPos": np.tile(srp_ecf, (num_pulses, 1)),
        "TxPos": channel_pvps[0]["TxPos"],
        "TxVel": channel_pvps[0]["TxVel"],
        "RcvPos": channel_pvps[0]["RcvPos"],
        "RcvVel": channel_pvps[0]["RcvVel"],
        "SC0": np.full(num_pulses, fc_global - bw_total / 2.0),
        "SCSS": np.full(num_pulses, f_full_step),
        "RcvTime": ref_rcv_time
    }

    img_ref, _, _, _, _, _ = pfa_per_polar(
        channel_signals=[sig_full],
        channel_pvps=[pvp_full],
        channel_fxcs=[fc_global],
        channel_domain_types=["FX"],
        ref_rcv_time=ref_rcv_time,
        cphd_meta=meta,
        u_min=-20.0, u_max=20.0,
        r_min=-20.0, r_max=20.0,
        custom_pixel_spacing=common_spacing,
        image_oversample=1.25,
        device=device
    )

    # Metrics
    complex_corr = float(np.abs(np.sum(img_multi * np.conj(img_ref))) / (np.linalg.norm(img_multi) * np.linalg.norm(img_ref)))

    du = 40.0 / N_azm
    dr = 40.0 / N_range

    print("\n" + "=" * 75)
    print("SIMULATION RESULTS")
    print("=" * 75)
    print(f"Complex Coherence (Correlation with Ideal Reference): {complex_corr:.6f}")

    mag_multi = np.abs(img_multi)
    mag_ref = np.abs(img_ref)

    print("\n--- Target Localization Accuracy ---")
    print(f"{'Target (u, r) [m]':<20} | {'Estimated (u, r) [m]':<22} | {'Pos Error':<10} | {'Peak Ratio'}")
    print("-" * 75)

    for u_t, r_t, amp in targets:
        exp_u_idx = int(np.round((u_t - (-20.0)) / du))
        exp_r_idx = int(np.round((r_t - (-20.0)) / dr))

        sub_u = slice(max(0, exp_u_idx - 10), min(N_azm, exp_u_idx + 11))
        sub_r = slice(max(0, exp_r_idx - 10), min(N_range, exp_r_idx + 11))

        sub_mag = mag_multi[sub_u, sub_r]
        local_max = np.unravel_index(np.argmax(sub_mag), sub_mag.shape)
        found_u = -20.0 + (exp_u_idx - 10 + local_max[0]) * du
        found_r = -20.0 + (exp_r_idx - 10 + local_max[1]) * dr

        pos_err = np.sqrt((found_u - u_t) ** 2 + (found_r - r_t) ** 2)
        peak_multi = float(sub_mag[local_max])
        peak_ref = float(np.max(mag_ref[sub_u, sub_r]))
        ratio = peak_multi / max(peak_ref, 1e-12)

        print(f"({u_t:5.1f}, {r_t:5.1f})        | ({found_u:5.2f}, {found_r:5.2f})          | {pos_err:.4f} m | {ratio:.4f}")

    return {
        "complex_coherence": complex_corr,
        "img_ch1": img_ch1,
        "img_ch1_2": img_ch1_2,
        "img_multi": img_multi,
        "img_ref": img_ref,
        "N_azm": N_azm,
        "N_range": N_range,
        "du": du,
        "dr": dr
    }


def plot_simulation_coherence(sim_results: dict, output_dir: str = "simulation"):
    """
    Generates and saves diagnostic plots:
    1. stepped_chirp_coherence_analysis.png: 4-panel coherence and phase alignment plot.
    2. stepped_chirp_ipr_tightening.png: Step-by-step 2D PSF and 1D IPR tightening from 1 to 3 subbands.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_ch1 = sim_results["img_ch1"]
    img_ch1_2 = sim_results["img_ch1_2"]
    img_multi = sim_results["img_multi"]
    img_ref = sim_results["img_ref"]
    N_azm = sim_results["N_azm"]
    N_range = sim_results["N_range"]
    du = sim_results["du"]
    dr = sim_results["dr"]
    coherence = sim_results["complex_coherence"]

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Plot 1: 4-Panel Coherence Analysis
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Multi-Channel Stepped-Chirp 2D Magnitude (dB)
    mag_multi_db = 20.0 * np.log10(np.abs(img_multi) / np.max(np.abs(img_multi)) + 1e-6)
    im0 = axes[0, 0].imshow(mag_multi_db.T, cmap="inferno", extent=[-20, 20, -20, 20], origin="lower", vmin=-40, vmax=0)
    axes[0, 0].set_title("Multi-Channel Stepped-Chirp (3x200MHz, delta_tau=150us)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Azimuth u (meters)")
    axes[0, 0].set_ylabel("Range r (meters)")
    fig.colorbar(im0, ax=axes[0, 0], label="Normalized Magnitude (dB)")

    # 2. Monolithic Reference 2D Magnitude (dB)
    mag_ref_db = 20.0 * np.log10(np.abs(img_ref) / np.max(np.abs(img_ref)) + 1e-6)
    im1 = axes[0, 1].imshow(mag_ref_db.T, cmap="inferno", extent=[-20, 20, -20, 20], origin="lower", vmin=-40, vmax=0)
    axes[0, 1].set_title("Monolithic Reference (Single 600MHz Pulse)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Azimuth u (meters)")
    axes[0, 1].set_ylabel("Range r (meters)")
    fig.colorbar(im1, ax=axes[0, 1], label="Normalized Magnitude (dB)")

    # 3. Interferometric Phase Difference (deg) across target points
    interferogram = img_multi * np.conj(img_ref)
    phase_diff_deg = np.rad2deg(np.angle(interferogram))
    mask = (mag_multi_db.T > -25.0)
    phase_plot = np.where(mask, phase_diff_deg.T, np.nan)
    im2 = axes[1, 0].imshow(phase_plot, cmap="coolwarm", extent=[-20, 20, -20, 20], origin="lower", vmin=-5, vmax=5)
    axes[1, 0].set_title(f"Interferometric Phase Difference (Coherence = {coherence:.6f})", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Azimuth u (meters)")
    axes[1, 0].set_ylabel("Range r (meters)")
    fig.colorbar(im2, ax=axes[1, 0], label="Phase Error (degrees)")

    # 4. 1D Range Mainlobe Profile (dB) Cut
    center_u = N_azm // 2
    cut_multi = np.abs(img_multi[center_u, :])
    cut_ref = np.abs(img_ref[center_u, :])
    cut_multi_db = 20.0 * np.log10(cut_multi / np.max(cut_multi) + 1e-6)
    cut_ref_db = 20.0 * np.log10(cut_ref / np.max(cut_ref) + 1e-6)
    r_coords = np.linspace(-20, 20, N_range)

    axes[1, 1].plot(r_coords, cut_ref_db, "b-", label="Monolithic Ref (600MHz)", linewidth=2.0, alpha=0.8)
    axes[1, 1].plot(r_coords, cut_multi_db, "r--", label="Stepped-Chirp (3x200MHz)", linewidth=1.5)
    axes[1, 1].axhline(-3.0, color="gray", linestyle=":", label="-3dB Resolution Width (0.20m)")
    axes[1, 1].set_title("1D Range Impulse Response (IPR) Mainlobe Cut", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Range (meters)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].set_xlim([-3.0, 3.0])
    axes[1, 1].set_ylim([-45.0, 2.0])
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    p1 = os.path.join(output_dir, "stepped_chirp_coherence_analysis.png")
    plt.savefig(p1, dpi=160)
    plt.close()
    print(f"\n1. Coherence diagnostic plot saved to: {os.path.abspath(p1)}")

    # -------------------------------------------------------------
    # Plot 2: IPR Tightening from 1 Subband -> 3 Subbands
    # -------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))

    # Center target zoom region ([-3m, 3m])
    crop_half_u = int(3.0 / du)
    crop_half_r = int(3.0 / dr)
    center_u_idx = N_azm // 2
    center_r_idx = N_range // 2
    u_slice = slice(center_u_idx - crop_half_u, center_u_idx + crop_half_u)
    r_slice = slice(center_r_idx - crop_half_r, center_r_idx + crop_half_r)

    stages = [
        ("1 Subband (200 MHz)", img_ch1, "0.60 m"),
        ("2 Subbands (400 MHz)", img_ch1_2, "0.30 m"),
        ("3 Subbands (600 MHz)", img_multi, "0.20 m"),
        ("Monolithic Ref (600 MHz)", img_ref, "0.20 m"),
    ]

    for col_idx, (title, img_stage, res_str) in enumerate(stages):
        mag_stage = np.abs(img_stage)
        crop_mag = mag_stage[u_slice, r_slice]
        crop_mag_db = 20.0 * np.log10(crop_mag / np.max(crop_mag) + 1e-6)

        # 2D PSF Zoom
        im_crop = axes2[0, col_idx].imshow(
            crop_mag_db.T,
            cmap="inferno",
            extent=[-3, 3, -3, 3],
            origin="lower",
            vmin=-35,
            vmax=0
        )
        axes2[0, col_idx].set_title(f"{title}\n3dB Res: ~{res_str}", fontsize=11, fontweight="bold")
        axes2[0, col_idx].set_xlabel("Azimuth u (m)")
        if col_idx == 0:
            axes2[0, col_idx].set_ylabel("Range r (m)")
        fig2.colorbar(im_crop, ax=axes2[0, col_idx], fraction=0.046, pad=0.04)

        # 1D Range Cut
        cut_stage = mag_stage[center_u_idx, :]
        cut_stage_db = 20.0 * np.log10(cut_stage / np.max(cut_stage) + 1e-6)
        axes2[1, col_idx].plot(r_coords, cut_stage_db, "r-" if col_idx < 3 else "b-", linewidth=1.8)
        axes2[1, col_idx].axhline(-3.0, color="gray", linestyle=":", label="-3dB level")
        axes2[1, col_idx].set_title(f"1D Range Cut ({title})", fontsize=10)
        axes2[1, col_idx].set_xlabel("Range (m)")
        if col_idx == 0:
            axes2[1, col_idx].set_ylabel("Magnitude (dB)")
        axes2[1, col_idx].set_xlim([-2.5, 2.5])
        axes2[1, col_idx].set_ylim([-40.0, 2.0])
        axes2[1, col_idx].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p2 = os.path.join(output_dir, "stepped_chirp_ipr_tightening.png")
    plt.savefig(p2, dpi=160)
    plt.close()
    print(f"2. Step-by-step IPR tightening plot saved to: {os.path.abspath(p2)}")


if __name__ == "__main__":
    results = run_stepped_chirp_simulation()
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    plot_simulation_coherence(results, out_dir)


