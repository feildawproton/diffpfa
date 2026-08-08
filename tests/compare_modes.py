import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sarpy.io.complex.sicd import SICDReader

from diffpfa.algo import PFAConfig, PFAEngine
from diffpfa.io import CPHDReader, SICDWriter

def process_cphd(cphd_path, output_dir, device="cpu"):
    """Runs all three modes on the CPHD and returns paths to the generated SICD files."""
    modes = ["nufft", "hybrid", "czt"]
    output_files = {}

    reader = CPHDReader(cphd_path, backend="auto")
    writer = SICDWriter(backend="auto")

    import gc
    import torch
    for mode in modes:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n--- Running mode: {mode} ---")
        config = PFAConfig(
            mode=mode,
            output_dir=os.path.join(output_dir, mode),
            device=device,
            enable_compile=False # Disable compile to save time for single run
        )
        engine = PFAEngine(reader, writer, config)
        paths = engine.run()
        # Assume there's only one combined output file for simplicity
        output_files[mode] = paths[0] 

    return output_files

def compare_images(paths, artifact_dir):
    """Compares CZT and Hybrid to NUFFT, and saves error plots."""
    print("\n--- Comparing Images ---")
    
    # Load complex data
    print(f"Loading NUFFT (Reference)...")
    with SICDReader(paths["nufft"]) as reader:
        img_nufft = reader[:]

    print(f"Loading Hybrid...")
    with SICDReader(paths["hybrid"]) as reader:
        img_hybrid = reader[:]

    print(f"Loading CZT...")
    with SICDReader(paths["czt"]) as reader:
        img_czt = reader[:]

    # We skip edge pixels to avoid division by zero or noise where signal is zero
    mask = np.abs(img_nufft) > 1e-10

    # Calculate Phase Differences
    print("Calculating Phase Differences...")
    phase_diff_hybrid = np.zeros_like(img_nufft, dtype=np.float32)
    phase_diff_czt = np.zeros_like(img_nufft, dtype=np.float32)

    phase_diff_hybrid[mask] = np.angle(img_hybrid[mask] * np.conj(img_nufft[mask]))
    phase_diff_czt[mask] = np.angle(img_czt[mask] * np.conj(img_nufft[mask]))

    # Plotting
    print("Generating Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Convert radians to degrees for readability
    im0 = axes[0].imshow(np.degrees(phase_diff_czt), cmap='RdBu', vmin=-180, vmax=180, aspect='auto')
    axes[0].set_title('Phase Error: CZT vs NUFFT (Degrees)')
    axes[0].set_xlabel('Range')
    axes[0].set_ylabel('Cross-Range')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.degrees(phase_diff_hybrid), cmap='RdBu', vmin=-180, vmax=180, aspect='auto')
    axes[1].set_title('Phase Error: Hybrid vs NUFFT (Degrees)')
    axes[1].set_xlabel('Range')
    axes[1].set_ylabel('Cross-Range')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    plot_path = os.path.join(artifact_dir, "phase_error_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cphd", type=str, required=True, help="Path to input CPHD")
    parser.add_argument("--out_dir", type=str, default="test_outputs")
    parser.add_argument("--artifact_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    paths = process_cphd(args.cphd, args.out_dir, device=args.device)
    compare_images(paths, args.artifact_dir)
