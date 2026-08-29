"""
SICD to 8-bit PNG Converter
===========================
Converts complex SAR imagery from SICD (NITF 2.1) format into 8-bit grayscale PNG images
using the standard SAR logarithmic density remapping algorithm.

Implementation Note:
The logarithmic density remapping formula implemented here is derived from the open-source
SARpy implementation (sarpy.visualization.remap.Density), which cites Kevin Mangis' 1994
publication "Softcopy Display of SAR Data". We gratefully acknowledge the SARpy authors
and NGA for documenting this formulation.
"""

import sys
import os
import numpy as np
from PIL import Image
import sarkit.sicd as sksicd


def density_remap_8bit(
    data: np.ndarray,
    dmin: float = 30.0,
    mmult: float = 40.0,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Applies standard monochromatic logarithmic density remapping to complex SAR imagery.
    
    Parameters
    ----------
    data : np.ndarray
        Complex or magnitude SAR image array.
    dmin : float
        Dynamic range minimum parameter (default 30.0).
    mmult : float
        Contrast multiplier parameter (default 40.0).
    eps : float
        Small offset to prevent log(0).
        
    Returns
    -------
    np.ndarray (uint8)
        8-bit density-remapped image array in [0, 255].
    """
    amplitude = np.abs(data)
    finite_mask = np.isfinite(amplitude)
    if not np.any(finite_mask) or np.all(amplitude == 0):
        return np.zeros(data.shape, dtype=np.uint8)

    data_mean = float(np.mean(amplitude[finite_mask]))
    c_l = 0.8 * data_mean
    c_h = mmult * c_l
    slope = (255.0 - dmin) / np.log10(c_h / c_l)
    constant = dmin - (slope * np.log10(c_l))
    
    val = slope * np.log10(np.maximum(amplitude, eps)) + constant
    return np.clip(val, 0, 255).astype(np.uint8)


def visualize(sicd_path: str, png_out_path: str):
    """
    Reads a SICD NITF file using sarkit, applies density remapping, and saves an 8-bit PNG.
    """
    print(f"Opening {sicd_path}...")
    with open(sicd_path, "rb") as f:
        reader = sksicd.NitfReader(f)
        print("Reading complex image data...")
        data = reader.read_image()

    print("Applying logarithmic density remap...")
    remapped = density_remap_8bit(data)

    print(f"Saving 8-bit PNG to {png_out_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(png_out_path)), exist_ok=True)
    img = Image.fromarray(remapped)
    img.save(png_out_path)
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/convert2png.py <path_to_sicd.nitf> <path_to_png.png>")
        sys.exit(1)
    visualize(sys.argv[1], sys.argv[2])
