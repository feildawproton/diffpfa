import sys
import numpy as np
import sarpy.io.complex as sarpy_complex
from PIL import Image

def visualize(filename, outpath):
    print(f"Opening {filename}...")
    reader = sarpy_complex.open(filename)
    
    print("Reading image data...")
    # Read the complex data from the first image segment
    # PFA algorithm outputs (Cross-Range, Range)
    # Transpose so Cross-Range is horizontal (X-axis) and Range is vertical (Y-axis)
    data = reader[:]
    
    from sarpy.visualization.remap import density
    
    # Use Sarpy's density remapper for optimal SAR visualization (returns 8-bit image)
    mapped_image = density(data)
    
    img = Image.fromarray(mapped_image)
    img.save(outpath)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_sicd.py <path_to_sicd.nitf> <path_to_png.png")
        sys.exit(1)
    visualize(sys.argv[1], sys.argv[2])
