import sys
import matplotlib.pyplot as plt
import numpy as np
import sarpy.io.complex as sarpy_complex

def visualize(filename, mode):
    print(f"Opening {filename}...")
    reader = sarpy_complex.open(filename)
    
    print("Reading image data...")
    # Read the complex data from the first image segment
    # PFA algorithm outputs (Cross-Range, Range)
    # Transpose so Cross-Range is horizontal (X-axis) and Range is vertical (Y-axis)
    data = reader[:]
    
    from sarpy.visualization.remap import density
    
    # Use Sarpy's density remapper for optimal SAR visualization (returns 8-bit image)
    if mode=="density":
        mapped_image = density(data)
    else:
        mag = 20*np.log10(np.abs(data))
        maxm = np.max(mag)
        minm = np.min(mag)
        mag = (mag - minm) / (maxm - minm)
        mag = mag * 255
        mapped_image = mag.astype(np.int8)

    print("Plotting image...")
    plt.figure(figsize=(10, 10))
    plt.imshow(mapped_image, cmap='gray', origin='lower')
    # plt.colorbar(label='8-bit Density')
    plt.title(f'SICD Image: {filename}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_sicd.py <path_to_sicd.nitf> <mode>")
        sys.exit(1)
    visualize(sys.argv[1], sys.argv[2])
