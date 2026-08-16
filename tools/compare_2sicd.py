import sys
import matplotlib.pyplot as plt
import numpy as np
import sarpy.io.complex as sarpy_complex
from sarpy.visualization.remap import density # mapped_image = density(data)

def visualize(filename1, filename2, mode: str):
    
    reader1 = sarpy_complex.open(filename1)
    data1 = reader1[:]
    reader2 = sarpy_complex.open(filename2)
    data2 = reader2[:]

    if mode=="density":
        mapd1 = density(data1)
        mapd2 = density(data2)
    else:
        mag1 = 20*np.log10(np.abs(data1))
        mag2 = 20*np.log10(np.abs(data2))
           
        maxm = max(np.max(mag1), np.max(mag2))
        minm = min(np.min(mag1), np.min(mag2))

        mapd1 = (mag1 - minm) / (maxm - minm)
        mapd2 = (mag2 - minm) / (maxm - minm)

    print("Plotting image...")
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(mapd1, cmap="viridis", origin="lower")
    ax[1].imshow(mapd2, cmap="viridis", origin="lower")
    plt.title(f'SICD Image: {filename1} and {filename2}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python visualize_sicd.py <path_to_sicd.nitf> <path_to_sicd.nitf> <mode>")
        sys.exit(1)
    visualize(sys.argv[1], sys.argv[2], sys.argv[3])
