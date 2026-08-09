import sys
import numpy as np
from sarpy.io.complex.converter import open_complex

def compute():
    reader = open_complex("simulation/output/SICD_U_V_V.nitf")
    img = reader.read_chip(None, None)
    img_mag = np.abs(img)
    center_u = img_mag.shape[0] // 2
    slice_arr = img_mag[center_u, :]
    slice_arr /= slice_arr.max()
    
    above_half = np.where(slice_arr > 0.5)[0]
    print(f"Indices above 0.5: {above_half}")
    peak = np.argmax(slice_arr)
    print(f"Peak idx: {peak}")
    print(f"Values around peak: {slice_arr[peak-4:peak+5]}")
        
if __name__ == '__main__':
    compute()
