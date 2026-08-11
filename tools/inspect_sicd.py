import numpy as np
import torch
from diffpfa.io import SICDReader

def get_center_slice(path):
    reader = SICDReader(path)
    img = reader.read_image()
    cu, cr = img.shape[0]//2, img.shape[1]//2
    return img[cu, :]

slice_1 = get_center_slice("simulation/output/SICD_U_V_V_ch_Ch1_VV_Low.nitf")
slice_2 = get_center_slice("simulation/output/SICD_U_V_V_ch_Ch3_VV_High.nitf")
slice_c = get_center_slice("simulation/output/SICD_U_V_V.nitf")

print("Max 1:", np.max(np.abs(slice_1)))
print("Max 2:", np.max(np.abs(slice_2)))
print("Max C:", np.max(np.abs(slice_c)))
print("Sum Max:", np.max(np.abs(slice_1 + slice_2)))

c_idx = slice_1.shape[0]//2
print("Phase 1 center:", np.angle(slice_1[c_idx]))
print("Phase 2 center:", np.angle(slice_2[c_idx]))
print("Phase C center:", np.angle(slice_c[c_idx]))
print("Abs 1 center:", np.abs(slice_1[c_idx]))
print("Abs 2 center:", np.abs(slice_2[c_idx]))
