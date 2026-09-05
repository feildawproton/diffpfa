import numpy as np
import sarkit.sicd as ss
from pathlib import Path

stem = "2025-10-26-05-00-15_UMBRA-08"
umbra_path = Path(f"/home/feildaw/data/{stem}_SICD.nitf")
diffpfa_path = Path(f"/home/feildaw/diffpfa/workspace/output/{stem}_SICDU_V_V.nitf")

with open(umbra_path, "rb") as f, ss.NitfReader(f) as r_u:
    xml_u = r_u.metadata.xmltree
    xh_u = ss.XmlHelper(xml_u)
    u_scp_r = xh_u.load("./{*}ImageData/{*}SCPPixel/{*}Row")
    u_scp_c = xh_u.load("./{*}ImageData/{*}SCPPixel/{*}Col")
    u_ss_r = xh_u.load("./{*}Grid/{*}Row/{*}SS")
    u_ss_c = xh_u.load("./{*}Grid/{*}Col/{*}SS")
    print(f"Umbra SCP: ({u_scp_r}, {u_scp_c}), SS: ({u_ss_r:.4f}, {u_ss_c:.4f})")
    
    # Read a 512x512 patch around SCP
    patch_u, _ = r_u.read_sub_image(
        start_row=u_scp_r - 256,
        stop_row=u_scp_r + 256,
        start_col=u_scp_c - 256,
        stop_col=u_scp_c + 256
    )

with open(diffpfa_path, "rb") as f, ss.NitfReader(f) as r_d:
    xml_d = r_d.metadata.xmltree
    xh_d = ss.XmlHelper(xml_d)
    d_scp_r = xh_d.load("./{*}ImageData/{*}SCPPixel/{*}Row")
    d_scp_c = xh_d.load("./{*}ImageData/{*}SCPPixel/{*}Col")
    d_ss_r = xh_d.load("./{*}Grid/{*}Row/{*}SS")
    d_ss_c = xh_d.load("./{*}Grid/{*}Col/{*}SS")
    print(f"DiffPFA SCP: ({d_scp_r}, {d_scp_c}), SS: ({d_ss_r:.4f}, {d_ss_c:.4f})")
    
    patch_d, _ = r_d.read_sub_image(
        start_row=d_scp_r - 256,
        stop_row=d_scp_r + 256,
        start_col=d_scp_c - 256,
        stop_col=d_scp_c + 256
    )

print("Patch Umbra shape:", patch_u.shape, "dtype:", patch_u.dtype, "mean mag:", np.mean(np.abs(patch_u)))
print("Patch DiffPFA shape:", patch_d.shape, "dtype:", patch_d.dtype, "mean mag:", np.mean(np.abs(patch_d)))
