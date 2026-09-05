import numpy as np
import sarkit.cphd as sc
import sarkit.sicd as ss
from pathlib import Path
import torch

from diffpfa.IFA.kspace import compute_kspace
from diffpfa.constants import SPEED_OF_LIGHT

stem = "2025-10-26-05-00-15_UMBRA-08"
cphd_path = Path(f"/home/feildaw/data/{stem}_CPHD.cphd")
umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")

with open(cphd_path, "rb") as f:
    r = sc.Reader(f)
    ch_elem = r.metadata.xmltree.find(".//{*}Data/{*}Channel/{*}Identifier")
    ch_id = ch_elem.text
    pvp_raw = r.read_pvps(ch_id)
    pvp = {}
    for name in pvp_raw.dtype.names:
        arr = pvp_raw[name]
        if hasattr(arr, "dtype") and arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        pvp[name] = arr
    sig = r.read_signal(ch_id)
    xh = sc.XmlHelper(r.metadata.xmltree)
    uIAX = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"))
    uIAY = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"))
    fx_min = float(xh.load("./{*}Global/{*}FxBand/{*}FxMin"))
    fx_max = float(xh.load("./{*}Global/{*}FxBand/{*}FxMax"))

with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r_u:
    xh_u = ss.XmlHelper(r_u.metadata.xmltree)
    u_row_bw = xh_u.load("./{*}Grid/{*}Row/{*}ImpRespBW")
    u_col_bw = xh_u.load("./{*}Grid/{*}Col/{*}ImpRespBW")
    u_row_u = np.array(xh_u.load("./{*}Grid/{*}Row/{*}UVectECF"))
    u_col_u = np.array(xh_u.load("./{*}Grid/{*}Col/{*}UVectECF"))
    u_delta_k1_r = xh_u.load("./{*}Grid/{*}Row/{*}DeltaK1")
    u_delta_k2_r = xh_u.load("./{*}Grid/{*}Row/{*}DeltaK2")
    u_delta_k1_c = xh_u.load("./{*}Grid/{*}Col/{*}DeltaK1")
    u_delta_k2_c = xh_u.load("./{*}Grid/{*}Col/{*}DeltaK2")
    u_k_ctr_r = xh_u.load("./{*}Grid/{*}Row/{*}KCtr")
    u_k_ctr_c = xh_u.load("./{*}Grid/{*}Col/{*}KCtr")

print("CPHD Global FxBand:", fx_min, "to", fx_max, "BW_Hz:", fx_max - fx_min)
cpm_rf = 2.0 * (fx_max - fx_min) / SPEED_OF_LIGHT
print(f"2 * BW_Hz / c = {cpm_rf:.6f} cycles/m")

print("\nUmbra SICD:")
print(f"  Row BW: {u_row_bw:.6f}, KCtr: {u_k_ctr_r}, DeltaK: [{u_delta_k1_r}, {u_delta_k2_r}]")
print(f"  Col BW: {u_col_bw:.6f}, KCtr: {u_k_ctr_c}, DeltaK: [{u_delta_k1_c}, {u_delta_k2_c}]")

# Now calculate K-space using diffpfa's slant vectors vs CPHD vectors
from diffpfa.IFP import IFAProcessor
p = IFAProcessor(str(cphd_path), "/tmp")
with open(cphd_path, "rb") as f:
    meta = p._read_metadata(sc.Reader(f))

srp = meta.srp_ecf
arp = meta.arp_pos_coa
arp_v = meta.arp_vel_coa
p_vec = srp - arp
u_row = p_vec / np.linalg.norm(p_vec)
u_v = arp_v / np.linalg.norm(arp_v)
u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
if meta.side_of_track == "L":
    u_col = -u_col

print("\nComputed slant vectors:")
print("  u_row (LOS):", u_row)
print("  u_col:", u_col)
print("  Umbra u_row:", u_row_u)
print("  Umbra u_col:", u_col_u)

# Compute K-space with u_col (uIAX) and u_row (uIAY)
Ku, Kr = compute_kspace(pvp, u_col, u_row, sig.shape[1], domain_type="FX", device="cpu")
print("\nSlant K-space computed:")
print(f"  Kr min: {Kr.min():.6f}, max: {Kr.max():.6f}, span: {Kr.max() - Kr.min():.6f}")
print(f"  Ku min: {Ku.min():.6f}, max: {Ku.max():.6f}, span: {Ku.max() - Ku.min():.6f}")

# What if we compute K-space along the center pulse only vs all pulses?
mid = sig.shape[0] // 2
print(f"  Center pulse Kr span: {Kr[mid].max() - Kr[mid].min():.6f}")
print(f"  Total Kr span across aperture: {Kr.max() - Kr.min():.6f}")
