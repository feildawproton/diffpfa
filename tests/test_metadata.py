from diffpfa.algo import PFAConfig, PFAEngine
from diffpfa.io.sarpy_cphd import SarpyCPHDReader
from diffpfa.io.sarkit_cphd import SarkitCPHDReader

cphd_path = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"

r_sarpy = SarpyCPHDReader(cphd_path)
m_sarpy = r_sarpy.get_metadata()

r_sarkit = SarkitCPHDReader(cphd_path)
m_sarkit = r_sarkit.get_metadata()

print("--- GLOBAL METADATA ---")
print(f"SARPY  FxMin: {m_sarpy.global_fx_min}, FxMax: {m_sarpy.global_fx_max}")
print(f"SARKIT FxMin: {m_sarkit.global_fx_min}, FxMax: {m_sarkit.global_fx_max}")

channels = r_sarpy.get_channel_names()
for ch in channels:
    print(f"\n--- CHANNEL: {ch} ---")
    ch_sarpy = r_sarpy.read_channel(ch)
    ch_sarkit = r_sarkit.read_channel(ch)
    
    print(f"SARPY  FxC: {ch_sarpy.fxc}, FxBW: {ch_sarpy.fxbw}")
    print(f"SARKIT FxC: {ch_sarkit.fxc}, FxBW: {ch_sarkit.fxbw}")
    
    sig_shape_sarpy = ch_sarpy.signal.shape
    sig_shape_sarkit = ch_sarkit.signal.shape
    print(f"SARPY  Signal Shape: {sig_shape_sarpy}")
    print(f"SARKIT Signal Shape: {sig_shape_sarkit}")
    
    # compare PVP sizes just to be sure
    print(f"SARPY  TxPos shape: {ch_sarpy.pvp.get('TxPos', []).shape}")
    print(f"SARKIT TxPos shape: {ch_sarkit.pvp.get('TxPos', []).shape}")

