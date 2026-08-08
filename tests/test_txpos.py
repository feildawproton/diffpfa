from diffpfa.io.sarpy_cphd import SarpyCPHDReader
from diffpfa.io.sarkit_cphd import SarkitCPHDReader
import numpy as np

cphd_path = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"
r_sarpy = SarpyCPHDReader(cphd_path)
ch_sarpy = r_sarpy.read_channel(r_sarpy.get_channel_names()[0])

r_sarkit = SarkitCPHDReader(cphd_path)
ch_sarkit = r_sarkit.read_channel(r_sarkit.get_channel_names()[0])

tx_sarpy = ch_sarpy.pvp["TxPos"]
tx_sarkit = ch_sarkit.pvp["TxPos"]

print("TxPos match:", np.allclose(tx_sarpy, tx_sarkit))
print("TxPos sarpy mean:", np.mean(tx_sarpy))
print("TxPos sarkit mean:", np.mean(tx_sarkit))
