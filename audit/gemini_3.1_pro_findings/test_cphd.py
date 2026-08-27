import numpy as np
import sarkit.cphd as skcphd
import glob

cphd_files = glob.glob("/home/feildaw/diffpfa/tests/data/*.cphd") + glob.glob("/home/feildaw/diffpfa/*.cphd") + glob.glob("/home/feildaw/diffpfa/simulation/*.cphd")
if not cphd_files:
    print("No CPHD files found to test.")
else:
    for f in cphd_files:
        print("File:", f)
        try:
            with open(f, "rb") as fp:
                reader = skcphd.Reader(fp)
                ch_id = reader.metadata.xmltree.find(".//{*}Channel/{*}Identifier").text
                pvp = reader.read_pvps(ch_id)
                print("SC0:", pvp["SC0"][0:2])
                print("SCSS:", pvp["SCSS"][0:2])
                print("TxFMRate:", pvp["TxFMRate"][0:2] if "TxFMRate" in pvp.dtype.names else "N/A")
                print("---")
        except Exception as e:
            print(e)
