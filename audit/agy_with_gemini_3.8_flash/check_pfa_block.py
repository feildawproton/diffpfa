import sarkit.sicd as ss
from pathlib import Path
import numpy as np

stem = "2025-10-26-05-00-15_UMBRA-08"
umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")
diffpfa_sicd = Path(f"/home/feildaw/diffpfa/workspace/output/{stem}_SICDU_V_V.nitf")

with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
    xh = ss.XmlHelper(r.metadata.xmltree)
    print("=== UMBRA GOLD PFA BLOCK ===")
    pfa_elem = r.metadata.xmltree.find(".//{*}PFA")
    print("Has PFA block?", pfa_elem is not None)
    if pfa_elem is not None:
        fpn = [float(pfa_elem.find(f"./{{*}}FPN/{{*}}{c}").text) for c in ["X", "Y", "Z"]]
        ipn = [float(pfa_elem.find(f"./{{*}}IPN/{{*}}{c}").text) for c in ["X", "Y", "Z"]]
        t_ref = float(pfa_elem.find("./{*}PolarAngRefTime").text)
        print("  FPN:", fpn)
        print("  IPN:", ipn)
        print("  PolarAngRefTime:", t_ref)
        krg1 = float(pfa_elem.find("./{*}Krg1").text)
        krg2 = float(pfa_elem.find("./{*}Krg2").text)
        kaz1 = float(pfa_elem.find("./{*}Kaz1").text)
        kaz2 = float(pfa_elem.find("./{*}Kaz2").text)
        print(f"  Krg: [{krg1}, {krg2}], Kaz: [{kaz1}, {kaz2}]")

with open(diffpfa_sicd, "rb") as f, ss.NitfReader(f) as r:
    print("\n=== DIFFPFA PFA BLOCK ===")
    pfa_elem = r.metadata.xmltree.find(".//{*}PFA")
    print("Has PFA block?", pfa_elem is not None)
    if pfa_elem is not None:
        fpn = [float(pfa_elem.find(f"./{{*}}FPN/{{*}}{c}").text) for c in ["X", "Y", "Z"]]
        ipn = [float(pfa_elem.find(f"./{{*}}IPN/{{*}}{c}").text) for c in ["X", "Y", "Z"]]
        t_ref = float(pfa_elem.find("./{*}PolarAngRefTime").text)
        print("  FPN:", fpn)
        print("  IPN:", ipn)
        print("  PolarAngRefTime:", t_ref)
        krg1 = float(pfa_elem.find("./{*}Krg1").text)
        krg2 = float(pfa_elem.find("./{*}Krg2").text)
        kaz1 = float(pfa_elem.find("./{*}Kaz1").text)
        kaz2 = float(pfa_elem.find("./{*}Kaz2").text)
        print(f"  Krg: [{krg1}, {krg2}], Kaz: [{kaz1}, {kaz2}]")
