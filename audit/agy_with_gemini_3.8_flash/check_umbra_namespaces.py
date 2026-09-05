import sarkit.sicd as ss
from pathlib import Path

DATA_DIR = Path("/home/feildaw/data")

print(f"{'Dataset':<35} | {'SICD Root Tag / Namespace'}")
print("-" * 80)
for p in sorted(DATA_DIR.glob("*_SICD.nitf")):
    with open(p, "rb") as f, ss.NitfReader(f) as r:
        tag = r.metadata.xmltree.getroot().tag
        print(f"{p.stem:<35} | {tag}")
