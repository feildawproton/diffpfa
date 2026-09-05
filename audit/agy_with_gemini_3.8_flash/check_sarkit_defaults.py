import sarkit.sicd as ss
import inspect

print("sarkit version:", getattr(ss, "__version__", "unknown"))

# Inspect sarkit.sicd modules and classes
print("\nAttributes in sarkit.sicd:")
for name in dir(ss):
    if not name.startswith("_"):
        val = getattr(ss, name)
        if isinstance(val, (str, int, float)):
            print(f"  {name} = {val}")

# Check schema / xml helper in sarkit.sicd
if hasattr(ss, "SCHEMA_VERSION"):
    print("ss.SCHEMA_VERSION:", ss.SCHEMA_VERSION)

# Check XmlHelper or NitfMetadata
import sarkit.sicd._xml as sxml
print("\nsarkit.sicd._xml contents:")
for name in dir(sxml):
    if not name.startswith("_"):
        print(f"  {name}: {getattr(sxml, name)}")

# Let's check which schema versions sarkit supports and what default urn it uses
import lxml.etree as ET
root_test = ET.Element("{urn:SICD:1.3.0}SICD")
try:
    print("\nValidating 1.3.0 element with sxml.validate_xml:")
    # see what validate functions exist
    for fn in ["validate_xml", "schema_validate", "load_schema"]:
        if hasattr(sxml, fn):
            print(f"  Found function {fn}")
except Exception as e:
    print("Error:", e)
