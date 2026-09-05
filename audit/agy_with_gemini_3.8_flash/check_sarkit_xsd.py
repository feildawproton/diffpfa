import sarkit.sicd._constants as sconst
import sarkit.sicd._xml as sxml

print("sarkit.sicd._constants:")
for name in dir(sconst):
    if not name.startswith("_"):
        print(f"  {name} = {getattr(sconst, name)}")

print("\nsarkit.sicd._xml.XsdHelper:")
for name in dir(sxml.XsdHelper):
    if not name.startswith("_"):
        print(f"  {name}")

# Check what default schema sarkit validates against
import inspect
print("\nXsdHelper.__init__ signature:", inspect.signature(sxml.XsdHelper.__init__))

xh = sxml.XsdHelper()
print("xh schema:", getattr(xh, "_schema", None))
print("xh namespaces / schemas:")
for k, v in getattr(xh, "__dict__", {}).items():
    print(f"  {k}: {v}")

# Check NitfWriter signature
import sarkit.sicd as ss
print("\nNitfWriter signature:", inspect.signature(ss.NitfWriter.__init__))
