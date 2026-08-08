import sarkit
print(dir(sarkit))
try:
    from sarkit._cphd import Cphd
    print("Found _cphd.Cphd")
except:
    pass
