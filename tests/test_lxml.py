import lxml.etree as ET

ns = "urn:SICD:1.3.0"
nsmap = {None: ns}

def E(tag):
    return f"{{{ns}}}{tag}"

root = ET.Element(E("SICD"), nsmap=nsmap)
coll_info = ET.SubElement(root, E("CollectionInfo"))
ET.SubElement(coll_info, E("CollectorName")).text = "CZTPFA"

print(ET.tostring(root).decode())
