import os
import lxml.etree as ET
import pytest
import sarkit.sicd as sksicd
from diffpfa.IFP import IFAProcessor

cphd_path = "/home/feildaw/data/2023-09-11-10-37-05_UMBRA-05_CPHD.cphd"

@pytest.mark.skipif(not os.path.exists(cphd_path), reason="Test CPHD dataset not found")
def test_sicd_slant_schema_validation(tmp_path):
    proc = IFAProcessor(
        cphd_path=cphd_path,
        output_dir=str(tmp_path),
        image_plane="SLANT",
        device="cuda"
    )
    out_files, _, _, _ = proc.run()
    assert len(out_files) > 0
    nitf_path = out_files[0]
    
    with open(nitf_path, "rb") as f:
        reader = sksicd.NitfReader(f)
        xmltree = reader.metadata.xmltree

    xsd_path = os.path.join(sksicd.schemas.__path__[0], 'SICD_schema_V1.3.0_2021_11_30.xsd')
    schema = ET.XMLSchema(ET.parse(xsd_path))
    assert schema.validate(xmltree), f"Schema errors: {schema.error_log}"

@pytest.mark.skipif(not os.path.exists(cphd_path), reason="Test CPHD dataset not found")
def test_sicd_ground_schema_validation(tmp_path):
    proc = IFAProcessor(
        cphd_path=cphd_path,
        output_dir=str(tmp_path),
        image_plane="GROUND",
        device="cuda"
    )
    out_files, _, _, _ = proc.run()
    assert len(out_files) > 0
    nitf_path = out_files[0]
    
    with open(nitf_path, "rb") as f:
        reader = sksicd.NitfReader(f)
        xmltree = reader.metadata.xmltree

    xsd_path = os.path.join(sksicd.schemas.__path__[0], 'SICD_schema_V1.3.0_2021_11_30.xsd')
    schema = ET.XMLSchema(ET.parse(xsd_path))
    assert schema.validate(xmltree), f"Schema errors: {schema.error_log}"
