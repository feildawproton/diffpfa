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

    schema_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "schemas"))
    xsd_path = os.path.join(schema_dir, "SICD_schema_V1.3.0_2021_11_30.xsd")
    if not os.path.exists(xsd_path):
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

    schema_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "schemas"))
    xsd_path = os.path.join(schema_dir, "SICD_schema_V1.3.0_2021_11_30.xsd")
    if not os.path.exists(xsd_path):
        xsd_path = os.path.join(sksicd.schemas.__path__[0], 'SICD_schema_V1.3.0_2021_11_30.xsd')
    schema = ET.XMLSchema(ET.parse(xsd_path))
    assert schema.validate(xmltree), f"Schema errors: {schema.error_log}"


@pytest.mark.skipif(not os.path.exists(cphd_path), reason="Test CPHD dataset not found")
def test_sicd_sarkit_consistency_check(tmp_path):
    import subprocess
    import sys
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
        h = sksicd.XmlHelper(reader.metadata.xmltree)
    
    # 1. Verify uniform window ImpRespWid * ImpRespBW ~ 0.886
    for d in ("Row", "Col"):
        k = h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespWid") * h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespBW")
        assert abs(k - 0.886) < 0.02, f"{d}: ImpRespWid*ImpRespBW = {k:.4f}, expected ~0.886"

    # 2. Verify Grid Type is RGAZIM per DIDD §4.15.1
    grid_type = h.load("./{*}Grid/{*}Type")
    assert grid_type == "RGAZIM", f"Expected Grid Type RGAZIM, got {grid_type}"

    # 3. Run sarkit sicdcheck executable
    sicdcheck_bin = os.path.join(os.path.dirname(sys.executable), "sicdcheck")
    if os.path.exists(sicdcheck_bin):
        res = subprocess.run([sicdcheck_bin, nitf_path], capture_output=True, text=True)
        errs = [l.strip() for l in res.stdout.splitlines() if "[Error]" in l]
        assert not errs, f"sicdcheck reported errors:\n" + "\n".join(errs)
