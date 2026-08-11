import sys
import os
import lxml.etree as ET
from pathlib import Path
import sarkit.sicd as sksicd

def test_schema():
    # Use the test to generate the XML
    from diffpfa.io.base import CPHDMetadata, SICDImagePayload
    from diffpfa.io.sarkit_sicd import SarkitSICDWriter
    import torch
    import numpy as np

    cphd = CPHDMetadata(
        domain_type="FX",
        sgn=-1,
        global_fx_min=9e9,
        global_fx_max=10e9,
        arp_pos_coa=np.zeros(3),
        arp_vel_coa=np.zeros(3),
        line_spacing=0.1,
        sample_spacing=0.1,
        uIAX=np.array([1, 0, 0]),
        uIAY=np.array([0, 1, 0]),
        iarp_ecf=np.array([0, 0, 0]),
        radar_mode="SPOTLIGHT",
        classification="UNCLASSIFIED",
        collection_start=None,
        image_area=None,
        extended_area=None,
        raw_meta=None
    )

    payload = SICDImagePayload(
        complex_image=torch.zeros((10, 10), dtype=torch.complex64),
        tx_pol="V",
        rcv_pol="V",
        uIAX=np.array([1, 0, 0]),
        uIAY=np.array([0, 1, 0]),
        iarp_ecf=np.array([6378137.0, 0, 0]),
        line_spacing=0.1,
        sample_spacing=0.1,
        first_line=0,
        first_sample=0,
        center_freq=9.5e9,
        bandwidth_u=1.0,
        bandwidth_r=1.0,
        channel_id="Ch1"
    )

    writer = SarkitSICDWriter()
    out = writer.write_sicd("/tmp/test.nitf", payload, cphd)
    print("Generated NITF successfully.")
    
    # Extract XML and validate
    with open("/tmp/test.nitf", "rb") as f:
        meta = sksicd.NitfMetadata.from_file(f)
        
    xml_str = ET.tostring(meta.xmltree, pretty_print=True)
    schema = sksicd.xml.get_schema()
    try:
        schema.assertValid(meta.xmltree)
        print("Schema valid!")
    except Exception as e:
        print("Schema invalid!")
        print(schema.error_log)

if __name__ == '__main__':
    test_schema()
