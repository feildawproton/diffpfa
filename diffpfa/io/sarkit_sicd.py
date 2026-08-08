import datetime
import os
import lxml.etree as ET
import numpy as np
import torch
import sarkit.sicd as sksicd
from sarkit.wgs84 import cartesian_to_geodetic

from diffpfa.io.base import BaseSICDWriter, CPHDMetadata, SICDImagePayload

class SarkitSICDWriter(BaseSICDWriter):
    """sarkit-backed implementation of SICDWriter."""

    def write_sicd(
        self,
        output_path: str,
        payload: SICDImagePayload,
        cphd_meta: CPHDMetadata
    ) -> str:
        if isinstance(payload.complex_image, torch.Tensor):
            img_arr = payload.complex_image.detach().cpu().numpy()
        else:
            img_arr = np.asarray(payload.complex_image)

        num_rows, num_cols = img_arr.shape
        img_arr = img_arr.astype(np.complex64)

        # 1. Build SICD XML Tree
        ns = "urn:SICD:1.3.0"
        nsmap = {None: ns}
        
        def sub(parent, tag, text=None, **attrib):
            elem = ET.SubElement(parent, f"{{{ns}}}{tag}", **attrib)
            if text is not None:
                elem.text = text
            return elem

        root = ET.Element(f"{{{ns}}}SICD", nsmap=nsmap)

        # CollectionInfo
        coll_info = sub(root, "CollectionInfo")
        sub(coll_info, "CollectorName", "CZTPFA")
        sub(coll_info, "CoreName", "CZTPFA_Output")
        radar_mode = sub(coll_info, "RadarMode")
        sub(radar_mode, "ModeType", cphd_meta.radar_mode or "SPOTLIGHT")

        # ImageCreation
        img_creation = sub(root, "ImageCreation")
        sub(img_creation, "Application", "CZTPFA")
        sub(img_creation, "DateTime", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

        # ImageData
        img_data = sub(root, "ImageData")
        sub(img_data, "PixelType", "RE32F_IM32F")
        sub(img_data, "NumRows", str(num_rows))
        sub(img_data, "NumCols", str(num_cols))
        sub(img_data, "FirstRow", "0")
        sub(img_data, "FirstCol", "0")
        sub(img_data, "FullImage", "0")
        
        # GeoData
        geo_data = sub(root, "GeoData")
        scp = sub(geo_data, "SCP")
        ecf = sub(scp, "ECF")
        sub(ecf, "X", str(payload.iarp_ecf[0]))
        sub(ecf, "Y", str(payload.iarp_ecf[1]))
        sub(ecf, "Z", str(payload.iarp_ecf[2]))
        llh = sub(scp, "LLH")
        lat_rad, lon_rad, hae = cartesian_to_geodetic(payload.iarp_ecf)
        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        sub(llh, "Lat", str(lat_deg))
        sub(llh, "Lon", str(lon_deg))
        sub(llh, "HAE", str(hae))

        # Approximate Image Corners for NITF headers
        ic = sub(geo_data, "ImageCorners")
        row_extent = num_rows * payload.line_spacing
        col_extent = num_cols * payload.sample_spacing
        r_deg = row_extent / 6378137.0 * 180.0 / np.pi
        c_deg = col_extent / (6378137.0 * max(0.01, np.cos(lat_rad))) * 180.0 / np.pi
        
        icp1 = sub(ic, "ICP", index="1:FRFC")
        sub(icp1, "Lat", str(lat_deg + r_deg/2))
        sub(icp1, "Lon", str(lon_deg - c_deg/2))
        
        icp2 = sub(ic, "ICP", index="2:FRLC")
        sub(icp2, "Lat", str(lat_deg + r_deg/2))
        sub(icp2, "Lon", str(lon_deg + c_deg/2))
        
        icp3 = sub(ic, "ICP", index="3:LRLC")
        sub(icp3, "Lat", str(lat_deg - r_deg/2))
        sub(icp3, "Lon", str(lon_deg + c_deg/2))
        
        icp4 = sub(ic, "ICP", index="4:LRFC")
        sub(icp4, "Lat", str(lat_deg - r_deg/2))
        sub(icp4, "Lon", str(lon_deg - c_deg/2))

        # Grid
        grid = sub(root, "Grid")
        sub(grid, "ImagePlane", "GROUND")
        sub(grid, "Type", "PLANE")
        
        for dir_name, ss, bw in [
            ("Row", payload.line_spacing, payload.bandwidth_u),
            ("Col", payload.sample_spacing, payload.bandwidth_r)
        ]:
            d = sub(grid, dir_name)
            sub(d, "SS", str(ss))
            sub(d, "ImpRespBW", str(bw))
            sub(d, "KCtr", "0.0")
            sub(d, "DeltaK1", str(-bw / 2.0))
            sub(d, "DeltaK2", str(bw / 2.0))

        # Timeline
        timeline = sub(root, "Timeline")
        collect_start = cphd_meta.collection_start if cphd_meta.collection_start else datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sub(timeline, "CollectStart", collect_start)

        # RadarCollection
        radar_coll = sub(root, "RadarCollection")
        sub(radar_coll, "TxPolarization", payload.tx_pol if payload.tx_pol != "UNKNOWN" else "OTHER")

        # ImageFormation
        img_form = sub(root, "ImageFormation")
        sub(img_form, "ImageFormAlgo", "OTHER")
        tx_proc = sub(img_form, "TxFrequencyProc")
        sub(tx_proc, "MinProc", str(cphd_meta.global_fx_min))
        sub(tx_proc, "MaxProc", str(cphd_meta.global_fx_max))

        # Position
        pos = sub(root, "Position")
        arp = sub(pos, "ARPPoly")
        sub(arp, "X", coefA="0.0")
        sub(arp, "Y", coefA="0.0")
        sub(arp, "Z", coefA="0.0")

        xmltree = ET.ElementTree(root)

        # 2. Package NITF Metadata
        sec = sksicd.NitfSecurityFields(clas="U")
        sicd_meta = sksicd.NitfMetadata(
            xmltree=xmltree,
            file_header_part=sksicd.NitfFileHeaderPart(ostaid="CZTPFA", ftitle="SICD Output", security=sec),
            im_subheader_part=sksicd.NitfImSubheaderPart(isorce="CZTPFA", security=sec),
            de_subheader_part=sksicd.NitfDeSubheaderPart(security=sec),
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

        # 3. Write NITF SICD
        with open(output_path, "wb") as f, sksicd.NitfWriter(f, sicd_meta) as writer:
            writer.write_image(img_arr)

        return output_path
