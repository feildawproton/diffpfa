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
        if cphd_meta.classification is not None:
            sub(coll_info, "Classification", cphd_meta.classification)

        # ImageCreation
        img_creation = sub(root, "ImageCreation")
        sub(img_creation, "Application", "CZTPFA")
        sub(img_creation, "DateTime", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

        # ImageData
        img_data = sub(root, "ImageData")
        sub(img_data, "PixelType", "RE32F_IM32F")
        sub(img_data, "NumRows", str(num_rows))
        sub(img_data, "NumCols", str(num_cols))
        sub(img_data, "FirstRow", "0")
        sub(img_data, "FirstCol", "0")
        full_image = sub(img_data, "FullImage")
        sub(full_image, "NumRows", str(num_rows))
        sub(full_image, "NumCols", str(num_cols))
        scp_pixel = sub(img_data, "SCPPixel")
        sub(scp_pixel, "Row", str(num_rows // 2))
        sub(scp_pixel, "Col", str(num_cols // 2))
        
        # GeoData
        geo_data = sub(root, "GeoData")
        sub(geo_data, "EarthModel", "WGS_84")
        scp = sub(geo_data, "SCP")
        ecf = sub(scp, "ECF")
        sub(ecf, "X", str(payload.iarp_ecf[0]))
        sub(ecf, "Y", str(payload.iarp_ecf[1]))
        sub(ecf, "Z", str(payload.iarp_ecf[2]))
        llh = sub(scp, "LLH")
        ecf_vec = payload.iarp_ecf
        if np.allclose(ecf_vec, 0.0):
            lat_deg, lon_deg, hae = 0.0, 0.0, 0.0
            lat_rad = 0.0
        else:
            lat_rad, lon_rad, hae = cartesian_to_geodetic(ecf_vec)
            lat_deg = np.degrees(lat_rad)
            lon_deg = np.degrees(lon_rad)
        
        sub(llh, "Lat", str(np.clip(lat_deg, -90.0, 90.0)))
        sub(llh, "Lon", str(np.clip(lon_deg, -180.0, 180.0)))
        sub(llh, "HAE", str(hae))

        # Approximate Image Corners for NITF headers
        ic = sub(geo_data, "ImageCorners")
        row_extent = num_rows * payload.line_spacing
        col_extent = num_cols * payload.sample_spacing
        r_deg = row_extent / 6378137.0 * 180.0 / np.pi
        c_deg = col_extent / (6378137.0 * max(0.01, np.cos(lat_rad))) * 180.0 / np.pi
        
        icp1 = sub(ic, "ICP", index="1:FRFC")
        sub(icp1, "Lat", str(np.clip(lat_deg + r_deg/2, -90, 90)))
        sub(icp1, "Lon", str(np.clip(lon_deg - c_deg/2, -180, 180)))
        
        icp2 = sub(ic, "ICP", index="2:FRLC")
        sub(icp2, "Lat", str(np.clip(lat_deg + r_deg/2, -90, 90)))
        sub(icp2, "Lon", str(np.clip(lon_deg + c_deg/2, -180, 180)))
        
        icp3 = sub(ic, "ICP", index="3:LRLC")
        sub(icp3, "Lat", str(np.clip(lat_deg - r_deg/2, -90, 90)))
        sub(icp3, "Lon", str(np.clip(lon_deg + c_deg/2, -180, 180)))
        
        icp4 = sub(ic, "ICP", index="4:LRFC")
        sub(icp4, "Lat", str(np.clip(lat_deg - r_deg/2, -90, 90)))
        sub(icp4, "Lon", str(np.clip(lon_deg - c_deg/2, -180, 180)))

        # Grid
        grid = sub(root, "Grid")
        sub(grid, "ImagePlane", "GROUND")
        sub(grid, "Type", "PLANE")
        time_coa = sub(grid, "TimeCOAPoly", order1="0", order2="0")
        sub(time_coa, "Coef", "0.0", exponent1="0", exponent2="0")
        
        for dir_name, ss, bw in [
            ("Row", payload.line_spacing, payload.bandwidth_u),
            ("Col", payload.sample_spacing, payload.bandwidth_r)
        ]:
            d = sub(grid, dir_name)
            uv = sub(d, "UVectECF")
            if dir_name == "Row":
                sub(uv, "X", "1.0"); sub(uv, "Y", "0.0"); sub(uv, "Z", "0.0")
            else:
                sub(uv, "X", "0.0"); sub(uv, "Y", "1.0"); sub(uv, "Z", "0.0")
            sub(d, "SS", str(ss))
            sub(d, "ImpRespWid", str(1.0 / max(1e-12, bw)))
            sub(d, "Sgn", "-1")
            sub(d, "ImpRespBW", str(bw))
            sub(d, "KCtr", "0.0")
            sub(d, "DeltaK1", str(-bw / 2.0))
            sub(d, "DeltaK2", str(bw / 2.0))

        # Timeline
        timeline = sub(root, "Timeline")
        if cphd_meta.collection_start:
            try:
                dt = datetime.datetime.fromisoformat(str(cphd_meta.collection_start).replace("Z", "+00:00"))
                collect_start = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            except:
                collect_start = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            collect_start = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        sub(timeline, "CollectStart", collect_start)
        sub(timeline, "CollectDuration", "0.0")

        # Position
        pos = sub(root, "Position")
        arp = sub(pos, "ARPPoly")
        x_elem = sub(arp, "X", order1="0")
        sub(x_elem, "Coef", "0.0", exponent1="0")
        y_elem = sub(arp, "Y", order1="0")
        sub(y_elem, "Coef", "0.0", exponent1="0")
        z_elem = sub(arp, "Z", order1="0")
        sub(z_elem, "Coef", "0.0", exponent1="0")

        # RadarCollection
        radar_coll = sub(root, "RadarCollection")
        tx_freq = sub(radar_coll, "TxFrequency")
        sub(tx_freq, "Min", str(cphd_meta.global_fx_min))
        sub(tx_freq, "Max", str(cphd_meta.global_fx_max))
        pol = payload.tx_pol if payload.tx_pol != "UNKNOWN" else "OTHER"
        sub(radar_coll, "TxPolarization", pol)
        tx_seq = sub(radar_coll, "TxSequence", size="1")
        tx_step = sub(tx_seq, "TxStep", index="1")
        sub(tx_step, "TxPolarization", pol)
        rcv_chans = sub(radar_coll, "RcvChannels", size="1")
        chan_params = sub(rcv_chans, "ChanParameters", index="1")
        sub(chan_params, "TxRcvPolarization", pol + ":" + (payload.rcv_pol if payload.rcv_pol != "UNKNOWN" else "OTHER"))

        # ImageFormation
        img_form = sub(root, "ImageFormation")
        rcv_proc = sub(img_form, "RcvChanProc")
        sub(rcv_proc, "NumChanProc", "1")
        sub(rcv_proc, "PRFScaleFactor", "1.0")
        sub(rcv_proc, "ChanIndex", "1")
        pol_proc = sub(img_form, "TxRcvPolarizationProc", pol + ":" + (payload.rcv_pol if payload.rcv_pol != "UNKNOWN" else "OTHER"))
        sub(img_form, "TStartProc", "0.0")
        sub(img_form, "TEndProc", "0.0")
        tx_proc = sub(img_form, "TxFrequencyProc")
        sub(tx_proc, "MinProc", str(cphd_meta.global_fx_min))
        sub(tx_proc, "MaxProc", str(cphd_meta.global_fx_max))
        sub(img_form, "ImageFormAlgo", "OTHER")
        sub(img_form, "STBeamComp", "NO")
        sub(img_form, "ImageBeamComp", "NO")
        sub(img_form, "AzAutofocus", "NO")
        sub(img_form, "RgAutofocus", "NO")
        
        # SCPCOA
        scpcoa = sub(root, "SCPCOA")
        sub(scpcoa, "SCPTime", "0.0")
        sub(scpcoa, "ARPPos")
        sub(scpcoa, "ARPVel")
        
        arp_pos = scpcoa.find("./{*}ARPPos")
        if arp_pos is not None:
            if cphd_meta.arp_pos_coa is not None and len(cphd_meta.arp_pos_coa) == 3:
                sub(arp_pos, "X", str(cphd_meta.arp_pos_coa[0]))
                sub(arp_pos, "Y", str(cphd_meta.arp_pos_coa[1]))
                sub(arp_pos, "Z", str(cphd_meta.arp_pos_coa[2]))
            else:
                sub(arp_pos, "X", "0.0"); sub(arp_pos, "Y", "0.0"); sub(arp_pos, "Z", "0.0")
                
        arp_vel = scpcoa.find("./{*}ARPVel")
        if arp_vel is not None:
            if cphd_meta.arp_vel_coa is not None and len(cphd_meta.arp_vel_coa) == 3:
                sub(arp_vel, "X", str(cphd_meta.arp_vel_coa[0]))
                sub(arp_vel, "Y", str(cphd_meta.arp_vel_coa[1]))
                sub(arp_vel, "Z", str(cphd_meta.arp_vel_coa[2]))
            else:
                sub(arp_vel, "X", "1.0"); sub(arp_vel, "Y", "0.0"); sub(arp_vel, "Z", "0.0")  # Avoid zero-norm
                
        sub(scpcoa, "ARPAcc")
        arp_acc = scpcoa.find("./{*}ARPAcc")
        if arp_acc is not None:
            sub(arp_acc, "X", "0.0"); sub(arp_acc, "Y", "0.0"); sub(arp_acc, "Z", "0.0")
        
        sub(scpcoa, "SideOfTrack", "L")
        sub(scpcoa, "SlantRange", "0.0")
        sub(scpcoa, "GroundRange", "0.0")
        sub(scpcoa, "DopplerConeAng", "90.0")
        sub(scpcoa, "GrazeAng", "45.0")
        sub(scpcoa, "IncidenceAng", "45.0")
        sub(scpcoa, "TwistAng", "0.0")
        sub(scpcoa, "SlopeAng", "0.0")
        sub(scpcoa, "AzimAng", "0.0")
        sub(scpcoa, "LayoverAng", "0.0")

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
