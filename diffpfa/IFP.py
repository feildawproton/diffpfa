import os
import torch
import numpy as np
import lxml.etree as ET
import datetime
from typing import List, Dict, Tuple, Optional
import sarkit.cphd as skcphd
import sarkit.sicd as sksicd
from pathlib import Path

from diffpfa.types import CPHDMetadata, ImageAreaBounds
from diffpfa.IFA.pfa import pfa_per_polar

def _cartesian_to_geodetic(x: np.ndarray) -> np.ndarray:
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b**2) / (a**2)
    ep2 = (a**2 - b**2) / (b**2)
    p = np.sqrt(x[0]**2 + x[1]**2)
    th = np.arctan2(a * x[2], b * p)
    lon = np.arctan2(x[1], x[0])
    lat = np.arctan2((x[2] + ep2 * b * np.sin(th)**3), (p - e2 * a * np.cos(th)**3))
    n = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - n
    return np.array([lat, lon, alt])

class IFAProcessor:
    def __init__(self, cphd_path: str, output_dir: str, image_area_mode: str = "ImageArea", custom_pixel_spacing: Optional[Tuple[float, float]] = None, device: str = "cuda"):
        self.cphd_path = cphd_path
        self.output_dir = output_dir
        self.image_area_mode = image_area_mode
        self.custom_pixel_spacing = custom_pixel_spacing
        self.device = device
        
    def _read_metadata(self, reader) -> CPHDMetadata:
        xmltree = reader.metadata.xmltree
        xml_helper = skcphd.XmlHelper(xmltree)

        domain_type = xml_helper.load("./{*}Global/{*}DomainType") or "FX"
        sgn = xml_helper.load("./{*}Global/{*}SGN") or -1
        fx_min = xml_helper.load("./{*}Global/{*}FxBand/{*}FxMin")
        fx_max = xml_helper.load("./{*}Global/{*}FxBand/{*}FxMax")
        iarp_ecf = xml_helper.load("./{*}SceneCoordinates/{*}IARP/{*}ECF")
        uIAX = xml_helper.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX")
        uIAY = xml_helper.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY")

        img_area = None
        ia_x1y1 = xml_helper.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1")
        ia_x2y2 = xml_helper.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
        if ia_x1y1 is not None and ia_x2y2 is not None:
            img_area = ImageAreaBounds(x1=ia_x1y1[0], y1=ia_x1y1[1], x2=ia_x2y2[0], y2=ia_x2y2[1], polygon=None)

        ea_x1y1 = xml_helper.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X1Y1")
        ea_x2y2 = xml_helper.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X2Y2")
        ext_area = None
        if ea_x1y1 is not None and ea_x2y2 is not None:
            ext_area = ImageAreaBounds(x1=ea_x1y1[0], y1=ea_x1y1[1], x2=ea_x2y2[0], y2=ea_x2y2[1], polygon=None)

        coll_start = xml_helper.load("./{*}Global/{*}Timeline/{*}CollectionStart")
        srp_ecf = xml_helper.load("./{*}ReferenceGeometry/{*}SRP/{*}ECF")
        arp_pos = xml_helper.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPPos")
        arp_vel = xml_helper.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPVel")

        return CPHDMetadata(
            domain_type=str(domain_type),
            sgn=int(sgn),
            global_fx_min=float(fx_min),
            global_fx_max=float(fx_max),
            iarp_ecf=iarp_ecf,
            uIAX=uIAX,
            uIAY=uIAY,
            image_area=img_area,
            extended_area=ext_area,
            collection_start=str(coll_start) if coll_start else None,
            radar_mode="UNKNOWN",
            classification="UNCLASSIFIED",
            srp_ecf=srp_ecf,
            arp_pos_coa=arp_pos,
            arp_vel_coa=arp_vel,
            line_spacing=None,
            sample_spacing=None,
            raw_meta=xmltree,
        )

    def _determine_spatial_bounds(self, cphd_meta):
        mode = self.image_area_mode
        if mode == "ImageArea" and cphd_meta.image_area is not None:
            ia = cphd_meta.image_area
            u_min, u_max = min(ia.x1, ia.x2), max(ia.x1, ia.x2)
            r_min, r_max = min(ia.y1, ia.y2), max(ia.y1, ia.y2)
        elif mode == "ExtendedArea" and cphd_meta.extended_area is not None:
            ea = cphd_meta.extended_area
            u_min, u_max = min(ea.x1, ea.x2), max(ea.x1, ea.x2)
            r_min, r_max = min(ea.y1, ea.y2), max(ea.y1, ea.y2)
        else:
            u_min, u_max = -100.0, 100.0
            r_min, r_max = -100.0, 100.0
        return u_min, u_max, r_min, r_max
        
    def _write_sicd(self, output_path: str, img_cpu: torch.Tensor, cphd_meta, tx_pol, rcv_pol, bw_u, bw_r, N_u, N_r, u_min, r_min, du, dr):
        img_arr = img_cpu.numpy().astype(np.complex64)
        num_rows, num_cols = img_arr.shape
        
        root = ET.Element("{urn:SICD:1.3.0}SICD")
        def sub(parent, tag, text=None, **attrib):
            child = ET.SubElement(parent, "{urn:SICD:1.3.0}" + tag, attrib)
            if text is not None: child.text = str(text)
            return child

        col_info = sub(root, "CollectionInfo")
        sub(col_info, "CollectorName", "CZTPFA")
        sub(col_info, "IlluminatorName", "CZTPFA")
        sub(col_info, "CoreName", "PFA_OUTPUT")
        sub(col_info, "CollectType", "MONOSTATIC")
        rm = sub(col_info, "RadarMode")
        sub(rm, "ModeType", "SPOTLIGHT")
        sub(col_info, "Classification", "UNCLASSIFIED")

        img_creation = sub(root, "ImageCreation")
        sub(img_creation, "Application", "diffpfa")
        sub(img_creation, "DateTime", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        sub(img_creation, "Site", "PFA_ENGINE")
        sub(img_creation, "Profile", "PFA")

        img_data = sub(root, "ImageData")
        sub(img_data, "PixelType", "RE32F_IM32F")
        # sub(img_data, "AmpTable")
        sub(img_data, "NumRows", str(num_rows))
        sub(img_data, "NumCols", str(num_cols))
        sub(img_data, "FirstRow", "0")
        sub(img_data, "FirstCol", "0")
        fi = sub(img_data, "FullImage")
        sub(fi, "NumRows", str(num_rows))
        sub(fi, "NumCols", str(num_cols))
        sp = sub(img_data, "SCPPixel")
        sub(sp, "Row", str(num_rows // 2))
        sub(sp, "Col", str(num_cols // 2))

        geo_data = sub(root, "GeoData")
        scp = sub(geo_data, "EarthModel", "WGS_84")
        scp = sub(geo_data, "SCP")
        ecf = sub(scp, "ECF")
        sub(ecf, "X", str(cphd_meta.iarp_ecf[0]))
        sub(ecf, "Y", str(cphd_meta.iarp_ecf[1]))
        sub(ecf, "Z", str(cphd_meta.iarp_ecf[2]))
        
        llh = sub(scp, "LLH")
        lat_rad, lon_rad, hae = _cartesian_to_geodetic(cphd_meta.iarp_ecf)
        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        sub(llh, "Lat", str(np.clip(lat_deg, -90.0, 90.0)))
        sub(llh, "Lon", str(np.clip(lon_deg, -180.0, 180.0)))
        sub(llh, "HAE", str(hae))

        # Approximate Image Corners for NITF headers
        ic = sub(geo_data, "ImageCorners")
        row_extent = num_rows * du
        col_extent = num_cols * dr
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


        grid = sub(root, "Grid")
        sub(grid, "ImagePlane", "GROUND")
        sub(grid, "Type", "PLANE")
        time_coa = sub(grid, "TimeCOAPoly", order1="0", order2="0")
        sub(time_coa, "Coef", "0.0", exponent1="0", exponent2="0")
        
        for dir_name, ss, bw in [("Row", du, bw_u), ("Col", dr, bw_r)]:
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

        timeline = sub(root, "Timeline")
        collect_start = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        sub(timeline, "CollectStart", collect_start)
        sub(timeline, "CollectDuration", "0.0")

        pos = sub(root, "Position")
        arp = sub(pos, "ARPPoly")
        x_elem = sub(arp, "X", order1="0"); sub(x_elem, "Coef", "0.0", exponent1="0")
        y_elem = sub(arp, "Y", order1="0"); sub(y_elem, "Coef", "0.0", exponent1="0")
        z_elem = sub(arp, "Z", order1="0"); sub(z_elem, "Coef", "0.0", exponent1="0")

        radar_coll = sub(root, "RadarCollection")
        tx_freq = sub(radar_coll, "TxFrequency")
        sub(tx_freq, "Min", str(cphd_meta.global_fx_min))
        sub(tx_freq, "Max", str(cphd_meta.global_fx_max))
        sub(radar_coll, "TxPolarization", tx_pol)
        tx_seq = sub(radar_coll, "TxSequence", size="1")
        tx_step = sub(tx_seq, "TxStep", index="1")
        sub(tx_step, "TxPolarization", tx_pol)
        rcv_chans = sub(radar_coll, "RcvChannels", size="1")
        chan_params = sub(rcv_chans, "ChanParameters", index="1")
        sub(chan_params, "TxRcvPolarization", f"{tx_pol}:{rcv_pol}")

        img_form = sub(root, "ImageFormation")
        rcv_proc = sub(img_form, "RcvChanProc")
        sub(rcv_proc, "NumChanProc", "1")
        sub(rcv_proc, "PRFScaleFactor", "1.0")
        sub(rcv_proc, "ChanIndex", "1")
        sub(img_form, "TxRcvPolarizationProc", f"{tx_pol}:{rcv_pol}")
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
        
        scpcoa = sub(root, "SCPCOA")
        sub(scpcoa, "SCPTime", "0.0")
        arp_pos = sub(scpcoa, "ARPPos")
        if cphd_meta.arp_pos_coa is not None and len(cphd_meta.arp_pos_coa) == 3:
            sub(arp_pos, "X", str(cphd_meta.arp_pos_coa[0]))
            sub(arp_pos, "Y", str(cphd_meta.arp_pos_coa[1]))
            sub(arp_pos, "Z", str(cphd_meta.arp_pos_coa[2]))
        else:
            sub(arp_pos, "X", "0.0"); sub(arp_pos, "Y", "0.0"); sub(arp_pos, "Z", "0.0")
            
        arp_vel = sub(scpcoa, "ARPVel")
        if cphd_meta.arp_vel_coa is not None and len(cphd_meta.arp_vel_coa) == 3:
            sub(arp_vel, "X", str(cphd_meta.arp_vel_coa[0]))
            sub(arp_vel, "Y", str(cphd_meta.arp_vel_coa[1]))
            sub(arp_vel, "Z", str(cphd_meta.arp_vel_coa[2]))
        else:
            sub(arp_vel, "X", "1.0"); sub(arp_vel, "Y", "0.0"); sub(arp_vel, "Z", "0.0")
            
        arp_acc = sub(scpcoa, "ARPAcc")
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

        with open(output_path, "wb") as f, sksicd.NitfWriter(f, sicd_meta) as writer:
            writer.write_image(img_arr)
            
        return output_path

    def run(self):
        print(f"Reading CPHD: {self.cphd_path}")
        output_files = []
        with open(self.cphd_path, "rb") as f:
            reader = skcphd.Reader(f)
            cphd_meta = self._read_metadata(reader)
            
            xmltree = reader.metadata.xmltree
            channels = xmltree.findall(".//{*}Data/{*}Channel")
            channel_names = [ch.find("./{*}Identifier").text for ch in channels]
            
            pol_groups = {}
            for ch_id in channel_names:
                ch_nodes = xmltree.findall(".//{*}Channel/{*}Parameters")
                tx_pol, rcv_pol, fxc = "UNKNOWN", "UNKNOWN", 0.0
                for node in ch_nodes:
                    ident = node.find("./{*}Identifier")
                    if ident is not None and ident.text == ch_id:
                        tp = node.find("./{*}Polarization/{*}TxPol")
                        if tp is not None: tx_pol = tp.text
                        rp = node.find("./{*}Polarization/{*}RcvPol")
                        if rp is not None: rcv_pol = rp.text
                        fc = node.find("./{*}FxC")
                        if fc is not None: fxc = float(fc.text)
                        break
                
                pol_key = (tx_pol, rcv_pol)
                if pol_key not in pol_groups:
                    pol_groups[pol_key] = []
                pol_groups[pol_key].append((ch_id, fxc))
                
            os.makedirs(self.output_dir, exist_ok=True)
            u_min, u_max, r_min, r_max = self._determine_spatial_bounds(cphd_meta)
            
            for pol_key, ch_info_list in pol_groups.items():
                tx_pol, rcv_pol = pol_key
                print(f"Processing Polarization Group: {tx_pol}/{rcv_pol} with {len(ch_info_list)} channels.")
                
                channel_signals = []
                channel_pvps = []
                channel_fxcs = []
                channel_domains = []
                
                for ch_id, fxc in ch_info_list:
                    pvp_struct = reader.read_pvps(ch_id)
                    pvp_dict = {}
                    for name in pvp_struct.dtype.names:
                        arr = pvp_struct[name]
                        if hasattr(arr, "dtype") and arr.dtype.byteorder not in ("=", "|"):
                            arr = arr.astype(arr.dtype.newbyteorder("="))
                        pvp_dict[name] = np.ascontiguousarray(arr)
                    
                    sig_np = reader.read_signal(ch_id)
                    sig_tensor = torch.from_numpy(sig_np.astype(np.complex64)).cfloat()
                    
                    channel_signals.append(sig_tensor)
                    channel_pvps.append(pvp_dict)
                    channel_fxcs.append(fxc)
                    channel_domains.append(cphd_meta.domain_type)
                    
                print("Calling IFP_PerPolar...")
                img_cpu, bw_u_actual, bw_r_actual, N_u, N_r = pfa_per_polar(
                    channel_signals=channel_signals,
                    channel_pvps=channel_pvps,
                    channel_fxcs=channel_fxcs,
                    channel_domain_types=channel_domains,
                    cphd_meta=cphd_meta,
                    u_min=u_min,
                    u_max=u_max,
                    r_min=r_min,
                    r_max=r_max,
                    custom_pixel_spacing=self.custom_pixel_spacing,
                    device=self.device
                )
                
                name = Path(self.cphd_path).name
                name = str(name).split("_CPHD")[0] # specific to umbra
                out_name = f"{name}_SICDU_{tx_pol}_{rcv_pol}.nitf"
                out_path = os.path.join(self.output_dir, out_name)
                
                du = (u_max - u_min) / N_u
                dr = (r_max - r_min) / N_r
                
                print(f"Writing {out_name}...")
                self._write_sicd(out_path, img_cpu, cphd_meta, tx_pol, rcv_pol, bw_u_actual, bw_r_actual, N_u, N_r, u_min, r_min, du, dr)
                output_files.append(out_path)
                
        return output_files
