import os
import numpy as np
import lxml.etree as ET
import datetime
from typing import Dict, Tuple, Optional
import sarkit.cphd as skcphd
import sarkit.sicd as sksicd
import sarkit.wgs84 as wgs84
import numpy.polynomial.polynomial as npp
from pathlib import Path
import concurrent.futures
import time

from diffpfa.types import CPHDMetadata, ImageAreaBounds
from diffpfa.IFA.PFA import pfa_per_polar
from diffpfa.IFA.kspace import compute_kspace
from diffpfa.constants import SPEED_OF_LIGHT

def _read_single_channel(cphd_path: str, ch_id: str, fxc: float, domain_type: str):
    """Worker function to read a single channel in its own thread/file handle."""
    import sarkit.cphd as skcphd

    # Each thread MUST open its own file handle
    with open(cphd_path, "rb") as f:
        reader = skcphd.Reader(f)

        # Read PVPs
        pvp_struct = reader.read_pvps(ch_id)
        pvp_dict = {}
        for name in pvp_struct.dtype.names:
            arr = pvp_struct[name]
            if hasattr(arr, "dtype") and arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="))
            pvp_dict[name] = np.ascontiguousarray(arr)

        # Read Signal
        sig_np = reader.read_signal(ch_id)

    return sig_np, pvp_dict, fxc, domain_type

class IFAProcessor:
    def __init__(self, 
                 cphd_path: str, 
                 output_dir: str, 
                 image_area_mode: str = "ImageArea", 
                 custom_pixel_spacing: Optional[Tuple[float, float]] = None, 
                 image_plane: str = "SLANT", 
                 image_oversample: float = 1.25, 
                 batch_size: int = 256,
                 pad_factor: float = 1.20,
                 device: str = "cuda"):
        self.cphd_path = cphd_path
        self.output_dir = output_dir
        self.image_area_mode = image_area_mode
        self.custom_pixel_spacing = custom_pixel_spacing
        self.image_plane = image_plane
        self.image_oversample = image_oversample
        self.batch_size = batch_size
        self.pad_factor = pad_factor
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
        ref_ch_id = xml_helper.load("./{*}Channel/{*}RefChId")

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
        side_of_track = xml_helper.load("./{*}ReferenceGeometry/{*}Monostatic/{*}SideOfTrack")
        side_of_track = str(side_of_track) if side_of_track is not None else "L"

        line_spacing = xml_helper.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}LineSpacing")
        sample_spacing = xml_helper.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}SampleSpacing")

        classification = xml_helper.load("./{*}CollectionID/{*}Classification")

        return CPHDMetadata(
            domain_type=str(domain_type),
            sgn=int(sgn),
            global_fx_min=float(fx_min),
            global_fx_max=float(fx_max),
            iarp_ecf=iarp_ecf,
            uIAX=uIAX,
            uIAY=uIAY,
            ref_ch_id = ref_ch_id,
            image_area=img_area,
            extended_area=ext_area,
            collection_start=str(coll_start) if coll_start else None,
            radar_mode="UNKNOWN",
            classification=str(classification) if classification else "UNCLASSIFIED",
            srp_ecf=srp_ecf,
            arp_pos_coa=arp_pos,
            arp_vel_coa=arp_vel,
            side_of_track=side_of_track,
            line_spacing=line_spacing,
            sample_spacing=sample_spacing,
            raw_meta=xmltree,
            ref_uIAX=uIAX.copy() if uIAX is not None else None,
            ref_uIAY=uIAY.copy() if uIAY is not None else None,
        )

    def _determine_spatial_bounds(self, cphd_meta):
        """
        Determines the spatial extent [u_min, u_max] x [r_min, r_max] in meters.

        NOTE ON RADAR RECTIFICATION (Audit Defect C3):
        Previously, this method set:
            u_min, u_max = min(ia.x1, ia.x2), max(ia.x1, ia.x2)
            r_min, r_max = min(ia.y1, ia.y2), max(ia.y1, ia.y2)
        directly for both GROUND and SLANT modes.
        This was a fundamental error in radar geometry understanding:
        ImageArea/X1Y1, X2Y2 in CPHD DIDD §6.2 are defined on the CPHD ReferenceSurface
        (ground plane spanned by ref_uIAX and ref_uIAY) relative to IARP.
        Using those ground extents directly as slant-plane extents caused a ~1.41x
        over-coverage in slant range (failing to account for the grazing/depression angle)
        and clipped azimuth by ~5% (because ground reference axes uIAX/uIAY are not aligned
        with the slant plane velocity/cross-range axis).
        
        To rectify this:
        When image_plane == 'SLANT':
            Project the 4 ground corners of the ImageArea onto the slant plane basis vectors:
                u_row = LOS unit vector (from ARP to SRP)
                u_col = cross-range unit vector (perpendicular to LOS along velocity)
            Then apply pad_factor (default 1.20) around the bounding box center to reproduce
            the standard slant over-formation extents.
        When image_plane == 'GROUND':
            The bounds along ref_uIAX and ref_uIAY are used directly without projection.
        """
        mode = self.image_area_mode
        area = cphd_meta.image_area if mode == "ImageArea" else cphd_meta.extended_area
        if area is None:
            return -100.0, 100.0, -100.0, 100.0

        x1, y1 = min(area.x1, area.x2), min(area.y1, area.y2)
        x2, y2 = max(area.x1, area.x2), max(area.y1, area.y2)

        if self.image_plane.upper() == "SLANT":
            # In SLANT mode, run() sets cphd_meta.uIAX = u_col and cphd_meta.uIAY = u_row
            u_col = cphd_meta.uIAX
            u_row = cphd_meta.uIAY

            # Original CPHD ground reference surface vectors
            ref_uIAX = cphd_meta.ref_uIAX if cphd_meta.ref_uIAX is not None else u_col
            ref_uIAY = cphd_meta.ref_uIAY if cphd_meta.ref_uIAY is not None else u_row

            # IARP offset relative to SRP (in ECF)
            iarp_offset = (cphd_meta.iarp_ecf - cphd_meta.srp_ecf) if (
                cphd_meta.iarp_ecf is not None and cphd_meta.srp_ecf is not None
            ) else np.zeros(3)

            ground_corners = [
                np.array([x1, y1]),
                np.array([x1, y2]),
                np.array([x2, y2]),
                np.array([x2, y1]),
            ]

            proj_r = []
            proj_u = []
            for c in ground_corners:
                dp = iarp_offset + c[0] * ref_uIAX + c[1] * ref_uIAY
                proj_r.append(float(np.dot(dp, u_row)))
                proj_u.append(float(np.dot(dp, u_col)))

            r_min_raw, r_max_raw = min(proj_r), max(proj_r)
            u_min_raw, u_max_raw = min(proj_u), max(proj_u)

            # Apply pad_factor around center of projected bounding box
            r_c = 0.5 * (r_min_raw + r_max_raw)
            r_half = 0.5 * (r_max_raw - r_min_raw) * self.pad_factor
            r_min, r_max = r_c - r_half, r_c + r_half

            u_c = 0.5 * (u_min_raw + u_max_raw)
            u_half = 0.5 * (u_max_raw - u_min_raw) * self.pad_factor
            u_min, u_max = u_c - u_half, u_c + u_half
        else:
            u_min, u_max = x1, x2
            r_min, r_max = y1, y2

        return u_min, u_max, r_min, r_max
        
    def _write_sicd(
        self,
        output_path: str,
        img_cpu: np.ndarray,
        cphd_meta,
        tx_pol,
        rcv_pol,
        bw_range,
        bw_azm,
        N_range,
        N_azm,
        u_min,
        r_min,
        du_azm,
        dr_range,
        ref_pvp: Optional[dict] = None,
        num_samples: Optional[int] = None,
        is_rotated: bool = False,
        channel_pvps: Optional[list] = None,
        channel_signals: Optional[list] = None,
        u_c: Optional[float] = None,
        r_c: Optional[float] = None
    ):
        
        num_rows, num_cols = img_cpu.shape
        
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
        sub(col_info, "Classification", cphd_meta.classification)

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

        if u_c is None:
            u_c = u_min + 0.5 * N_azm * du_azm
        if r_c is None:
            r_c = r_min + 0.5 * N_range * dr_range

        # SCP Pixel coordinates in image grid (rows=range, cols=azimuth)
        # Accounting for any asymmetric spatial framing offset (u_c, r_c) (Audit G2 remediation)
        scp_row = int(round(num_rows / 2.0 - r_c / dr_range))
        scp_col = int(round(num_cols / 2.0 - u_c / du_azm))

        sp = sub(img_data, "SCPPixel")
        sub(sp, "Row", str(scp_row))
        sub(sp, "Col", str(scp_col))

        geo_data = sub(root, "GeoData")
        sub(geo_data, "EarthModel", "WGS_84")
        scp_elem = sub(geo_data, "SCP")
        ecf = sub(scp_elem, "ECF")
        sub(ecf, "X", str(cphd_meta.iarp_ecf[0]))
        sub(ecf, "Y", str(cphd_meta.iarp_ecf[1]))
        sub(ecf, "Z", str(cphd_meta.iarp_ecf[2]))

        llh = sub(scp_elem, "LLH")
        lat_deg, lon_deg, hae = wgs84.cartesian_to_geodetic(cphd_meta.iarp_ecf)
        sub(llh, "Lat", f"{np.clip(lat_deg, -90.0, 90.0):.9f}")
        sub(llh, "Lon", f"{np.clip(lon_deg, -180.0, 180.0):.9f}")
        sub(llh, "HAE", f"{hae:.9f}")

        # Determine Row/Col basis vectors based on orientation
        u_row_vec = cphd_meta.uIAX if is_rotated else cphd_meta.uIAY
        u_col_vec = cphd_meta.uIAY if is_rotated else cphd_meta.uIAX

        # Initial ImageCorners placeholder
        ic = sub(geo_data, "ImageCorners")
        row_extent = num_rows * dr_range
        col_extent = num_cols * du_azm
        r_deg = row_extent / 6378137.0 * 180.0 / np.pi
        c_deg = col_extent / (6378137.0 * max(0.01, np.cos(np.radians(lat_deg)))) * 180.0 / np.pi
        sub(sub(ic, "ICP", index="1:FRFC"), "Lat", f"{np.clip(lat_deg + r_deg/2, -90, 90):.9f}")
        ic.find("./{*}ICP[@index='1:FRFC']").append(ET.Element("{urn:SICD:1.3.0}Lon"))
        ic.find("./{*}ICP[@index='1:FRFC']/{*}Lon").text = f"{np.clip(lon_deg - c_deg/2, -180, 180):.9f}"

        sub(sub(ic, "ICP", index="2:FRLC"), "Lat", f"{np.clip(lat_deg + r_deg/2, -90, 90):.9f}")
        ic.find("./{*}ICP[@index='2:FRLC']").append(ET.Element("{urn:SICD:1.3.0}Lon"))
        ic.find("./{*}ICP[@index='2:FRLC']/{*}Lon").text = f"{np.clip(lon_deg + c_deg/2, -180, 180):.9f}"

        sub(sub(ic, "ICP", index="3:LRLC"), "Lat", f"{np.clip(lat_deg - r_deg/2, -90, 90):.9f}")
        ic.find("./{*}ICP[@index='3:LRLC']").append(ET.Element("{urn:SICD:1.3.0}Lon"))
        ic.find("./{*}ICP[@index='3:LRLC']/{*}Lon").text = f"{np.clip(lon_deg + c_deg/2, -180, 180):.9f}"

        sub(sub(ic, "ICP", index="4:LRFC"), "Lat", f"{np.clip(lat_deg - r_deg/2, -90, 90):.9f}")
        ic.find("./{*}ICP[@index='4:LRFC']").append(ET.Element("{urn:SICD:1.3.0}Lon"))
        ic.find("./{*}ICP[@index='4:LRFC']/{*}Lon").text = f"{np.clip(lon_deg - c_deg/2, -180, 180):.9f}"

        # Calculate exact geometry and kinematics when ref_pvp is provided (Audit C2 remediation)
        has_pvp = (ref_pvp is not None and "TxTime" in ref_pvp and "TxPos" in ref_pvp and num_samples is not None)
        if has_pvp:
            t_tx = np.asarray(ref_pvp["TxTime"], dtype=np.float64)
            t_rcv = np.asarray(ref_pvp.get("RcvTime", ref_pvp["TxTime"]), dtype=np.float64)
            t_mid = 0.5 * (t_tx + t_rcv)
            rcv_pos = ref_pvp.get("RcvPos", ref_pvp["TxPos"])
            arp_mid = 0.5 * (np.asarray(ref_pvp["TxPos"], dtype=np.float64) + np.asarray(rcv_pos, dtype=np.float64))

            # Reference channel k-space for polynomial fits
            Ku_ref, Kr_ref = compute_kspace(ref_pvp, u_col_vec, u_row_vec, num_samples, cphd_meta.domain_type, device="cpu")
            Ku_ref = Ku_ref.numpy() if hasattr(Ku_ref, "numpy") else np.asarray(Ku_ref)
            Kr_ref = Kr_ref.numpy() if hasattr(Kr_ref, "numpy") else np.asarray(Kr_ref)
            ns_ = Ku_ref.shape[1]
            Ku_mid = Ku_ref[:, ns_ // 2]
            Kr_mid = Kr_ref[:, ns_ // 2]
            plr = np.arctan2(Ku_mid, Kr_mid)
            fit_deg = min(5, max(1, len(t_mid) - 1))
            plr_coef = npp.polyfit(t_mid, plr, fit_deg)

            # Reference time: zero of polar angle nearest aperture centre
            roots = npp.polyroots(plr_coef)
            roots = roots[np.isreal(roots)].real
            if len(roots) > 0:
                t_ref = float(roots[np.argmin(np.abs(roots - t_mid.mean()))])
            else:
                t_ref = float(t_mid.mean())

            # Spatial frequency scale factor as function of polar angle
            F_mid = np.asarray(ref_pvp["SC0"], dtype=np.float64) + (ns_ // 2) * np.asarray(ref_pvp["SCSS"], dtype=np.float64)
            ksf = np.sqrt(Ku_mid**2 + Kr_mid**2) / (2.0 * F_mid / SPEED_OF_LIGHT)
            ksf_coef = npp.polyfit(plr, ksf, fit_deg)

            # ARP polynomial in absolute time (since CollectionStart)
            arp_coef = np.stack([npp.polyfit(t_mid, arp_mid[:, i], fit_deg) for i in range(3)])

            # Determine k-space extents across all channels of the polarization group (Audit G1 remediation)
            pvps_to_compute = []
            samples_to_compute = []
            if channel_pvps is not None and len(channel_pvps) > 0:
                for idx, ch_pvp in enumerate(channel_pvps):
                    ns_ch = channel_signals[idx].shape[1] if (channel_signals is not None and idx < len(channel_signals) and hasattr(channel_signals[idx], "shape")) else num_samples
                    pvps_to_compute.append(ch_pvp)
                    samples_to_compute.append(ns_ch)
            else:
                pvps_to_compute.append(ref_pvp)
                samples_to_compute.append(num_samples)

            kr_min_val, kr_max_val = float("inf"), float("-inf")
            ku_min_val, ku_max_val = float("inf"), float("-inf")
            for pvp_i, ns_i in zip(pvps_to_compute, samples_to_compute):
                Kui, Kri = compute_kspace(pvp_i, u_col_vec, u_row_vec, ns_i, cphd_meta.domain_type, device="cpu")
                Kui = Kui.numpy() if hasattr(Kui, "numpy") else np.asarray(Kui)
                Kri = Kri.numpy() if hasattr(Kri, "numpy") else np.asarray(Kri)
                kr_min_val = min(kr_min_val, float(Kri.min()))
                kr_max_val = max(kr_max_val, float(Kri.max()))
                ku_min_val = min(ku_min_val, float(Kui.min()))
                ku_max_val = max(ku_max_val, float(Kui.max()))

            # If only single-channel PVP was passed but CPHD metadata declares a wider multi-band FX range,
            # synthesize the full fast-time band so metadata reflects the full processed spectrum.
            if len(pvps_to_compute) == 1 and getattr(cphd_meta, "global_fx_min", None) is not None and getattr(cphd_meta, "global_fx_max", None) is not None:
                chan_fx_min = float(np.min(ref_pvp["SC0"]))
                chan_fx_max = float(np.max(ref_pvp["SC0"] + (num_samples - 1) * ref_pvp["SCSS"]))
                if cphd_meta.global_fx_min < chan_fx_min - 1e3 or cphd_meta.global_fx_max > chan_fx_max + 1e3:
                    pvp_full = dict(ref_pvp)
                    ns_synth = max(num_samples, 256)
                    pvp_full["SC0"] = np.full(len(ref_pvp["SC0"]), cphd_meta.global_fx_min)
                    pvp_full["SCSS"] = np.full(len(ref_pvp["SC0"]), (cphd_meta.global_fx_max - cphd_meta.global_fx_min) / max(ns_synth - 1, 1))
                    Kui, Kri = compute_kspace(pvp_full, u_col_vec, u_row_vec, ns_synth, cphd_meta.domain_type, device="cpu")
                    Kui = Kui.numpy() if hasattr(Kui, "numpy") else np.asarray(Kui)
                    Kri = Kri.numpy() if hasattr(Kri, "numpy") else np.asarray(Kri)
                    kr_min_val = min(kr_min_val, float(Kri.min()))
                    kr_max_val = max(kr_max_val, float(Kri.max()))
                    ku_min_val = min(ku_min_val, float(Kui.min()))
                    ku_max_val = max(ku_max_val, float(Kui.max()))

            krg1, krg2 = kr_min_val, kr_max_val
            kaz1, kaz2 = ku_min_val, ku_max_val
            kctr_dict = {
                "Row": float(0.5 * (krg1 + krg2)),
                "Col": float(0.5 * (kaz1 + kaz2))
            }
            collect_duration = float(max(t_tx[-1], t_rcv[-1]))
            t_start_proc = float(t_tx[0])
            t_end_proc = float(t_tx[-1])
        else:
            t_ref = 0.0
            kctr_dict = {"Row": 0.0, "Col": 0.0}
            collect_duration = 0.0
            t_start_proc = 0.0
            t_end_proc = 0.0
            arp_coef = np.zeros((3, 1))
            plr_coef = np.zeros(1)
            ksf_coef = np.ones(1)
            krg1, krg2, kaz1, kaz2 = 0.0, 0.0, 0.0, 0.0

        # --- Grid ---
        grid = sub(root, "Grid")
        sub(grid, "ImagePlane", self.image_plane.upper())
        sub(grid, "Type", "RGAZIM")
        time_coa = sub(grid, "TimeCOAPoly", order1="0", order2="0")
        sub(time_coa, "Coef", f"{t_ref:.12f}", exponent1="0", exponent2="0")

        # Row maps to Range, Col maps to Azimuth
        K_UNIFORM = 0.8859  # DIDD §4.14.6 normative uniform window resolution constant
        for dir_name, ss, bw, uvect in [("Row", dr_range, bw_range, u_row_vec), ("Col", du_azm, bw_azm, u_col_vec)]:
            d = sub(grid, dir_name)
            uv = sub(d, "UVectECF")
            sub(uv, "X", str(uvect[0]))
            sub(uv, "Y", str(uvect[1]))
            sub(uv, "Z", str(uvect[2]))
            sub(d, "SS", str(ss))
            sub(d, "ImpRespWid", f"{K_UNIFORM / max(1e-12, bw):.12f}")
            sub(d, "Sgn", "-1")
            sub(d, "ImpRespBW", str(bw))
            sub(d, "KCtr", f"{kctr_dict[dir_name]:.12f}")
            sub(d, "DeltaK1", str(-bw / 2.0))
            sub(d, "DeltaK2", str(bw / 2.0))
            wgt = sub(d, "WgtType")
            sub(wgt, "WindowName", "UNIFORM")

        # --- Timeline ---
        timeline = sub(root, "Timeline")
        if cphd_meta.collection_start:
            collect_start = str(cphd_meta.collection_start).replace(" ", "T").replace("+00:00", "Z")
        else:
            collect_start = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        sub(timeline, "CollectStart", collect_start)
        sub(timeline, "CollectDuration", f"{collect_duration:.9f}")

        # --- Position ---
        pos = sub(root, "Position")
        arp = sub(pos, "ARPPoly")
        for i, coord in enumerate(["X", "Y", "Z"]):
            coord_elem = sub(arp, coord, order1=str(len(arp_coef[i]) - 1))
            for k, val in enumerate(arp_coef[i]):
                sub(coord_elem, "Coef", f"{val:.15e}", exponent1=str(k))

        # --- RadarCollection ---
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

        # --- ImageFormation ---
        img_form = sub(root, "ImageFormation")
        rcv_proc = sub(img_form, "RcvChanProc")
        sub(rcv_proc, "NumChanProc", "1")
        sub(rcv_proc, "PRFScaleFactor", "1.0")
        sub(rcv_proc, "ChanIndex", "1")
        sub(img_form, "TxRcvPolarizationProc", f"{tx_pol}:{rcv_pol}")
        sub(img_form, "TStartProc", f"{t_start_proc:.9f}")
        sub(img_form, "TEndProc", f"{t_end_proc:.9f}")
        tx_proc = sub(img_form, "TxFrequencyProc")
        sub(tx_proc, "MinProc", str(cphd_meta.global_fx_min))
        sub(tx_proc, "MaxProc", str(cphd_meta.global_fx_max))
        sub(img_form, "ImageFormAlgo", "PFA")
        sub(img_form, "STBeamComp", "NO")
        sub(img_form, "ImageBeamComp", "NO")
        sub(img_form, "AzAutofocus", "NO")
        sub(img_form, "RgAutofocus", "NO")

        # --- SCPCOA ---
        scpcoa = sub(root, "SCPCOA")
        sub(scpcoa, "SCPTime", f"{t_ref:.12f}")
        sub(scpcoa, "SideOfTrack", cphd_meta.side_of_track)
        if has_pvp:
            try:
                temp_tree = ET.ElementTree(root)
                new_scpcoa = sksicd.compute_scp_coa(temp_tree)
                root.replace(scpcoa, new_scpcoa)
                scpcoa = new_scpcoa
            except Exception:
                pass
        if scpcoa.find("./{*}ARPPos") is None:
            # Fallback basic geometry if compute_scp_coa was not run
            arp_pos = sub(scpcoa, "ARPPos")
            sub(arp_pos, "X", "0.0"); sub(arp_pos, "Y", "0.0"); sub(arp_pos, "Z", "0.0")
            arp_vel = sub(scpcoa, "ARPVel")
            sub(arp_vel, "X", "1.0"); sub(arp_vel, "Y", "0.0"); sub(arp_vel, "Z", "0.0")
            arp_acc = sub(scpcoa, "ARPAcc")
            sub(arp_acc, "X", "0.0"); sub(arp_acc, "Y", "0.0"); sub(arp_acc, "Z", "0.0")
            sub(scpcoa, "SlantRange", "0.0")
            sub(scpcoa, "GroundRange", "0.0")
            sub(scpcoa, "DopplerConeAng", "90.0")
            sub(scpcoa, "GrazeAng", "45.0")
            sub(scpcoa, "IncidenceAng", "45.0")
            sub(scpcoa, "TwistAng", "0.0")
            sub(scpcoa, "SlopeAng", "0.0")
            sub(scpcoa, "AzimAng", "0.0")
            sub(scpcoa, "LayoverAng", "0.0")

        # --- Radiometric Calibration (Relative) ---
        rad = sub(root, "Radiometric")
        noise = sub(rad, "NoiseLevel")
        sub(noise, "NoiseLevelType", "ABSOLUTE")
        noise_poly = sub(noise, "NoisePoly", order1="0", order2="0")
        sub(noise_poly, "Coef", "0.0", exponent1="0", exponent2="0")

        rcssf = 1.0
        slant_area = dr_range * du_azm
        beta_zero = rcssf / slant_area if slant_area > 0 else 1.0

        # -- NOTE (Audit C8 Remediated): Radiometric Calibration Ratios --
        # Per SICD DIDD §4.10.4:
        #   BetaZero (beta_0) is reflectivity per unit area in the slant plane.
        #   SigmaZero (sigma_0) is reflectivity per unit area on the ground surface:
        #       sigma_0 = beta_0 * cos(SlopeAng)
        #       (Previously mistakenly coded as beta_0 * cos(GrazeAng)).
        #   GammaZero (gamma_0) is reflectivity per unit area normal to the slant range vector:
        #       gamma_0 = sigma_0 / cos(IncidenceAng) = beta_0 * cos(SlopeAng) / cos(IncidenceAng)
        #       (Previously mistakenly coded as beta_0 * sin(GrazeAng)).
        slope_elem = scpcoa.find("./{*}SlopeAng")
        inc_elem = scpcoa.find("./{*}IncidenceAng")
        if slope_elem is not None and inc_elem is not None and slope_elem.text and inc_elem.text:
            slope_rad = np.radians(float(slope_elem.text))
            inc_rad = np.radians(float(inc_elem.text))
            sigma_zero = beta_zero * np.cos(slope_rad)
            gamma_zero = sigma_zero / max(1e-12, np.cos(inc_rad))
        else:
            slope_rad = np.radians(45.0)
            inc_rad = np.radians(45.0)
            sigma_zero = beta_zero * np.cos(slope_rad)
            gamma_zero = sigma_zero / max(1e-12, np.cos(inc_rad))

        for poly_name, poly_val in [
            ("RCSSFPoly", rcssf),
            ("SigmaZeroSFPoly", sigma_zero),
            ("BetaZeroSFPoly", beta_zero),
            ("GammaZeroSFPoly", gamma_zero)
        ]:
            poly = sub(rad, poly_name, order1="0", order2="0")
            sub(poly, "Coef", f"{poly_val:.6e}", exponent1="0", exponent2="0")

        # --- PFA Block ---
        pfa = sub(root, "PFA")
        ipn = np.cross(u_row_vec, u_col_vec)
        ipn /= np.linalg.norm(ipn)
        fpn = sub(pfa, "FPN")
        sub(fpn, "X", f"{ipn[0]:.12f}")
        sub(fpn, "Y", f"{ipn[1]:.12f}")
        sub(fpn, "Z", f"{ipn[2]:.12f}")

        ipn_elem = sub(pfa, "IPN")
        sub(ipn_elem, "X", f"{ipn[0]:.12f}")
        sub(ipn_elem, "Y", f"{ipn[1]:.12f}")
        sub(ipn_elem, "Z", f"{ipn[2]:.12f}")

        sub(pfa, "PolarAngRefTime", f"{t_ref:.12f}")

        pap = sub(pfa, "PolarAngPoly", order1=str(len(plr_coef) - 1))
        for k, val in enumerate(plr_coef):
            sub(pap, "Coef", f"{val:.15e}", exponent1=str(k))

        sf_poly = sub(pfa, "SpatialFreqSFPoly", order1=str(len(ksf_coef) - 1))
        for k, val in enumerate(ksf_coef):
            sub(sf_poly, "Coef", f"{val:.15e}", exponent1=str(k))

        sub(pfa, "Krg1", f"{krg1:.12f}")
        sub(pfa, "Krg2", f"{krg2:.12f}")
        sub(pfa, "Kaz1", f"{kaz1:.12f}")
        sub(pfa, "Kaz2", f"{kaz2:.12f}")

        # --- Refine ImageCorners via ground-plane projection ---
        if has_pvp:
            try:
                temp_tree = ET.ElementTree(root)
                corners_rc = np.array([
                    [0, 0],
                    [0, num_cols - 1],
                    [num_rows - 1, num_cols - 1],
                    [num_rows - 1, 0]
                ], dtype=np.float64)
                xrow_ycol = np.stack([
                    (corners_rc[:, 0] - scp_row) * dr_range,
                    (corners_rc[:, 1] - scp_col) * du_azm
                ], axis=1)
                scp_ecf = np.asarray(cphd_meta.iarp_ecf, dtype=np.float64)
                up = wgs84.up(wgs84.cartesian_to_geodetic(scp_ecf))
                gpp, delta, ok = sksicd.image_to_ground_plane(temp_tree, xrow_ycol, scp_ecf, up)
                if ok:
                    llh = wgs84.cartesian_to_geodetic(gpp)
                    ic_elem = root.find("./{*}GeoData/{*}ImageCorners")
                    for c in list(ic_elem):
                        ic_elem.remove(c)
                    indices = ["1:FRFC", "2:FRLC", "3:LRLC", "4:LRFC"]
                    for idx, (lat, lon, _) in zip(indices, llh):
                        icp = sub(ic_elem, "ICP", index=idx)
                        sub(icp, "Lat", f"{lat:.9f}")
                        sub(icp, "Lon", f"{lon:.9f}")
            except Exception:
                pass

        xmltree = ET.ElementTree(root)
        clas_char = cphd_meta.classification[0].upper() if cphd_meta.classification else "U"
        sec = sksicd.NitfSecurityFields(clas=clas_char)
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
            writer.write_image(img_cpu)
            
        return output_path

    def run(self):
        print(f"Reading CPHD: {self.cphd_path}")
        t0 = time.perf_counter()
        read_time = 0
        output_files = []
        with open(self.cphd_path, "rb") as f:
            reader = skcphd.Reader(f)
            cphd_meta = self._read_metadata(reader)
            
            if self.image_plane.upper() == "SLANT":
                srp = cphd_meta.srp_ecf
                arp = cphd_meta.arp_pos_coa
                arp_v = cphd_meta.arp_vel_coa
                
                p_vec = srp - arp
                u_row = p_vec / np.linalg.norm(p_vec)
                
                u_v = arp_v / np.linalg.norm(arp_v)
                u_col_unnorm = u_v - np.dot(u_v, u_row) * u_row
                u_col = u_col_unnorm / np.linalg.norm(u_col_unnorm)
                
                if cphd_meta.side_of_track == "L":
                    u_col = -u_col
                
                cphd_meta.uIAX = u_col
                cphd_meta.uIAY = u_row
                self.custom_pixel_spacing = None
            
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
            
            # need for correcting phase to reference channel RcvTime
            ref_pvp = reader.read_pvps(cphd_meta.ref_ch_id)
            ref_rcv_time = np.ascontiguousarray(ref_pvp["RcvTime"].astype(ref_pvp["RcvTime"].dtype.newbyteorder("=")))

            read_time = time.perf_counter() - t0

            proc_time = 0
            write_time = 0
            for pol_key, ch_info_list in pol_groups.items():
                start_copy = time.perf_counter()
                tx_pol, rcv_pol = pol_key
                print(f"Processing Polarization Group: {tx_pol}/{rcv_pol} with {len(ch_info_list)} channels.")
               
                channel_signals = [None] * len(ch_info_list)
                channel_pvps = [None] * len(ch_info_list)
                channel_fxcs = [None] * len(ch_info_list)
                channel_domains = [None] * len(ch_info_list)

                # You can adjust max_workers based on your disk's parallel read capabilities (e.g. NVMe vs HDD)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit all channel read tasks
                    future_to_index = {
                        executor.submit(_read_single_channel, self.cphd_path, ch_id, fxc, cphd_meta.domain_type): i
                        for i, (ch_id, fxc) in enumerate(ch_info_list)
                    }

                    # Harvest results as they complete, maintaining original list order
                    for future in concurrent.futures.as_completed(future_to_index):
                        i = future_to_index[future]
                        sig_np, pvp_dict, fxc, domain_type = future.result()

                        channel_signals[i] = sig_np
                        channel_pvps[i] = pvp_dict
                        channel_fxcs[i] = fxc
                        channel_domains[i] = domain_type
 
                # 1. Use user-provided spacing
                # 2. Fall back to CPHD suggested grid spacing
                # 3. Fall back to None (let pfa_per_polar calculate Nyquist limit)
                active_spacing = self.custom_pixel_spacing
                if active_spacing is None and self.image_plane == "GROUND":
                    active_spacing = (cphd_meta.line_spacing, cphd_meta.sample_spacing)

                stop_copy = time.perf_counter()

                read_time += stop_copy - start_copy

                print("Calling IFP_PerPolar...")
                img_cpu, bw_range, bw_azm, N_range, N_azm, is_rotated = pfa_per_polar(
                    channel_signals=channel_signals,
                    channel_pvps=channel_pvps,
                    channel_fxcs=channel_fxcs,
                    channel_domain_types=channel_domains,
                    ref_rcv_time=ref_rcv_time,
                    cphd_meta=cphd_meta,
                    u_min=u_min,
                    u_max=u_max,
                    r_min=r_min,
                    r_max=r_max,
                    custom_pixel_spacing=active_spacing,
                    image_oversample=self.image_oversample,
                    batch_size=self.batch_size,
                    device=self.device
                )

                img_cpu = img_cpu.T # either SICD wants different x-y than natural from cphd or i'm confused as usual

                stop_proc = time.perf_counter()
                proc_time += stop_proc - stop_copy

                name = Path(self.cphd_path).name
                name = str(name).split("_CPHD")[0] # specific to umbra
                out_name = f"{name}_SICDU_{tx_pol}_{rcv_pol}.nitf"
                out_path = os.path.join(self.output_dir, out_name)
                
                du_azm = (u_max - u_min) / N_azm
                dr_range = (r_max - r_min) / N_range
                
                print(f"Writing {out_name}...")
                self._write_sicd(
                    out_path,
                    img_cpu,
                    cphd_meta,
                    tx_pol,
                    rcv_pol,
                    bw_range,
                    bw_azm,
                    N_range,
                    N_azm,
                    u_min,
                    r_min,
                    du_azm,
                    dr_range,
                    ref_pvp=channel_pvps[0],
                    num_samples=channel_signals[0].shape[1],
                    is_rotated=is_rotated,
                    channel_pvps=channel_pvps,
                    channel_signals=channel_signals,
                    u_c=(u_min + u_max) / 2.0,
                    r_c=(r_min + r_max) / 2.0
                )
                output_files.append(out_path)
                write_time += (time.perf_counter() - stop_proc)

        return (output_files, read_time, proc_time, write_time)
