import numpy as np
import torch
import sarkit.cphd as skcphd

from diffpfa.io.base import (
    BaseCPHDReader,
    CPHDChannelData,
    CPHDMetadata,
    ImageAreaBounds,
)

class SarkitCPHDReader(BaseCPHDReader):
    """sarkit-backed implementation of CPHDReader."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file = open(file_path, "rb")
        self.reader = skcphd.Reader(self.file)
        self._meta_cache = None

    def get_metadata(self) -> CPHDMetadata:
        if self._meta_cache is not None:
            return self._meta_cache

        xmltree = self.reader.metadata.xmltree
        xml_helper = skcphd.XmlHelper(xmltree)

        domain_type = xml_helper.load("./{*}Global/{*}DomainType")
        if domain_type is None:
            domain_type = "FX"
            
        sgn_val = xml_helper.load("./{*}Global/{*}SGN")
        print(f"SGN is {sgn_val}")
        sgn = sgn_val if sgn_val is not None else -1

        fx_min = xml_helper.load("./{*}Global/{*}FxBand/{*}FxMin")
        if fx_min is None:
            raise ValueError("CPHD missing required field: Global/FxBand/FxMin")
        fx_max = xml_helper.load("./{*}Global/{*}FxBand/{*}FxMax")
        if fx_max is None:
            raise ValueError("CPHD missing required field: Global/FxBand/FxMax")

        iarp_ecf = xml_helper.load("./{*}SceneCoordinates/{*}IARP/{*}ECF")
        if iarp_ecf is None:
            raise ValueError("CPHD missing required field: SceneCoordinates/IARP/ECF")

        uIAX = xml_helper.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX")
        if uIAX is None:
            raise ValueError("CPHD missing required field: uIAX")
        uIAY = xml_helper.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY")
        if uIAY is None:
            raise ValueError("CPHD missing required field: uIAY")

        # ImageArea and ExtendedArea parsing
        img_area = None
        ext_area = None
        
        # Helper for parsing polygon vertices
        def parse_polygon(poly_xpath):
            vertex_nodes = xmltree.findall(f"{poly_xpath}/{{*}}Vertex")
            if not vertex_nodes:
                return None
            polygon = []
            for node in vertex_nodes:
                x_node = node.find("./{*}X")
                y_node = node.find("./{*}Y")
                if x_node is not None and y_node is not None:
                    polygon.append((float(x_node.text), float(y_node.text)))
            return polygon

        ia_x1y1 = xml_helper.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1")
        ia_x2y2 = xml_helper.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2")
        if ia_x1y1 is not None and ia_x2y2 is not None:
            ia_poly = parse_polygon(".//{*}SceneCoordinates/{*}ImageArea/{*}Polygon")
            img_area = ImageAreaBounds(
                x1=ia_x1y1[0], y1=ia_x1y1[1], x2=ia_x2y2[0], y2=ia_x2y2[1], polygon=ia_poly
            )

        ea_x1y1 = xml_helper.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X1Y1")
        ea_x2y2 = xml_helper.load("./{*}SceneCoordinates/{*}ExtendedArea/{*}X2Y2")
        if ea_x1y1 is not None and ea_x2y2 is not None:
            ea_poly = parse_polygon(".//{*}SceneCoordinates/{*}ExtendedArea/{*}Polygon")
            ext_area = ImageAreaBounds(
                x1=ea_x1y1[0], y1=ea_x1y1[1], x2=ea_x2y2[0], y2=ea_x2y2[1], polygon=ea_poly
            )

        coll_start = xml_helper.load("./{*}Global/{*}Timeline/{*}CollectionStart")
        radar_mode = xml_helper.load("./{*}CollectionID/{*}RadarMode/{*}ModeType")
        classification = xml_helper.load("./{*}CollectionID/{*}Classification")

        ls = xml_helper.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAXExtent/{*}LineSpacing")
        ss = xml_helper.load("./{*}SceneCoordinates/{*}ImageGrid/{*}IAYExtent/{*}SampleSpacing")
        srp_ecf = xml_helper.load("./{*}ReferenceGeometry/{*}SRP/{*}ECF")
        arp_pos = xml_helper.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPPos")
        arp_vel = xml_helper.load("./{*}ReferenceGeometry/{*}Monostatic/{*}ARPVel")

        self._meta_cache = CPHDMetadata(
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
            radar_mode=str(radar_mode) if radar_mode else None,
            classification=str(classification) if classification else None,
            srp_ecf=srp_ecf,
            arp_pos_coa=arp_pos,
            arp_vel_coa=arp_vel,
            line_spacing=float(ls) if ls is not None else None,
            sample_spacing=float(ss) if ss is not None else None,
            raw_meta=xmltree,
        )
        return self._meta_cache

    def get_channel_names(self) -> list[str]:
        xmltree = self.reader.metadata.xmltree
        xml_helper = skcphd.XmlHelper(xmltree)
        # Sarkit nodes return strings directly if leaf
        channels = xmltree.findall(".//{*}Data/{*}Channel")
        names = []
        for ch in channels:
            ident = ch.find("./{*}Identifier")
            if ident is not None:
                names.append(ident.text)
        return names

    def read_channel(self, channel_id: str) -> CPHDChannelData:
        ch_names = self.get_channel_names()
        if channel_id not in ch_names:
            raise ValueError(f"Channel {channel_id} not found in CPHD channels: {ch_names}")

        pvp_struct = self.reader.read_pvps(channel_id)
        pvp_dict = {}
        for name in pvp_struct.dtype.names:
            arr = pvp_struct[name]
            if hasattr(arr, "dtype") and arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="))
            pvp_dict[name] = np.ascontiguousarray(arr)

        sig_np = self.reader.read_signal(channel_id)
        sig_np = sig_np.astype(np.complex64)
        sig_tensor = torch.from_numpy(sig_np).cfloat()

        # Find channel parameters
        xmltree = self.reader.metadata.xmltree
        ch_nodes = xmltree.findall(".//{*}Channel/{*}Parameters")
        params_node = None
        for node in ch_nodes:
            ident = node.find("./{*}Identifier")
            if ident is not None and ident.text == channel_id:
                params_node = node
                break

        tx_pol = None
        rcv_pol = None
        fxc = 0.0
        fxbw = 0.0

        if params_node is not None:
            pol_node = params_node.find("./{*}Polarization")
            if pol_node is not None:
                tp = pol_node.find("./{*}TxPol")
                if tp is not None: tx_pol = tp.text
                rp = pol_node.find("./{*}RcvPol")
                if rp is not None: rcv_pol = rp.text
            fc = params_node.find("./{*}FxC")
            if fc is not None: fxc = float(fc.text)
            fbw = params_node.find("./{*}FxBW")
            if fbw is not None: fxbw = float(fbw.text)

        if tx_pol is None and "TxPol" in pvp_dict:
            tx_pol = str(pvp_dict["TxPol"][0])
        if rcv_pol is None and "RcvPol" in pvp_dict:
            rcv_pol = str(pvp_dict["RcvPol"][0])

        if tx_pol is None: tx_pol = "UNKNOWN"
        if rcv_pol is None: rcv_pol = "UNKNOWN"

        meta = self.get_metadata()

        return CPHDChannelData(
            identifier=channel_id,
            tx_pol=str(tx_pol),
            rcv_pol=str(rcv_pol),
            fxc=float(fxc),
            fxbw=float(fxbw),
            signal=sig_tensor,
            pvp=pvp_dict,
            domain_type=meta.domain_type,
        )

    def close(self):
        if hasattr(self, "file") and self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None

    def __del__(self):
        self.close()
