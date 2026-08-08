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
        sgn = sgn_val if sgn_val is not None else -1

        fx_min = xml_helper.load("./{*}Global/{*}FxBand/{*}FxMin")
        if fx_min is None:
            fx_min = 0.0
        fx_max = xml_helper.load("./{*}Global/{*}FxBand/{*}FxMax")
        if fx_max is None:
            fx_max = 0.0

        iarp_ecf = xml_helper.load("./{*}SceneCoordinates/{*}IARP/{*}ECF")
        if iarp_ecf is None:
            iarp_ecf = np.array([0.0, 0.0, 0.0])

        uIAX = xml_helper.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX")
        if uIAX is None:
            uIAX = np.array([1.0, 0.0, 0.0])
        uIAY = xml_helper.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY")
        if uIAY is None:
            uIAY = np.array([0.0, 1.0, 0.0])

        # ImageArea (not fully implemented in sarkit wrapper, keeping simple)
        img_area = None
        ext_area = None

        coll_start = xml_helper.load("./{*}Global/{*}Timeline/{*}CollectionStart")
        radar_mode = xml_helper.load("./{*}CollectionID/{*}RadarMode/{*}ModeType")

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
            tp = params_node.find("./{*}TxPol")
            if tp is not None: tx_pol = tp.text
            rp = params_node.find("./{*}RcvPol")
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
