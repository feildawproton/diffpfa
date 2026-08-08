import numpy as np
import torch
from sarpy.io.phase_history.cphd import CPHDReader

from diffpfa.io.base import (
    BaseCPHDReader,
    CPHDChannelData,
    CPHDMetadata,
    ImageAreaBounds,
)


class SarpyCPHDReader(BaseCPHDReader):
    """sarpy-backed implementation of CPHDReader."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.reader = CPHDReader(file_path)
        self._meta_cache = None

    def get_metadata(self) -> CPHDMetadata:
        if self._meta_cache is not None:
            return self._meta_cache

        cphd_meta = self.reader.cphd_meta
        g_meta = cphd_meta.Global
        sc_meta = cphd_meta.SceneCoordinates

        domain_type = getattr(g_meta, "DomainType", "FX")
        sgn = getattr(g_meta, "SGN", -1)
        fx_min = g_meta.FxBand.FxMin if hasattr(g_meta, "FxBand") else 0.0
        fx_max = g_meta.FxBand.FxMax if hasattr(g_meta, "FxBand") else 0.0

        iarp_ecf = sc_meta.IARP.ECF.get_array()
        uIAX = sc_meta.ReferenceSurface.Planar.uIAX.get_array()
        uIAY = sc_meta.ReferenceSurface.Planar.uIAY.get_array()

        # Parse ImageArea
        img_area = None
        if hasattr(sc_meta, "ImageArea") and sc_meta.ImageArea is not None:
            ia = sc_meta.ImageArea
            x1, y1 = ia.X1Y1.X, ia.X1Y1.Y
            x2, y2 = ia.X2Y2.X, ia.X2Y2.Y
            poly = [(p.X, p.Y) for p in ia.Polygon] if hasattr(ia, "Polygon") and ia.Polygon else None
            img_area = ImageAreaBounds(x1=x1, y1=y1, x2=x2, y2=y2, polygon=poly)

        # Parse ExtendedArea
        ext_area = None
        if hasattr(sc_meta, "ExtendedArea") and sc_meta.ExtendedArea is not None:
            ea = sc_meta.ExtendedArea
            x1, y1 = ea.X1Y1.X, ea.X1Y1.Y
            x2, y2 = ea.X2Y2.X, ea.X2Y2.Y
            poly = [(p.X, p.Y) for p in ea.Polygon] if hasattr(ea, "Polygon") and ea.Polygon else None
            ext_area = ImageAreaBounds(x1=x1, y1=y1, x2=x2, y2=y2, polygon=poly)

        coll_start = None
        if hasattr(g_meta, "Timeline") and hasattr(g_meta.Timeline, "CollectionStart"):
            coll_start = str(g_meta.Timeline.CollectionStart)

        radar_mode = None
        if hasattr(cphd_meta, "CollectionID") and hasattr(cphd_meta.CollectionID, "RadarMode"):
            radar_mode = getattr(cphd_meta.CollectionID.RadarMode, "ModeType", None)

        self._meta_cache = CPHDMetadata(
            domain_type=domain_type,
            sgn=sgn,
            global_fx_min=fx_min,
            global_fx_max=fx_max,
            iarp_ecf=iarp_ecf,
            uIAX=uIAX,
            uIAY=uIAY,
            image_area=img_area,
            extended_area=ext_area,
            collection_start=coll_start,
            radar_mode=radar_mode,
            raw_meta=cphd_meta,
        )
        return self._meta_cache

    def get_channel_names(self) -> list[str]:
        names = []
        for p in self.reader.cphd_meta.Channel.Parameters:
            names.append(p.Identifier)
        return names

    def read_channel(self, channel_id: str) -> CPHDChannelData:
        ch_names = self.get_channel_names()
        if channel_id not in ch_names:
            raise ValueError(f"Channel {channel_id} not found in CPHD channels: {ch_names}")

        ch_idx = ch_names.index(channel_id)
        params = self.reader.cphd_meta.Channel.Parameters[ch_idx]

        # Read PVP array and convert fields to native byteorder
        pvp_struct = self.reader.read_pvp_array(channel_id)
        pvp_dict = {}
        for name in pvp_struct.dtype.names:
            arr = pvp_struct[name]
            if hasattr(arr, "dtype") and arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="))
            pvp_dict[name] = np.ascontiguousarray(arr)

        # Read signal array
        # CPHDReader index slice: [pulse_start:pulse_stop, sample_start:sample_stop, ch_idx]
        sig_np = self.reader[:, :, ch_idx]
        sig_tensor = torch.from_numpy(sig_np.copy()).cfloat()

        # Polarization extraction
        tx_pol = getattr(params, "TxPol", None)
        rcv_pol = getattr(params, "RcvPol", None)

        if tx_pol is None and "TxPol" in pvp_dict:
            tx_pol = str(pvp_dict["TxPol"][0])
        if rcv_pol is None and "RcvPol" in pvp_dict:
            rcv_pol = str(pvp_dict["RcvPol"][0])

        if tx_pol is None:
            tx_pol = "UNKNOWN"
        if rcv_pol is None:
            rcv_pol = "UNKNOWN"

        fxc = getattr(params, "FxC", 0.0)
        fxbw = getattr(params, "FxBW", 0.0)

        domain_type = getattr(self.reader.cphd_meta.Global, "DomainType", "FX")

        return CPHDChannelData(
            identifier=channel_id,
            tx_pol=str(tx_pol),
            rcv_pol=str(rcv_pol),
            fxc=float(fxc),
            fxbw=float(fxbw),
            signal=sig_tensor,
            pvp=pvp_dict,
            domain_type=domain_type,
        )

    def close(self):
        if hasattr(self, "reader") and self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
