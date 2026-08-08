import datetime
import os
import numpy as np
import torch

from sarpy.geometry import geocoords
from sarpy.io.complex.sicd import SICDWriter
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Grid import DirParamType, GridType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.ImageFormation import (
    ImageFormationType,
    TxFrequencyProcType,
)
from sarpy.io.complex.sicd_elements.Position import PositionType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType

from diffpfa.io.base import BaseSICDWriter, CPHDMetadata, SICDImagePayload


class SarpySICDWriter(BaseSICDWriter):
    """sarpy-backed implementation for writing uncompensated SICD (SICD-U) NITF files."""

    def write_sicd(
        self,
        output_path: str,
        payload: SICDImagePayload,
        cphd_meta: CPHDMetadata
    ) -> str:
        # Convert PyTorch tensor complex image to 2D numpy complex64 array (NumRows, NumCols)
        if isinstance(payload.complex_image, torch.Tensor):
            img_arr = payload.complex_image.detach().cpu().numpy()
        else:
            img_arr = np.asarray(payload.complex_image)

        num_rows, num_cols = img_arr.shape
        img_arr = img_arr.astype(np.complex64)

        sicd = SICDType()

        # Image Data
        sicd.ImageData = ImageDataType(
            NumRows=num_rows,
            NumCols=num_cols,
            FirstRow=0,
            FirstCol=0,
            PixelType="RE32F_IM32F",
        )

        # Geo Data (IARP)
        ecf = payload.iarp_ecf
        try:
            llh = geocoords.ecf_to_geodetic(ecf)
        except Exception:
            llh = [0.0, 0.0, 0.0]

        sicd.GeoData = GeoDataType(
            SCP=SCPType(
                ECF={"X": ecf[0], "Y": ecf[1], "Z": ecf[2]},
                LLH={"Lat": llh[0], "Lon": llh[1], "HAE": llh[2]},
            )
        )

        # Grid (Row = u / Line, Col = r / Sample)
        row_param = DirParamType(
            SS=payload.line_spacing,
            ImpRespBW=payload.bandwidth_u,
            KCtr=0.0,
            DeltaK1=-payload.bandwidth_u / 2.0,
            DeltaK2=payload.bandwidth_u / 2.0,
        )
        col_param = DirParamType(
            SS=payload.sample_spacing,
            ImpRespBW=payload.bandwidth_r,
            KCtr=0.0,
            DeltaK1=-payload.bandwidth_r / 2.0,
            DeltaK2=payload.bandwidth_r / 2.0,
        )

        sicd.Grid = GridType(
            ImagePlane="GROUND",
            Type="PLANE",
            Row=row_param,
            Col=col_param,
        )

        # Image Formation
        tx_freq_proc = TxFrequencyProcType(
            MinProc=cphd_meta.global_fx_min,
            MaxProc=cphd_meta.global_fx_max,
        )
        sicd.ImageFormation = ImageFormationType(
            ImageFormAlgo="OTHER",
            TxFrequencyProc=tx_freq_proc,
        )

        # Collection Info & Radar Collection
        radar_mode = cphd_meta.radar_mode or "SPOTLIGHT"
        sicd.CollectionInfo = CollectionInfoType(
            CollectorName="CZTPFA",
            CoreName="CZTPFA_Output",
            RadarMode={"ModeType": radar_mode},
        )
        sicd.RadarCollection = RadarCollectionType(
            TxPolarization=payload.tx_pol if payload.tx_pol != "UNKNOWN" else "OTHER",
        )

        # Position (Dummy polynomial for uncompensated raw PFA)
        sicd.Position = PositionType(
            ARPPoly={"X": [0.0], "Y": [0.0], "Z": [0.0]}
        )

        # Timeline
        if cphd_meta.collection_start:
            try:
                coll_start = datetime.datetime.fromisoformat(
                    cphd_meta.collection_start.replace("Z", "+00:00")
                )
            except Exception:
                coll_start = datetime.datetime.now()
        else:
            coll_start = datetime.datetime.now()

        sicd.Timeline = TimelineType(CollectStart=coll_start)

        # Create destination directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

        with SICDWriter(output_path, sicd) as writer:
            writer.write_chip(img_arr, start_indices=(0, 0))

        return output_path
