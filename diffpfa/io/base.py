from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


@dataclass
class CPHDChannelData:
    """Standardized CPHD channel data payload for PFA processing."""
    identifier: str
    tx_pol: str
    rcv_pol: str
    fxc: float
    fxbw: float
    signal: torch.Tensor  # Complex tensor: (N_pulses, N_samples)
    pvp: Dict[str, np.ndarray]  # Per Vector Parameters dictionary (e.g. SRPPos, RcvPos, FX1, FX2, SC0, SCSS)
    domain_type: str = "FX"


@dataclass
class ImageAreaBounds:
    """Image area bounds specified in planar coordinates (u, r) in meters."""
    x1: float
    y1: float
    x2: float
    y2: float
    polygon: Optional[List[Tuple[float, float]]] = None


@dataclass
class CPHDMetadata:
    """Global and Scene coordinate metadata extracted from CPHD."""
    domain_type: str
    sgn: int  # Sign of phase history (-1 or +1)
    global_fx_min: float
    global_fx_max: float
    iarp_ecf: np.ndarray  # Image Activity Reference Point (3,)
    uIAX: np.ndarray      # Image Plane Cross-Range unit vector (3,)
    uIAY: np.ndarray      # Image Plane Range unit vector (3,)
    image_area: Optional[ImageAreaBounds] = None
    extended_area: Optional[ImageAreaBounds] = None
    collection_start: Optional[str] = None
    radar_mode: Optional[str] = None
    raw_meta: Any = None


@dataclass
class SICDImagePayload:
    """Payload for writing an uncompensated SICD (SICD-U) image file."""
    complex_image: torch.Tensor  # 2D Complex image tensor (Lines/CrossRange, Samples/Range)
    tx_pol: str
    rcv_pol: str
    uIAX: np.ndarray
    uIAY: np.ndarray
    iarp_ecf: np.ndarray
    line_spacing: float   # du (meters)
    sample_spacing: float # dr (meters)
    first_line: float     # min u offset (meters)
    first_sample: float   # min r offset (meters)
    center_freq: float    # Hz
    bandwidth_u: float    # cycles/meter
    bandwidth_r: float    # cycles/meter
    channel_id: Optional[str] = None


class BaseCPHDReader(ABC):
    """Abstract interface for CPHD Readers."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def get_metadata(self) -> CPHDMetadata:
        """Extract standardized CPHD metadata."""
        pass

    @abstractmethod
    def get_channel_names(self) -> List[str]:
        """List channel identifiers in the CPHD dataset."""
        pass

    @abstractmethod
    def read_channel(self, channel_id: str) -> CPHDChannelData:
        """Read signal array and PVP metadata for a specific channel into PyTorch format."""
        pass

    @abstractmethod
    def close(self):
        """Close open file handles."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BaseSICDWriter(ABC):
    """Abstract interface for SICD Writers."""

    @abstractmethod
    def write_sicd(
        self,
        output_path: str,
        payload: SICDImagePayload,
        cphd_meta: CPHDMetadata
    ) -> str:
        """Construct SICD metadata and export uncompensated NITF complex image."""
        pass
