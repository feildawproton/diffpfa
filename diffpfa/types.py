import numpy as np
import lxml.etree as ET
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ImageAreaBounds:
    x1: float
    y1: float
    x2: float
    y2: float
    polygon: Optional[List[Tuple[float, float]]]

@dataclass
class CPHDMetadata:
    domain_type: str
    sgn: int
    global_fx_min: float
    global_fx_max: float
    iarp_ecf: np.ndarray
    uIAX: np.ndarray
    uIAY: np.ndarray
    ref_ch_id: str
    image_area: Optional[ImageAreaBounds]
    extended_area: Optional[ImageAreaBounds]
    collection_start: Optional[str]
    radar_mode: Optional[str]
    classification: Optional[str]
    srp_ecf: np.ndarray
    arp_pos_coa: Optional[np.ndarray]
    arp_vel_coa: Optional[np.ndarray]
    side_of_track: str
    line_spacing: Optional[float]
    sample_spacing: Optional[float]
    raw_meta: ET.Element
