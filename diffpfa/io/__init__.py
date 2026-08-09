from diffpfa.io.base import (
    BaseCPHDReader,
    BaseSICDWriter,
    CPHDChannelData,
    CPHDMetadata,
    ImageAreaBounds,
    SICDImagePayload,
)

def CPHDReader(file_path: str, backend: str = "auto") -> BaseCPHDReader:
    """Factory for CPHD Readers."""
    from diffpfa.io.sarkit_cphd import SarkitCPHDReader
    return SarkitCPHDReader(file_path)

def SICDWriter(backend: str = "auto") -> BaseSICDWriter:
    """Factory for SICD Writers."""
    from diffpfa.io.sarkit_sicd import SarkitSICDWriter
    return SarkitSICDWriter()

__all__ = [
    "BaseCPHDReader",
    "BaseSICDWriter",
    "CPHDChannelData",
    "CPHDMetadata",
    "ImageAreaBounds",
    "SICDImagePayload",
    "CPHDReader",
    "SICDWriter",
]
