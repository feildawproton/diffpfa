from diffpfa.io.base import (
    BaseCPHDReader,
    BaseSICDWriter,
    CPHDChannelData,
    CPHDMetadata,
    ImageAreaBounds,
    SICDImagePayload,
)
from diffpfa.io.sarpy_cphd import SarpyCPHDReader
from diffpfa.io.sarpy_sicd import SarpySICDWriter

def _get_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    
    try:
        import sarkit
        return "sarkit"
    except ImportError:
        pass

    try:
        import sarpy
        return "sarpy"
    except ImportError:
        pass
    
    raise ImportError("No backend available. Please install 'sarkit' or 'sarpy'.")

def CPHDReader(file_path: str, backend: str = "auto") -> BaseCPHDReader:
    """Factory for CPHD Readers."""
    selected_backend = _get_backend(backend)
    
    if selected_backend == "sarkit":
        from diffpfa.io.sarkit_cphd import SarkitCPHDReader
        return SarkitCPHDReader(file_path)
    elif selected_backend == "sarpy":
        return SarpyCPHDReader(file_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def SICDWriter(backend: str = "auto") -> BaseSICDWriter:
    """Factory for SICD Writers."""
    selected_backend = _get_backend(backend)
    
    if selected_backend == "sarkit":
        from diffpfa.io.sarkit_sicd import SarkitSICDWriter
        return SarkitSICDWriter()
    elif selected_backend == "sarpy":
        return SarpySICDWriter()
    else:
        raise ValueError(f"Unknown backend: {backend}")

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
