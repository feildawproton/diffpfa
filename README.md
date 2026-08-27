# diffpfa

`diffpfa` is a PyTorch-based Polar Format Algorithm (PFA) processor that forms complex synthetic aperture radar (SAR) images from Compensated Phase History Data (CPHD) files.

## Goals
- Perform PFA on CPHD data to form SICD (SICD-U) images in NITF format.
- Support multi-channel CPHD datasets (e.g., stepped-chirp subbands, full polarimetric data).
- Uses a CZT (range) NUFFT (cross-range) implementation.
- Must be differentiable WRT the signal.

## Usage
The primary entry point is `run_pfa.py`, designed to batch process directories of CPHD files while tracking I/O and compute performance.

```bash
python run_pfa.py --input_dir /path/to/cphds --output_dir /path/to/output_sicds
```

To convert the generated SICD files to 8-bit PNG images (using a density remapper), run:
```bash
python visualize_results.py /path/to/output_sicds
```

## Architecture
The pipeline is split between I/O + state and processing to separate concerns and make the algos more tractable.

### 1. I/O and State (`IFAProcessor`)
- **Parallel Read**: Uses threaded execution to read independent CPHD channels.
- **Dynamic Geometry**: Checks dataset orientation against the Center of Aperture where range and cross-range axes need to be transposed for this czt implementation's assumptions. Supports both `GROUND` and `SLANT` image planes, correctly handling `SideOfTrack` mirroring for Left/Right looking collections.
- **NGA-Standard Oversampling**: Forms perfectly orthogonal K-space grids in the `SLANT` plane (default) with native Nyquist calculations, cleanly oversampled at 1.25x to exactly match NGA-certified "gold" implementations (e.g., Umbra native processing).

### 2. Mathematics and Image Formation (`pfa_per_polar`)
- **Differentiable Backend**: Operations are written in PyTorch, retaining differentiability with respect to the raw signal tensor.
- **CZT-NUFFT Processing**: Applies a 1D-CZT along range and a 1D Type-1 NUFFT along cross-range to correct aperture curvature.
- **RVP Deskew**: Corrects Residual Video Phase artifacts present in stretch-processed data, while bypassing the step if the data was processed with downconversion + matched filter.
- **In-Place Accumulation**: Disjoint subbands are projected onto a shared Cartesian K-space grid. Incoming channels are processed sequentially into an accumulator and discarded from RAM immediately, enforcing a minimal memory footprint. Coherently summing channels prior to the 2D IFFT and spatial deconvolution reduces total computational overhead (repeated 2D IFFTs for each channel).

## Validation and Testing
The `simulation/` directory contains tools (`benchmark_kspace.py`, `simulate_stepped_chirp.py`) for generating synthetic data. This enables direct testing of subband phase alignment and coherent combination and inspecting improvements in spatial resolution (e.g., main lobe FWHM halving when combining two contiguous subbands).

## Notes on Optimization
- **Memory Handling**: Need to manage PyTorch's Caching Allocator. Deletion of intermediate tensors between channels and careful use of `torch.cuda.empty_cache()` are used to prevent memory fragmentation on wideband, multi-channel collections.
- **Separating I/O**: The pipeline delineates `setup_and_read_time` from `proc_time` and 'write_time' to isolate disk performance from CPU+GPU throughput; useful for determining where the juice is worth the squeeze (depends on deployed system).
