# diffpfa

`diffpfa` is a PyTorch-based Polar Format Algorithm (PFA) processor that forms complex synthetic aperture radar (SAR) images from Compensated Phase History Data (CPHD) files.

## Goals
- Perform PFA on CPHD data to form Unfocused SICD (SICD-U) images in NITF format.
- Support multi-channel CPHD datasets (e.g., stepped-chirp subbands, full polarimetric data).
- Uses a CZT (range) NUFFT (cross-range) implementation.

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
The pipeline is split between I/O orchestration and mathematical processing to separate concerns and manage system resources efficiently.

### 1. I/O and Orchestration (`IFAProcessor`)
- **Parallel Read**: Uses threaded execution to read independent CPHD channels.
- **Dynamic Geometry**: Checks dataset orientation against the Center of Aperture where range and cross-range axes need to be transpose for czt implementation.
- **Direct Formatting**: Utilizes `sarkit` for reading CPHDs and writing compliant SICD-U.

### 2. Mathematics and Image Formation (`pfa_per_polar`)
- **Differentiable Backend**: Operations are written in PyTorch, retaining differentiability with respect to the raw signal tensor.
- **CZT-NUFFT Processing**: Applies a 1D-CZT along range and a 1D Type-1 NUFFT along cross-range to correct aperture curvature.
- **RVP Deskew**: Dynamically corrects Residual Video Phase artifacts present in stretch-processed data, while safely bypassing the step if the data was processed with matched filters.
- **In-Place Accumulation**: Disjoint subbands are geometrically projected onto a shared Cartesian K-space grid. Incoming channels are folded sequentially into an accumulator and discarded from RAM immediately, enforcing a minimal memory footprint. Coherently summing channels prior to the 2D IFFT and spatial deconvolution reduces total computational overhead.

## Validation and Testing
The `simulation/` directory contains tools (`benchmark_kspace.py`, `simulate_stepped_chirp.py`) for generating synthetic data. This enables direct validation of subband phase alignment and coherent combination, ensuring expected theoretical improvements in spatial resolution (e.g., main lobe FWHM halving when combining two contiguous subbands). Basic functionality and schema compliance are verified via the `tests/` directory.

## Notes on Optimization
- **Memory Handling**: Sequential processing of memory-intensive datasets necessitates careful management of PyTorch's Caching Allocator. Deletion of intermediate tensors and strategic use of `torch.cuda.empty_cache()` are employed to prevent memory fragmentation on wideband, multi-channel collections.
- **Separating I/O**: The pipeline clearly delineates `setup_and_read_time` from `proc_time` to isolate disk performance from GPU throughput, providing actionable statistics when scaling to large datasets.
