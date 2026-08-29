# diffpfa

`diffpfa` is a PyTorch-based Polar Format Algorithm (PFA) processor that forms complex synthetic aperture radar (SAR) images from Compensated Phase History Data (CPHD) files.

## Goals
- Perform PFA on CPHD data to form strictly standard-compliant SICD (SICD-U) images in NITF format.
- Support multi-channel CPHD datasets (e.g., stepped-chirp subbands, inter-pulse delay bursts, full polarimetric data).
- Uses an exact 1D CZT (range) and 1D Type-1 NUFFT (cross-range) implementation.
- Must be differentiable with respect to the input signal tensor.

## Usage

### 1. Batch Processing
The primary entry point is `run_pfa.py`, designed to batch process directories of CPHD files while tracking I/O and compute performance:

```bash
python run_pfa.py --input_dir /path/to/cphds --output_dir /path/to/output_sicds
```

### 2. PNG Conversion
To convert the generated SICD NITF files to 8-bit density-remapped PNG images:
```bash
python tools/convert2png.py /path/to/output_sicds/image.nitf /path/to/output.png
```

### 3. Simulation & Validation
To run the physical stepped-chirp simulation suite with inter-step time delays ($\Delta \tau$), platform motion, and downconversion:
```bash
python simulation/stepped_chirp_simulation.py
```

### 4. Running Tests
To execute the automated unit and schema validation test suite:
```bash
pytest tests/ -v
```

## Architecture

The pipeline is split between I/O + state and processing to separate concerns and maximize throughput.

### 1. I/O and State (`IFAProcessor`)
- **Parallel Read**: Uses threaded execution to read independent CPHD channels asynchronously.
- **Dynamic Geometry**: Supports both `SLANT` (default) and `GROUND` image planes, correctly handling `SideOfTrack` mirroring for Left/Right looking collections and rotating basis vectors when Line-of-Sight is aligned with CPHD planar axes.
- **NGA-Standard Oversampling**: Forms orthogonal K-space grids with native Nyquist calculations, cleanly oversampled at 1.25x to match NGA-certified "gold" implementations.
- **Full Geolocation Metadata**: Computes exact NGA standard analytical formulas for all 9 SCPCOA angles (`DopplerConeAng`, `GrazeAng`, `IncidenceAng`, `TwistAng`, `SlopeAng`, `AzimAng`, `LayoverAng`, `SlantRange`, `GroundRange`).
- **5th-Order Kinematics (`ARPPoly`)**: Fits degree-5 polynomials to $(TxPos + RcvPos)/2$ across the dwell time, extracting exact analytical velocity and acceleration vectors at the Center of Aperture ($t_{COA}$).
- **Space-Variant `<PFA>` Metadata**: Generates `<FPN>`, `<IPN>`, `<PolarAngRefTime>`, `<PolarAngPoly>`, `<SpatialFreqSFPoly>`, and frequency bounds (`Krg1/2`, `Kaz1/2`) for full 2D space-variant Impulse Response (IPR) description.

### 2. Mathematics and Image Formation (`pfa_per_polar`)
- **Differentiable Backend**: Operations are written in PyTorch, retaining differentiability with respect to the raw signal tensor.
- **CZT-NUFFT Processing**: Applies a 1D-CZT along range and a 1D Type-1 NUFFT (Kaiser-Bessel kernel with closed-form continuous Fourier deconvolution) along cross-range to correct aperture curvature.
- **RVP Deskew**: Corrects Residual Video Phase artifacts present in stretch-processed data, while bypassing the step if the data was processed with downconversion + matched filter.
- **In-Place Accumulation**: Disjoint subbands are projected onto a shared Cartesian K-space grid and accumulated prior to the 2D IFFT. Applies differential carrier phase rotations $\phi_{corr} = -2\pi(f_{c,global} - f_{xc})\tau$ to preserve coherence across moving platform subband bursts.

## Notes on Optimization
- **Memory Handling**: Deletion of intermediate tensors between channels and explicit use of `torch.cuda.empty_cache()` are utilized to prevent memory fragmentation on wideband, multi-channel collections.
- **Separating I/O**: The pipeline delineates `setup_and_read_time` from `proc_time` and `write_time` to isolate disk performance from GPU throughput.
