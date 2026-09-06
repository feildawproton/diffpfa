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

### 4. Differentiable Tensor Usage
For adversarial attacks, downstream optimization, or learned phase history estimation, use the tensor-native entry point to compute gradients back to the signal tensor:

```python
import torch
from diffpfa import run_diffpfa_tensor

# Input signal with autograd tracking
signal = torch.randn(N_pulses, N_samples, dtype=torch.complex64, device="cuda", requires_grad=True)

# Run differentiable PFA directly on GPU
img_tensor, bw_r, bw_u, N_r, N_u, is_rotated = run_diffpfa_tensor(
    signal=signal,
    pvp=pvp_dict,
    cphd_meta=cphd_metadata,
    fxc=center_freq_hz,
    L=image_extent_meters,
    device="cuda"
)

# Backpropagate loss w.r.t. raw signal
loss = torch.sum(torch.abs(img_tensor)**2)
loss.backward()
print("Signal gradient norm:", signal.grad.norm().item())
```

### 5. Running Tests
To execute the automated unit and schema validation test suite:
```bash
pytest tests/ -v
```

## Architecture

The pipeline is split between I/O + state and processing to separate concerns and maximize throughput.

### 1. I/O and State (`IFAProcessor`)
- **Parallel Read**: Uses threaded execution to read independent CPHD channels asynchronously.
- **Dynamic Geometry & Slant Framing**: Supports both `SLANT` (default) and `GROUND` image planes. For slant plane imaging, projects ground `ImageArea` corners onto line-of-sight and cross-range axes with configurable padding (`pad_factor=1.20`), closely matching vendor slant framing (and reproducing modern vendor processor extents within 0.2%–0.4%).
- **NGA-Standard Oversampling**: Forms orthogonal K-space grids with native Nyquist calculations, cleanly oversampled at 1.25x to match NGA-certified "gold" implementations.
- **Full Geolocation Metadata**: Computes exact NGA standard analytical formulas for all 9 SCPCOA angles (`DopplerConeAng`, `GrazeAng`, `IncidenceAng`, `TwistAng`, `SlopeAng`, `AzimAng`, `LayoverAng`, `SlantRange`, `GroundRange`).
- **5th-Order Kinematics (`ARPPoly`)**: Fits degree-5 polynomials to $(TxPos + RcvPos)/2$ across the dwell time, extracting exact analytical velocity and acceleration vectors at the Center of Aperture ($t_{COA}$).
- **Space-Variant `<PFA>` Metadata**: Generates `<FPN>`, `<IPN>`, `<PolarAngRefTime>`, `<PolarAngPoly>`, `<SpatialFreqSFPoly>`, and frequency bounds (`Krg1/2`, `Kaz1/2`) for full 2D space-variant Impulse Response (IPR) description.

### 2. Mathematics and Image Formation (`pfa_per_polar`)
- **Differentiable Backend**: Operations are written in PyTorch using out-of-place accumulations, retaining full end-to-end differentiability with respect to the raw signal tensor.
- **CZT-NUFFT Processing**: Applies a 1D-CZT along range and a 1D Type-1 NUFFT (Kaiser-Bessel kernel with closed-form continuous Fourier deconvolution) along cross-range to correct aperture curvature. Chirp phases are evaluated in double precision (`float64`) to prevent phase truncation errors.
- **Normalized Multi-Band Resampling**: CZT resampler normalizes by $\|k_{\text{step}}\| \cdot r_{\text{step}}$ to ensure equal weighting across stepped-chirp subbands regardless of fast-time sample spacing (`SCSS`).
- **SRP-Referenced Coherent Accumulation**: Subbands are already SRP-referenced and RF-frequency labeled per CPHD DIDD §1.4, enabling coherent Cartesian accumulation without inter-burst carrier remodulation.
- **Arbitrary Area Centring**: Applies a linear K-space phase shift to accurately place the Scene Reference Point (SRP) at $(0, 0)$ when framing asymmetric or offset image regions.

## Notes on Optimization
- **Memory Handling**: Deletion of intermediate tensors between channels and explicit use of `torch.cuda.empty_cache()` are utilized to prevent memory fragmentation on wideband, multi-channel collections.
- **Separating I/O**: The pipeline delineates `setup_and_read_time` from `proc_time` and `write_time` to isolate disk performance from GPU throughput.
