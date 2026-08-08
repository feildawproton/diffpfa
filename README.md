# diffpfa

`diffpfa` is a PyTorch-based Polar Format Algorithm (PFA) processor that forms complex synthetic aperture radar (SAR) images from **Compensated Phase History Data (CPHD)** files (data that has been motion compensated to the Scene Reference Point and match filtered).

## Goals
- Perform PFA on CPHD data to form Uncompensated SICD (SICD-U) images in NITF format.
- Support modern multi-channel CPHD datasets (e.g. step-chirp sub-bands, polarimetric).
- Output must be purely uncompensated (no side-lobe suppression, no automatic refocusing/autofocus applied by default).
- Provide multiple Image Formation Processes (IFPs): CZT for speed, NUFFT for perfect wide-aperture geometry, and Hybrid (Range CZT + Cross-Range NUFFT).

## Constraints
- **Differentiability**: The core algorithms MUST be implemented in PyTorch and remain differentiable with respect to the raw signal tensor. This enables downstream gradient-based Machine Learning Auto-Focus workflows.
- **I/O Abstraction**: The core mathematical engine must remain agnostic to the CPHD reader and SICD writer backends.
- **Sarkit Priority**: The project supports both `sarpy` and `sarkit` backends. However, `sarpy` is being deprecated; if both are available, the system will automatically prioritize `sarkit`.
- **Memory Safety**: PyTorch memory management must be explicitly configured by the caller (batch sizes) to prevent VRAM Out-of-Memory (OOM) errors on large datasets. Default fallbacks for batch sizes should not be provided to enforce safety.

## Approach
### 1. I/O Abstraction
- Defined abstract interfaces (`CPHDReader`, `SICDWriter`) that parse data into generalized `CPHDMetadata` and `SICDImagePayload` dataclasses.
- Implemented concrete subclasses for `sarpy` and `sarkit`.

### 2. Geometric Mapping
- Phase math uses rigorous 3D slant-range distances (`torch.linalg.norm(P_vecs)`) from the Phase Center to the Scene Reference Point to prevent spatial scaling and squint errors.
- Image planes can be mapped to Earth-tangent `Ground` (using metadata uIAX/uIAY) or true perspective `Slant` (using dynamic Line-of-Sight and Velocity vectors at the Center of Aperture).

### 3. IFP Formation Modes
- **`mode="nufft"`**: Exact 2D Type-1 NUFFT. Solves wide-aperture curvature flawlessly by accurately placing 2D frequencies on a true Cartesian grid. Memory intensive.
- **`mode="hybrid"`**: Fast 1D-CZT along Range (linear projection), followed by 1D Type-1 NUFFT along Cross-Range to correct aperture curvature. (Padded internally to highly composite numbers `next_fast_len` to avoid Blustein's FFT workspace allocation spikes).
- **`mode="czt"`**: Standard 2D separable CZT. Extremely fast but introduces quadratic phase errors at the edges for wide apertures. Mitigated via dynamic `num_subpatches`.

### 4. Step-Chirp Recombination
- Disjoint sub-bands are geometrically projected onto a shared absolute Cartesian grid.
- Alignment algorithms default to `False`. The engine trusts hardware coherence to natively stack orthogonal sub-bands without forcing noisy cross-correlation.

## Tests
The testing suite has been moved into the `tests/` directory:
- `compare_modes.py`: End-to-end integration test that runs the CPHD through `nufft`, `hybrid`, and `czt` modes, saves them to SICD-U using `sarkit`, and generates phase difference plots.
- `test_grid.py`: Validates Cartesian mapping equations.
- `test_mem.py`: Audits VRAM utilization and peak tracking across reader backends.
- `test_metadata.py`: Validates generic CPHD metadata extraction matches `sarpy`.
- `test_pfa.py`: Validates step-chirp alignment logic.
- `test_txpos.py`: Validates PVP arrays (e.g. TXPos, FX_Min).

## TODOs
- **Speed & Bottlenecks**: Profile the GPU execution to find remaining bottlenecks. Investigate enabling `torch.compile()` Dynamo JIT-compilation for Triton kernels.
- **Multi-Channel**: Extend end-to-end testing to validate full polarimetric (e.g., HH, VV, HV) stacking.
- **Size Discrepancy**: Investigate why `sarkit` SICD output sizes slightly differ from `sarpy` SICD outputs in byte footprint.
- **NUFFT Oversampling**: Remove the hardcoded oversample factor (1.5) in the NUFFT algorithm; derive output image grid size directly using `LineSpacing`, `NumLines`, `SampleSpacing`, and `NumSamples` metadata where available.
- **Subpatching**: Address the `ku_center` error natively in the algorithm implementation to reduce strict reliance on subpatch scaling.
