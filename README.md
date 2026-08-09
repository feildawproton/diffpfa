# diffpfa

`diffpfa` is a PyTorch-based Polar Format Algorithm (PFA) processor that forms complex synthetic aperture radar (SAR) images from **Compensated Phase History Data (CPHD)** files (data that has been motion compensated to the Scene Reference Point and match filtered).

## Goals
- Perform PFA on CPHD data to form Uncompensated SICD (SICD-U) images in NITF format.
- Support modern multi-channel CPHD datasets (e.g. step-chirp sub-bands, polarimetric).
- Output must be purely uncompensated (no side-lobe suppression, no automatic refocusing/autofocus applied by default).
- Provide multiple Image Formation Processes (IFPs): CZT for speed, NUFFT for accurate wide-aperture geometry, and Hybrid (Range CZT + Cross-Range NUFFT).

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
- **`mode="nufft"`**: 2D Type-1 NUFFT. Handles wide-aperture curvature by placing non-uniform 2D frequencies onto a Cartesian grid. Memory intensive.
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

## Validation and Simulation
To validate the mathematical correctness of sub-band and polarimetric coherent combination, the `simulation/simulate_stepped_chirp.py` script provides a full synthetic testbench.
- It generates a coherent, dual-band, polarimetric point target geometry (at the Scene Reference Point) mapped to cycles/meter space.
- It intentionally injects synthetic phase offsets between the bands to test alignment correction.
- Output IPRs (Impulse Responses) are plotted to prove that the coherent combination of two 250 MHz subbands yields an exact halving of the spatial Main Lobe FWHM compared to the individual uncombined subbands.
- The output `debug_save_channels` SICDs can be visualized using the `visualize_sicd.py` root script.

## Implementation Notes
- **CUDA/PyTorch Determinism:** The engine supports PyTorch GPU acceleration with `torch.compile()` Dynamo support for efficient processing loops. GPU non-determinism (`index_put_`) has been addressed.
- **Stepped-Chirp Subband Combination:** The engine allocates a **Global K-space Grid Envelope** for all sub-bands, preserving spatial frequency carrier phase when heterodyning the IFFT to baseband.
- **Native Packaging:** Standard Python packaging (`pyproject.toml`) and `pytest` harnesses support integration into existing workflows.

## Notes
Output SICD byte size discrepancies between Sarkit and Sarpy are expected. Both write exact RE32F_IM32F pixels, but differences in XML padding and NITF segment block layout cause differing byte footprints.
