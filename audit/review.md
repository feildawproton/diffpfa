# diffpfa review notes

Review date: 2026-08-11

This is an issue inventory, not a claim that every suspected mathematical issue is a
defect. Items are classified as **confirmed**, **unresolved**, or **documentation** so
that uncertainty is visible. The core image-formation mathematics was not changed as
part of the cleanup that created this file.

## Intended project

`diffpfa` is intended to form uncompensated complex SAR images from compensated CPHD
signal arrays. The numerical engine is written in PyTorch so gradients can propagate
to the input signal. It supports a two-dimensional gridding NUFFT and a hybrid range
CZT/cross-range NUFFT, groups channels by polarization, combines frequency subbands,
and writes SICD NITF through Sarkit.

## Real Umbra file inspected

File: `/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd`

- CPHD 1.1.0, FX domain, `SGN=-1`.
- One channel (`Primary`), 8,671 vectors by 8,639 samples.
- `SignalNormal=true` and every `SIGNAL` PVP is 1.
- `SC0`, `SCSS`, `TxTime`, and `RcvTime` are present.
- `TxFMRate` is absent. Consequently the current automatic RVP branch does not run
  on this file.
- The retained global band is approximately 9.4557 to 9.7443 GHz.

## Confirmed implementation and test problems

### SGN is ignored

The reader parses `Global/SGN`, but no formation code uses it. The current transform
convention appears tailored to `SGN=-1`, which happens to match the Umbra file and the
synthetic fixtures. Both signs need off-center target tests before changing the engine.

For the CPHD model

`phase(f) = SGN * f * delta_TOA` cycles,

and first-order monostatic geometry

`delta_TOA ~= (2/c) * p_hat dot x`,

the effective coordinates used with the implementation's positive-exponent inverse
FFT appear to be `K_effective = -SGN * K_geometry`. This is a derivation to verify,
not yet an implemented conclusion.

### The stepped-frequency simulation is circular

`simulation/simulate_stepped_chirp.py` injects the exact negative of the engine's
RcvTime/frequency correction. It therefore proves only that identical expressions
cancel. It does not independently derive or validate physical inter-channel phase.
The replacement should generate off-center target signals directly from the CPHD
phase model, including SGN and actual Tx/Rcv geometry.

### The alignment control appears dead

`align_subchannels`, `align_phase`, and CLI `--no_align` do not control the analytical
correction. Alignment happens whenever RcvTime is present. If alignment is intended to
be mandatory, these controls are misleading. This does not validate the current
correction formula.

### Analytical alignment assumptions are unchecked

The correction subtracts one channel's RcvTime array from the first channel's array.
It assumes equal vector counts and index-wise correspondence. It also assumes a valid
nonzero channel `FxC`. Neither condition is checked. Interleaved stepped-frequency
channels do not necessarily have corresponding receive times at the same indices.

### `num_subpatches > 1` is broken in CZT-NUFFT

`process_channel_czt_nufft` deletes `F_hz_full` and `dR_full` inside the first patch
iteration. A later patch needs them. The advertised localized wavefront correction
path therefore cannot finish when more than one patch is requested.

### Image-area modes appear falsely advertised

`InscribedRectangle` and `TargetPixelSpacing` have no distinct implementations and
fall through to generic bounds. They are nevertheless advertised by the public
configuration and CLI. Custom pixel spacing is independently available as
`custom_pixel_spacing`.

### Schema test previously did not validate

The schema test calls an API absent from the installed Sarkit
(`NitfMetadata.from_file`) and fails before validation. Its exception handler would
also print schema failure rather than failing the test. A temporary experiment using
`NitfReader` and Sarkit's SICD 1.3.0 XSD showed that the generated test document passes
that schema; that code change was reverted. Schema validity still does not establish
that placeholder geometry is physically correct.

### Empty tests

`tests/test_grid.py`, `tests/test_metadata.py`, and `tests/test_txpos.py` contain no
collected tests. The meaningful test suite is still too weak for the claims in the
README.

### Comparison script was nonfunctional

The script depends on Sarpy and attempts to access a nonexistent `czt` result despite
running only `nufft` and `cztnufft`; it therefore cannot complete as written. A future
comparison should define registration, normalization, and error metrics before
comparing complex pixels.

### Sarkit-only support is not consistently represented

Packaging, factories, CLI, scripts, and the memory benchmark advertise Sarpy despite
the project declaring Sarkit-only support.
`tools/visualize_sicd.py` still uses Sarpy. The tool was outside the approved cleanup
scope and remains unchanged. Decide separately whether visualization is intentionally
allowed to retain Sarpy as an optional utility dependency or should later be converted.

### SICD inspection tool imported a nonexistent project API

`tools/inspect_sicd.py` imports `diffpfa.io.SICDReader`, which does not exist. The tool
was outside the approved cleanup scope and remains unchanged.

### CUDA verification was not completed

The sandboxed full-suite run could not initialize an NVIDIA driver. A request to rerun
the suite with host GPU access was aborted, so no CUDA or full real-Umbra formation
result should be claimed from this review. The sandboxed run reached 6 passing tests,
3 failures, and 3 skips before the schema API correction; two failures were solely the
CUDA driver initialization error. The temporary corrected schema test subsequently
passed by itself. All test-file changes were then reverted. A complete suite result is
still outstanding.

## Unresolved mathematical and signal-model questions

### Automatic residual-video-phase correction

The engine applies `exp(+j*pi*f^2/gamma)` whenever `TxFMRate`, `SC0`, and `SCSS` exist.
The inspected Umbra CPHD has no TxFMRate, so it is unaffected. CPHD signal arrays are
already compensated products, and presence of a field alone has not been shown to
mean that this correction remains outstanding. The trigger, sign, and formula require
a specification-based derivation. Removing the automatic correction is preferable to
silently altering other CPHD products if no valid trigger can be established.

### Bistatic geometry

The current code replaces Tx/Rcv positions with their midpoint and applies the
monostatic `2f/c` factor. This is exact for coincident Tx and Rcv, not for general
bistatic geometry. First-order bistatic spatial frequency depends separately on the
Tx-to-target and Rcv-to-target unit vectors and is proportional to

`(f/c) * (p_hat_tx + p_hat_rcv)`.

Until bistatic support is implemented, the engine should detect materially separated
Tx/Rcv geometry and reject it rather than silently use a midpoint approximation.

### Direction and coordinate conventions

`compute_look_vectors` returns phase-center-to-SRP vectors. Off-center targets are
needed to verify image orientation, SCP shift sign, row/column assignment, and SGN for
both transform modes. A centered point target cannot reveal a mirror or sign error.

### Sampling endpoints

Several places use extent divided by `N`, while other places use `torch.linspace` or
CZT endpoints spanning both limits, which implies division by `N-1`. These conventions
can change pixel coordinates, grid spacing, and phase ramps. They require a single
documented convention and tests at known nonzero positions.

### Gridding and deconvolution

The 2-D NUFFT and hybrid path use related but not identical Kaiser-Bessel
deconvolution expressions. The hybrid path deconvolves only cross-range; the range CZT
resampling has a separate approximation. Absolute scaling, kernel transform,
oversampling, crop parity, and edge behavior have not been validated against a direct
DFT. The existing peak-to-mean synthetic assertion is insufficient.

### Global K-space centering

The run-level K-space envelope is computed using metadata ground-plane vectors even
when `image_plane="Slant"`; channel processing later uses slant vectors. Thus slant
mode can use centers and bounds derived in a different coordinate system.

### Differentiability is not adequately tested

The project requires differentiability with respect to raw signal tensors, but no test
performs backward propagation through either complete formation path. In-place grid
accumulation, preallocated batch assignment, channel mutation, and optional
`torch.compile` should be covered by gradient tests and compared with a direct DFT on
small problems.

### Threaded reader construction is over-assumed

`PFAEngine.run` reconstructs readers as `reader.__class__(file_path)` in worker threads.
This is not part of the `BaseCPHDReader` contract and forced synthetic readers to accept
unused arguments. Parallel channel loading should be an explicit reader capability or
performed by the reader rather than inferred by the engine.

## SICD payload and writer concerns

### Payload fields

- `first_line`, `first_sample`, and `center_freq` are populated but never consumed by
  the writer.
- `channel_id` is not used by `SarkitSICDWriter`, but simulations use it to distinguish
  captured debug payloads. It is therefore not currently dead.
- `uIAX` and `uIAY` are ignored by the writer even though they contain meaningful image
  plane geometry. They should not simply be removed: the writer currently substitutes
  hard-coded ECF axes, which is likely incorrect metadata.

### Writer emits placeholders rather than derived metadata

The writer hard-codes or approximates important fields, including grid unit vectors,
ARP polynomial, collection duration, SCP time, side of track, ranges, angles, and image
corners. `Grid/ImagePlane` is always `GROUND`, including when processing slant imagery.
The image corner calculation treats rows and columns as latitude/longitude offsets and
does not apply the supplied image-plane ECF vectors. `Grid/Row/Sgn` and
`Grid/Col/Sgn` are hard-coded to -1. This output must not be described as rigorously
correct SICD metadata until values are derived and schema/consistency checks pass.

The writer also deletes an existing destination with `os.remove` before writing. This
is operationally destructive and unnecessary if the subsequent open uses `wb`.

## Testing gaps

Add small, independent tests for:

1. Off-center point locations in both axes and for both SGN values.
2. Direct-DFT agreement for NUFFT and CZT-NUFFT, including complex phase.
3. Gradient propagation and finite-difference agreement.
4. Multiple subpatch execution and continuity across patch boundaries.
5. Multi-channel alignment derived from a physical signal model rather than an
   injected inverse implementation formula.
6. Unequal/interleaved channel timing and explicit rejection where unsupported.
7. Monostatic enforcement or exact bistatic geometry.
8. SICD schema validation using the installed Sarkit API plus checks that geometry
   fields match the payload and CPHD.
9. CPU/CUDA numerical agreement with stated tolerances.
10. Output pixel spacing, orientation, target location, resolution, and sidelobes.

The current synthetic tests only require peak magnitude to exceed ten times the image
mean. That can pass with incorrect orientation, spatial scaling, phase, or resolution.

## Documentation corrections to make later

- README claims rigorous 3-D geometry but does not disclose the bistatic midpoint
  approximation.
- README claims rigorous SICD schema compliance although schema validation currently
  fails to run and many physical metadata fields are placeholders.
- README says alignment algorithms default to false while the engine always applies
  its RcvTime correction when available.
- README describes exact stepped-chirp resolution validation, but the simulation is
  circular and its metrics are not assertions.
- README describes both image planes, but run-level bounds/centers and SICD metadata
  are not consistently slant-aware.
- README states batch-size defaults should not be provided, while `PFAConfig` supplies
  defaults for both NUFFT and CZT batching.
- README states PyTorch differentiability as a hard requirement without a gradient
  regression test.
- Performance audit claims should retain their original measurement conditions and
  should not be treated as current regression results. No reproducible benchmark
  currently verifies them.

## Repository hygiene

- Simulation output is ignored by `simulation/output/`, `*.nitf`, and `*.png`; the
  current approximately 517 MB of generated output is untracked.
- LibreOffice lock files and Windows `Zone.Identifier` streams are not generally
  ignored. The ignored simulation output directory happens to cover the observed lock
  file there.
- A PDF `Zone.Identifier` sidecar is tracked and appears to be Windows download
  metadata rather than project content.
- The CPHD specification PDF is a legitimate reference. The extracted
  `cphd_spec_scratch.txt` is redundant but useful for searchable local reference and
  is currently tracked.
