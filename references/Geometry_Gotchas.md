# SAR Geometry and Processing Gotchas

This document captures several critical, easy-to-miss geometry standards required to correctly match "gold" reference DoD implementations (like Umbra's native processing) when forming SICD products.

## 1. Matrix vs. Cartesian Transpose (Row/Col vs X/Y)
When mapping a processed image array to SICD metadata, it is extremely easy to accidentally transpose the image.
- **Cartesian**: `X` is horizontal, `Y` is vertical.
- **Matrix / SICD**: The 1st dimension is `Row` (vertical/Y), the 2nd dimension is `Col` (horizontal/X).
- **Rule**: `Row` maps to the `uIAY` (Range) vector. `Col` maps to the `uIAX` (Azimuth / Cross-Range) vector.

If your image appears X-Y transposed, it means your underlying `numpy` or `torch` array is shaped `(Azimuth, Range)`. It must be explicitly transposed to `(Range, Azimuth)` before being written to the SICD `ImageSegment`.

## 2. Left/Right Flip (Side Of Track)
SAR is collected from an aircraft flying forward (Velocity vector). It can look out the Left or Right side.
To maintain a valid **Right-Handed Coordinate System** (required by SICD/CPHD):
- **Right Looking**: The cross-range (azimuth) vector naturally aligns with the velocity vector.
- **Left Looking**: The cross-range vector must be mathematically negated (`-1`). Failure to do this will result in an image that is perfectly mirrored horizontally.

## 3. Slant Plane vs. Ground Plane (Skewed IPRs)
- **Ground Plane**: Forming an image directly on the Earth's surface mathematically "shears" the native rectangular K-space into a parallelogram. This results in a "tilted" or "X-shaped" Impulse Response (IPR).
- **Slant Plane (Gold Standard)**: Forming an image on the plane defined directly by the Line-of-Sight and Velocity vectors perfectly preserves the orthogonal K-space. The IPR will be a perfect cross-hair (`+`). Most "gold" algorithms process in the Slant Plane to maximize the performance of downstream algorithms like autofocus.

## 4. Oversampling and Pixel Spacing
- The CPHD provides a suggested `LineSpacing` and `SampleSpacing`. These are pre-calculated for the Ground Plane and usually include a built-in oversample factor.
- If you switch to the **Slant Plane**, you must discard the CPHD spacing because slant resolution is substantially different from ground resolution.
- Instead, calculate the Nyquist rate (`1.0 / Bandwidth`) and apply a generic oversample factor (e.g., `1.25x`). Reference algorithms heavily utilize `~1.25x` oversampling to produce smooth, alias-free pixels.
