# CPHD 1.1.0 Schema Reference Notes

This document serves as a reference for key CPHD metadata elements as defined in the official NGA standard (`NGA.STND.0068-1_1.1.0_CPHD_DIDD_FINAL.pdf`). It documents the explicit paths and structures required for a CPHD file to be compliant.

## 1. `Global` Node
Global parameters that apply to the entire dataset.
- **Path:** `Global`
- **DomainType** (`Global / DomainType`): Defines the data domain (e.g., `FX` for frequency/cross-range, `TOA` for time of arrival).
- **SGN** (`Global / SGN`): Sign of the phase. Determines the direction of the FFT.
- **FxBand**: Bounding fast-time frequencies for all channels.
  - `Global / FxBand / FxMin`
  - `Global / FxBand / FxMax`
- **Timeline**: Collection temporal boundaries.
  - `Global / Timeline / CollectionStart`: UTC timestamp of the collection start.

## 2. `Channel` Node
Contains parameters specific to individual data channels (polarizations, sub-bands, etc.).
- **Path:** `Channel / Parameters`
- **Identifier** (`Channel / Parameters / Identifier`): Unique string matching the data array.
- **FxC** (`Channel / Parameters / FxC`): Center frequency for the specific channel.
- **FxBW** (`Channel / Parameters / FxBW`): Bandwidth for the specific channel.
- **Polarization**: Transmit and Receive polarizations.
  - `Channel / Parameters / Polarization / TxPol` (e.g., H, V, RHC, LHC)
  - `Channel / Parameters / Polarization / RcvPol`

## 3. `SceneCoordinates`
Parameters that define the geographic coordinates for the reference surface and the image areas.
- **Path:** `SceneCoordinates`
- **IARP** (Image Area Reference Point):
  - `SceneCoordinates / IARP / ECF` (Type `XYZ`): `X`, `Y`, `Z` coordinates in Earth-Centered Fixed (ECF) meters.
- **ReferenceSurface**: Defines the projection plane.
  - `SceneCoordinates / ReferenceSurface / Planar / uIAX` (Type `XYZ`): Unit vector for Image Area X.
  - `SceneCoordinates / ReferenceSurface / Planar / uIAY` (Type `XYZ`): Unit vector for Image Area Y.

### Area Boundaries
- **ImageArea**: Defined by a rectangle aligned with Image Area coordinates (IAX, IAY). May be reduced by an optional polygon.
  - `SceneCoordinates / ImageArea / X1Y1` (Type `XY`): Corner `(IA_X1, IA_Y1)`.
  - `SceneCoordinates / ImageArea / X2Y2` (Type `XY`): Corner `(IA_X2, IA_Y2)`.
  - `SceneCoordinates / ImageArea / Polygon / Vertex` (Type `XY`): 3 or more vertices in clockwise order. (Contains `X` and `Y` children).
- **ExtendedArea**: Similar to ImageArea but defines the maximum valid bounds of the data including padding/over-collect.
  - `SceneCoordinates / ExtendedArea / X1Y1`
  - `SceneCoordinates / ExtendedArea / X2Y2`
  - `SceneCoordinates / ExtendedArea / Polygon / Vertex`
- **ImageAreaCornerPoints** (IACP): Approximate geographic locations of the four corners (not for rigorous analytic use).
  - `SceneCoordinates / ImageAreaCornerPoints / IACP` (Type `LL`): Array of 4 points containing `Lat` and `Lon`.

## 4. `ReferenceGeometry`
Contains the parameters that define the reference geometry vectors for the CPHD. Crucial for PFA and phase deskewing.
- **Path:** `ReferenceGeometry`
- **SRP** (Staring Reference Point):
  - `ReferenceGeometry / SRP / ECF` (Type `XYZ`): `X`, `Y`, `Z` in ECF meters.
- **Monostatic**: Geometry for a monostatic collection (Tx and Rcv from same platform).
  - `ReferenceGeometry / Monostatic / ARPPos` (Type `XYZPoly` or `XYZ`): Aperture Reference Point Position.
  - `ReferenceGeometry / Monostatic / ARPVel` (Type `XYZPoly` or `XYZ`): Aperture Reference Point Velocity.

## 5. `PVP` (Per-Vector Parameters)
Binary data fields stored alongside the signal data for every pulse.
- **TxPos / RcvPos**: Transmit and receive antenna phase center ECF positions (meters).
- **TxFMRate** (gamma): Chirp rate (Hz/s) for deramped data.
- **SC0** (Starting Frequency): Frequency of the first sample (Hz).
- **SCSS** (Frequency Step): Frequency step between samples (Hz).

---

## Sarkit Implementation Notes
When using `sarkit.cphd.XmlHelper.load()`, you must point it directly at the specific node implementing a transcodable type (e.g., `XYZ`, `XY`).
- **SRP Loading:** Use `./{*}ReferenceGeometry/{*}SRP/{*}ECF`
- **Polygon Loading:** Iterate over the `Vertex` nodes (`./{*}Polygon/{*}Vertex`), as `Polygon` itself is a container type.
