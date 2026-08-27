# CPHD (Compensated Phase History Data) Standard Summary

## 1. Core Data Structures
A CPHD file is structured sequentially into four primary data blocks (in a "flat file" format):
1. **File Header (Required):** UTF-8 text block that identifies the file type (CPHD) and version. Contains Key-Value Pairs (KVPs) defining the byte sizes and offsets of the other data blocks.
2. **XML Block (Required):** UTF-8 text containing the XML instance that describes the collection parameters, metadata, and the structure of the binary data blocks.
3. **Support Block (Optional):** Binary-formatted data containing support arrays, such as 2D arrays of scene surface heights or sampled antenna patterns.
4. **PVP (Per Vector Parameters) Block (Required):** Binary data containing 1D arrays of parameter sets for each signal vector.
5. **Signal Block (Required):** Binary data containing the 2D arrays of phase history data samples (complex-valued, formatted as CI2, CI4, or CF8). Signal arrays can optionally be compressed.

**Data Channels:**
A product contains one or more data channels, each comprising a signal array and a corresponding PVP array. Signal vectors ("slow time") are ordered by increasing time. Each vector has been motion-compensated to a fixed point in the scene known as the **Stabilization Reference Point (SRP)**. The signals are represented in either the **FX (transmit frequency)** or **TOA (Time of Arrival)** domains.

## 2. Coordinate Systems
- **Earth Centered Fixed (ECF):** A 3D Cartesian coordinate system (X, Y, Z) in meters. Used for specifying positions (e.g., Antenna Phase Centers, SRP) and velocities.
- **Geodetic Coordinates (LLH):** Specified using the WGS_84 Earth Model (Latitude, Longitude, Height Above Ellipsoid - HAE).
- **Image Area Reference Point (IARP):** The origin of the Image Area Coordinate system. It must be located on the image reference surface.
- **Image Reference Surface:** Defines the geolocated surface for the imaged scene. Can be:
  - **PLANAR:** A plane containing the IARP, defined by unit vectors `uIAX` and `uIAY`.
  - **HAE:** A surface of constant Height Above the Ellipsoid containing the IARP.
- **Image Area Coordinates (IAC):** A 3D coordinate system (IAX, IAY, IAZ) relative to the IARP, expressed in meters. IAX and IAY define positions along the reference surface, while IAZ is the normal distance from the surface.
- **Antenna Coordinate Frame (ACF):** A right-handed frame (ACX, ACY, ACZ) fixed relative to the physical antenna, with +ACZ pointing to the "front" (mechanical boresight).

## 3. Key Metadata XML Schemas (Relevant to Polar Format Algorithm - PFA)

### `CPHD / Global`
Contains parameters applicable to all data channels and signal arrays.
- `DomainType`: Indicates the sample dimension domain ("FX" or "TOA").
- `SGN`: Phase sign (+1 or -1) applied during compensation processing.
- `Timeline`: Absolute `CollectionStart` time (UTC) and relative limits (`TxTime1`, `TxTime2`).
- `FxBand` / `TOASwath`: The global minimum and maximum bounds for the FX and TOA domains across all vectors in the product.

### `CPHD / Channel`
Describes the specific properties of each data channel.
- `RefChId`: Identifies the reference channel for the product.
- `Parameters` (repeated per channel):
  - `RefVectorIndex`: The index of the reference vector for this channel (often the center vector).
  - `Polarization`: Transmit (`TxPol`) and receive (`RcvPol`) polarizations.
  - `FxC`, `FxBW`: Center frequency and bandwidth of the saved FX signal.
  - `TOASaved`: Full resolution TOA swath.
  - `DwellTimes`: References to Center of Dwell (COD) and Dwell Time polynomials over the image area.
  - `Antenna`, `TxRcv`: Identifiers linking to the specific Antenna Phase Centers (APC), antenna patterns, and transmit/receive waveforms used.

### `CPHD / PVP` (Per Vector Parameters)
Defines the structure of the binary PVP block. For each vector in the signal array, a set of parameters is provided describing the precise collection geometry and signal timing.
- `TxTime`, `RcvTime`: Transmit time and receive Time of Arrival (TOA) for the SRP echo.
- `TxPos`, `TxVel`, `RcvPos`, `RcvVel`: Transmit and Receive Antenna Phase Center (APC) positions and velocities in ECF coordinates.
- `SRPPos`: The Stabilization Reference Point (SRP) position in ECF coordinates for the given vector.
- **Signal Model Micro Parameters:** `aFDOP` (Doppler shift/time dilation), `aFRR1` (linear phase), and `aFRR2` (quadratic phase). These dimensionless scale factors account for platform motion and time dilation differences relative to the SRP echo.
- **Signal Extent:** `FX1`, `FX2`, `TOA1`, `TOA2` define the valid frequency and TOA limits retained in the signal vector.
- **Sample Coordinates:** `SC0` (starting coordinate value) and `SCSS` (sample spacing) for the samples in the vector.

### `CPHD / ReferenceGeometry`
Provides a simplified summary of the imaging geometry computed for the reference vector (`v_CH_REF`) of the reference channel. Useful for direct search and discovery.
- `SRP`: SRP position in ECF and IAC coordinates, plus reference times (`ReferenceTime`, `SRPCODTime`, `SRPDwellTime`).
- `Monostatic` (or `Bistatic`):
  - `ARPPos`, `ARPVel`: Aperture Reference Point (ARP) position and velocity.
  - `SideOfTrack`: Look direction ("L" or "R").
  - `SlantRange`, `GroundRange`.
  - **Angles:** `DopplerConeAngle`, `GrazeAngle`, `IncidenceAngle`, `AzimuthAngle`, `TwistAngle`, `SlopeAngle`, and `LayoverAngle`.
