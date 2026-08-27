# Sensor Independent Complex Data (SICD) Standard Summary

## 1. Core Data Structures
A SICD product consists of a pair of data components:
1. **Complex Image Pixel Data**: A single two-dimensional array of complex numbers representing radar reflectivity. Stored as `NumRows` by `NumCols`. Supported formats (`PixelType`):
   - `RE32F_IM32F`: Real and imaginary components as 32-bit floating point (8 bytes/pixel).
   - `RE16I_IM16I`: Real and imaginary components as 16-bit signed integer (4 bytes/pixel).
   - `AMP8I_PHS8I`: Amplitude and phase components as 8-bit unsigned integer (2 bytes/pixel).
2. **Metadata**: Expressed in XML, containing product identification parameters and SAR science parameters needed for downstream exploitation.

## 2. NITF File Format Basics
SICD products are packaged in the National Imagery Transmission Format Version 2.1 (NITF 2.1) container:
- **NITF File Header**: Contains basic file attributes and the number of image and data extension segments.
- **Image Segments (IS)**: The complex pixel array is stored here. A single IS can hold up to ~10 GB. Arrays exceeding this or 99,999 rows are split across multiple continuous Image Segments.
- **Data Extension Segment (DES)**: The first DES must be of type `XML_DATA_CONTENT` and contains the SICD XML metadata.

## 3. Metadata XML Schemas
The SICD XML metadata is grouped into specialized blocks. Key schemas include:

### 3.1 Grid Block
Describes the spatial sampling represented by the image grid:
- **ImagePlane**: `GROUND`, `SLANT`, or `OTHER`.
- **Type**: Defines the spatial sampling grid (e.g., `RGAZIM` for Range & Azimuth, `RGZERO` for Range & Zero Doppler, `XRGYCR`, `XCTYAT`, `PLANE`).
- **TimeCOAPoly**: Center of Aperture (COA) time as a 2D polynomial of image coordinates.
- **Row & Column Parameters**: Specify unit vectors (`UVectECF`), sample spacing (`SS`), impulse response widths, and spatial frequency domain attributes (e.g., center frequency, bandwidth, windowing functions).

### 3.2 ImageCreation Block
Contains optional general information about the image creation process:
- **Application**: Name and version of the processing software.
- **DateTime**: Date and time of processing.
- **Site** & **Profile**: Creation site and the specific metadata profile used.

### 3.3 SCPCOA (Scene Center Point / Center of Aperture) Block
Provides key COA parameters evaluated at the Scene Center Point (SCP) for direct search and discovery:
- **SCPTime**: Precise COA time for the SCP.
- **Aperture Reference Point (ARP)**: Position, velocity, and acceleration vectors at the SCP COA.
- **Viewing Geometry**: Parameters like `SlantRange`, `GroundRange`, `SideOfTrack`, and `DopplerConeAng`.
- **Earth Tangent Plane (ETP) Angles**: `GrazeAng`, `IncidenceAng`, `TwistAng`, `SlopeAng`, `AzimAng`, and `LayoverAng` relative to the ETP at the SCP.
- **Bistatic Geometry**: Additional transmit and receive phase center parameters if `CollectType = BISTATIC`.

### 3.4 RadarCollection Block
Describes the physical radar collection:
- **TxFrequency**: Minimum and maximum transmitted RF frequency.
- **Waveform**: Parameters such as pulse length, RF bandwidth, FM rate, and receive demodulation type (`STRETCH` or `CHIRP`).
- **TxPolarization**: Transmit polarization (e.g., `V`, `H`, `RHC`, `LHC`, `SEQUENCE`).
- **RcvChannels**: Details for each receive data channel, including polarization.
- **Area**: Corner points and polygons defining the exact imaged region.

## 4. Image Projections
Image projections relate the 2D image pixel grid to the 3D geolocated scene. Projections occur along contours of constant range and range rate (R/Rdot) defined by the COA geometry:

- **Image-to-Scene Projection**: Maps a specific image grid location to a 3D geolocated point on a scene surface (e.g., a constant Height Above Ellipsoid (HAE) surface or a Digital Elevation Model (DEM)). This involves intersecting the precise R/Rdot contour for the given pixel with the scene surface, often solved iteratively.
- **Scene-to-Image Projection**: Maps a geolocated 3D point in the scene to a 2D image pixel grid location. This iterative approach computes a sequence of ground plane points projected along straight lines to the image plane until converging on the precise image grid location.
- **Simple Ground Plane Projection**: A computationally faster, less rigorous projection used for general resampling (e.g., generating overview images). It assumes the image grid is uniformly spaced in the image plane and projects along straight lines to the ground plane, yielding high accuracy near the SCP that degrades slowly towards the edges.
