import numpy as np
import sarkit.sicd as ss
import sarkit.cphd as sc
from pathlib import Path

stem = "2025-10-26-05-00-15_UMBRA-08"
cphd_path = Path(f"/home/feildaw/data/{stem}_CPHD.cphd")
umbra_sicd = Path(f"/home/feildaw/data/{stem}_SICD.nitf")

with open(cphd_path, "rb") as f:
    r = sc.Reader(f)
    xh = sc.XmlHelper(r.metadata.xmltree)
    uIAX = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAX"))
    uIAY = np.array(xh.load("./{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar/{*}uIAY"))
    ia_x1y1 = np.array(xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X1Y1"))
    ia_x2y2 = np.array(xh.load("./{*}SceneCoordinates/{*}ImageArea/{*}X2Y2"))
    iarp = np.array(xh.load("./{*}SceneCoordinates/{*}IARP/{*}ECF"))

with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r:
    xh = ss.XmlHelper(r.metadata.xmltree)
    row_u = np.array(xh.load("./{*}Grid/{*}Row/{*}UVectECF"))
    col_u = np.array(xh.load("./{*}Grid/{*}Col/{*}UVectECF"))
    nr = xh.load("./{*}ImageData/{*}NumRows")
    nc = xh.load("./{*}ImageData/{*}NumCols")
    row_ss = xh.load("./{*}Grid/{*}Row/{*}SS")
    col_ss = xh.load("./{*}Grid/{*}Col/{*}SS")
    scp_r = xh.load("./{*}ImageData/{*}SCPPixel/{*}Row")
    scp_c = xh.load("./{*}ImageData/{*}SCPPixel/{*}Col")

print(f"CPHD ImageArea: X=[{ia_x1y1[0]}, {ia_x2y2[0]}], Y=[{ia_x1y1[1]}, {ia_x2y2[1]}]")

# The 4 corners of CPHD ImageArea on ReferenceSurface (relative to IARP):
# (x, y):
corners = np.array([
    [ia_x1y1[0], ia_x1y1[1]],
    [ia_x1y1[0], ia_x2y2[1]],
    [ia_x2y2[0], ia_x2y2[1]],
    [ia_x2y2[0], ia_x1y1[1]],
])

# For each corner, the displacement vector in 3D ECF is:
# dP = x * uIAX + y * uIAY
dP_corners = corners[:, 0:1] * uIAX + corners[:, 1:2] * uIAY

# Project dP onto slant plane basis vectors (Row and Col):
proj_row = np.dot(dP_corners, row_u)
proj_col = np.dot(dP_corners, col_u)

print("\nProjection of CPHD ImageArea corners onto Slant Plane (Row=Range, Col=Azimuth):")
for i in range(4):
    print(f"  Corner {i+1} ({corners[i, 0]}, {corners[i, 1]}): Row={proj_row[i]:.2f} m, Col={proj_col[i]:.2f} m")

print(f"\nExtents of projected CPHD ImageArea on Slant Plane:")
print(f"  Row (Range):   [{proj_row.min():.2f}, {proj_row.max():.2f}] m (Total span: {proj_row.max() - proj_row.min():.2f} m)")
print(f"  Col (Azimuth): [{proj_col.min():.2f}, {proj_col.max():.2f}] m (Total span: {proj_col.max() - proj_col.min():.2f} m)")

print(f"\nActual Umbra SICD Slant Plane Extents:")
umbra_row_extent = (np.array([0, nr - 1]) - scp_r) * row_ss
umbra_col_extent = (np.array([0, nc - 1]) - scp_c) * col_ss
print(f"  Row (Range):   [{umbra_row_extent.min():.2f}, {umbra_row_extent.max():.2f}] m (Total span: {umbra_row_extent.max() - umbra_row_extent.min():.2f} m)")
print(f"  Col (Azimuth): [{umbra_col_extent.min():.2f}, {umbra_col_extent.max():.2f}] m (Total span: {umbra_col_extent.max() - umbra_col_extent.min():.2f} m)")

# DiffPFA current extents:
print(f"\nDiffPFA SICD Slant Plane Extents:")
print(f"  Row (Range):   [-2500.00, 2500.00] m (Total span: 5000.00 m)")
print(f"  Col (Azimuth): [-2500.00, 2500.00] m (Total span: 5000.00 m)")
