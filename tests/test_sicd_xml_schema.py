import os
import lxml.etree as ET
import pytest
import sarkit.sicd as sksicd
from diffpfa.IFP import IFAProcessor

cphd_path = "/home/feildaw/data/2023-09-11-10-37-05_UMBRA-05_CPHD.cphd"

@pytest.mark.skipif(not os.path.exists(cphd_path), reason="Test CPHD dataset not found")
def test_sicd_slant_schema_validation(tmp_path):
    proc = IFAProcessor(
        cphd_path=cphd_path,
        output_dir=str(tmp_path),
        image_plane="SLANT",
        device="cuda"
    )
    out_files, _, _, _ = proc.run()
    assert len(out_files) > 0
    nitf_path = out_files[0]
    
    with open(nitf_path, "rb") as f:
        reader = sksicd.NitfReader(f)
        xmltree = reader.metadata.xmltree

    schema_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "schemas"))
    xsd_path = os.path.join(schema_dir, "SICD_schema_V1.3.0_2021_11_30.xsd")
    if not os.path.exists(xsd_path):
        xsd_path = os.path.join(sksicd.schemas.__path__[0], 'SICD_schema_V1.3.0_2021_11_30.xsd')
    schema = ET.XMLSchema(ET.parse(xsd_path))
    assert schema.validate(xmltree), f"Schema errors: {schema.error_log}"

@pytest.mark.skipif(not os.path.exists(cphd_path), reason="Test CPHD dataset not found")
def test_sicd_ground_schema_validation(tmp_path):
    proc = IFAProcessor(
        cphd_path=cphd_path,
        output_dir=str(tmp_path),
        image_plane="GROUND",
        device="cuda"
    )
    out_files, _, _, _ = proc.run()
    assert len(out_files) > 0
    nitf_path = out_files[0]
    
    with open(nitf_path, "rb") as f:
        reader = sksicd.NitfReader(f)
        xmltree = reader.metadata.xmltree

    schema_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "schemas"))
    xsd_path = os.path.join(schema_dir, "SICD_schema_V1.3.0_2021_11_30.xsd")
    if not os.path.exists(xsd_path):
        xsd_path = os.path.join(sksicd.schemas.__path__[0], 'SICD_schema_V1.3.0_2021_11_30.xsd')
    schema = ET.XMLSchema(ET.parse(xsd_path))
    assert schema.validate(xmltree), f"Schema errors: {schema.error_log}"


@pytest.mark.skipif(not os.path.exists(cphd_path), reason="Test CPHD dataset not found")
def test_sicd_sarkit_consistency_check(tmp_path):
    import subprocess
    import sys
    proc = IFAProcessor(
        cphd_path=cphd_path,
        output_dir=str(tmp_path),
        image_plane="SLANT",
        device="cuda"
    )
    out_files, _, _, _ = proc.run()
    assert len(out_files) > 0
    nitf_path = out_files[0]
    
    with open(nitf_path, "rb") as f:
        reader = sksicd.NitfReader(f)
        h = sksicd.XmlHelper(reader.metadata.xmltree)
    
    # 1. Verify uniform window ImpRespWid * ImpRespBW ~ 0.886
    for d in ("Row", "Col"):
        k = h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespWid") * h.load(f"./{{*}}Grid/{{*}}{d}/{{*}}ImpRespBW")
        assert abs(k - 0.886) < 0.02, f"{d}: ImpRespWid*ImpRespBW = {k:.4f}, expected ~0.886"

    # 2. Verify Grid Type is RGAZIM per DIDD §4.15.1
    grid_type = h.load("./{*}Grid/{*}Type")
    assert grid_type == "RGAZIM", f"Expected Grid Type RGAZIM, got {grid_type}"

    # 3. Run sarkit sicdcheck executable
    sicdcheck_bin = os.path.join(os.path.dirname(sys.executable), "sicdcheck")
    if os.path.exists(sicdcheck_bin):
        res = subprocess.run([sicdcheck_bin, nitf_path], capture_output=True, text=True)
        errs = [l.strip() for l in res.stdout.splitlines() if "[Error]" in l]
        assert not errs, f"sicdcheck reported errors:\n" + "\n".join(errs)


def _make_synthetic_group(nsub=3, ns=128, npulse=128, bw_total=600e6, fc=9.6e9):
    import numpy as np
    import sarkit.wgs84 as w
    from diffpfa.types import CPHDMetadata, ImageAreaBounds
    from diffpfa.constants import SPEED_OF_LIGHT as C

    t = np.arange(npulse) / 1000.0 + 0.5
    srp = w.geodetic_to_cartesian([46.9, -96.8, 250.0])
    up = w.up([46.9, -96.8, 250.0])
    east = np.array([-np.sin(np.radians(-96.8)), np.cos(np.radians(-96.8)), 0.0])
    north = np.cross(up, east)
    R = 567000.0
    arp0 = srp + R * (np.sin(np.radians(45)) * up + np.cos(np.radians(45)) * (-north))
    vel = 7500.0 * east
    pos = arp0[None, :] + (t - t.mean())[:, None] * vel[None, :]
    P = srp[None, :] - pos
    Pn = np.linalg.norm(P, axis=1)
    u_row = P[npulse // 2] / Pn[npulse // 2]
    u_v = vel[0] / np.linalg.norm(vel[0])
    u_col = u_v - np.dot(u_v, u_row) * u_row
    u_col /= np.linalg.norm(u_col)
    u_col = -u_col

    cos_t = P @ u_col / Pn
    sin_t = P @ u_row / Pn
    bw_sub = bw_total / nsub
    chans = []
    for m in range(nsub):
        f0 = fc - bw_total / 2 + m * bw_sub
        step = bw_sub / ns
        F = f0 + np.arange(ns) * step
        s = np.exp(-1j * 2 * np.pi * (2 * F[None, :] / C) * (0.0 * cos_t + 0.0 * sin_t)[:, None])
        pvp = dict(
            SRPPos=np.tile(srp, (npulse, 1)), TxPos=pos, RcvPos=pos,
            TxVel=np.tile(vel, (npulse, 1)), RcvVel=np.tile(vel, (npulse, 1)),
            SC0=np.full(npulse, f0), SCSS=np.full(npulse, step),
            TxTime=t + m * 137.3217e-6, RcvTime=t + m * 137.3217e-6 + 2 * Pn / C,
            SIGNAL=np.ones(npulse, int)
        )
        chans.append((s.astype(np.complex64), pvp, f0 + bw_sub / 2))

    meta = CPHDMetadata(
        domain_type="FX", sgn=-1, global_fx_min=fc - bw_total / 2, global_fx_max=fc + bw_total / 2,
        iarp_ecf=srp, uIAX=u_col, uIAY=u_row, ref_ch_id="0",
        image_area=ImageAreaBounds(-20, -20, 20, 20, None), extended_area=None,
        collection_start="2026-01-01T00:00:00Z", radar_mode="SPOTLIGHT",
        classification="UNCLASSIFIED", srp_ecf=srp, arp_pos_coa=pos[npulse // 2],
        arp_vel_coa=vel[0], side_of_track="L", line_spacing=None, sample_spacing=None,
        raw_meta=None, ref_uIAX=u_col, ref_uIAY=u_row
    )
    return chans, meta


def test_multichannel_stepped_chirp_sicd_metadata(tmp_path):
    """Audit G1: Verify SICD metadata spans combined k-space grid for stepped chirps."""
    import subprocess
    import sys
    from diffpfa.IFA.PFA import pfa_per_polar

    chans, meta = _make_synthetic_group(nsub=3, ns=128, npulse=128, bw_total=600e6)
    img, bw_r, bw_u, N_r, N_u, rot = pfa_per_polar(
        channel_signals=[c[0] for c in chans],
        channel_pvps=[c[1] for c in chans],
        channel_fxcs=[c[2] for c in chans],
        channel_domain_types=["FX"] * len(chans),
        ref_rcv_time=chans[0][1]["RcvTime"],
        cphd_meta=meta,
        u_min=-20, u_max=20, r_min=-20, r_max=20,
        image_oversample=1.25,
        device="cpu"
    )

    proc = IFAProcessor(cphd_path="synthetic_CPHD.cphd", output_dir=str(tmp_path), image_plane="SLANT", device="cpu")
    out_nitf = os.path.join(str(tmp_path), "stepped_chirp_test.nitf")
    proc._write_sicd(
        out_nitf, img.T, meta, "V", "V", bw_r, bw_u, N_r, N_u,
        -20, -20, 40.0 / N_u, 40.0 / N_r,
        ref_pvp=chans[0][1], num_samples=chans[0][0].shape[1], is_rotated=rot,
        channel_pvps=[c[1] for c in chans], channel_signals=[c[0] for c in chans]
    )

    with open(out_nitf, "rb") as f:
        reader = sksicd.NitfReader(f)
        h = sksicd.XmlHelper(reader.metadata.xmltree)

    row_kctr = h.load("./{*}Grid/{*}Row/{*}KCtr")
    krg1 = h.load("./{*}PFA/{*}Krg1")
    krg2 = h.load("./{*}PFA/{*}Krg2")
    imp_resp_bw = h.load("./{*}Grid/{*}Row/{*}ImpRespBW")

    # Range k-space center and bounds must reflect all 3 sub-bands
    assert 63.5 < row_kctr < 64.5, f"Row/KCtr {row_kctr} not centered around ~64.0 rad/m"
    assert krg2 - krg1 >= imp_resp_bw, f"PFA Krg [{krg1}, {krg2}] does not support ImpRespBW {imp_resp_bw}"

    sicdcheck_bin = os.path.join(os.path.dirname(sys.executable), "sicdcheck")
    if os.path.exists(sicdcheck_bin):
        res = subprocess.run([sicdcheck_bin, out_nitf], capture_output=True, text=True)
        errs = [l.strip() for l in res.stdout.splitlines() if "[Error]" in l]
        assert not errs, f"sicdcheck reported errors on stepped-chirp SICD:\n" + "\n".join(errs)


def test_asymmetric_framing_scp_pixel(tmp_path):
    """Audit G2: Verify SCPPixel and ICP projection reflect asymmetric spatial bounds."""
    import subprocess
    import sys
    from diffpfa.IFA.PFA import pfa_per_polar

    chans, meta = _make_synthetic_group(nsub=1, ns=128, npulse=128, bw_total=600e6)
    sig, pvp, fxc = chans[0]
    # Asymmetric bounds: u in [0, 40], r in [-10, 30]
    u_min, u_max = 0.0, 40.0
    r_min, r_max = -10.0, 30.0
    img, bw_r, bw_u, N_r, N_u, rot = pfa_per_polar(
        channel_signals=[sig], channel_pvps=[pvp], channel_fxcs=[fxc], channel_domain_types=["FX"],
        ref_rcv_time=pvp["RcvTime"], cphd_meta=meta,
        u_min=u_min, u_max=u_max, r_min=r_min, r_max=r_max,
        image_oversample=1.25,
        device="cpu"
    )

    proc = IFAProcessor(cphd_path="synthetic_CPHD.cphd", output_dir=str(tmp_path), image_plane="SLANT", device="cpu")
    out_nitf = os.path.join(str(tmp_path), "asymmetric_test.nitf")
    proc._write_sicd(
        out_nitf, img.T, meta, "V", "V", bw_r, bw_u, N_r, N_u,
        u_min, r_min, (u_max - u_min) / N_u, (r_max - r_min) / N_r,
        ref_pvp=pvp, num_samples=sig.shape[1], is_rotated=rot
    )

    with open(out_nitf, "rb") as f:
        reader = sksicd.NitfReader(f)
        h = sksicd.XmlHelper(reader.metadata.xmltree)

    scp_row = h.load("./{*}ImageData/{*}SCPPixel/{*}Row")
    scp_col = h.load("./{*}ImageData/{*}SCPPixel/{*}Col")

    # Target placed at SRP (0, 0):
    # u_idx = N_u/2 - u_c / du = N_u/2 - 20 / (40/N_u) = 0
    # r_idx = N_r/2 - r_c / dr = N_r/2 - 10 / (40/N_r) = N_r/4
    # In transposed image (rows=r, cols=u):
    expected_row = int(round(N_r / 2.0 - 10.0 / (40.0 / N_r)))
    expected_col = int(round(N_u / 2.0 - 20.0 / (40.0 / N_u)))
    assert scp_row == expected_row, f"SCPPixel/Row {scp_row} != expected {expected_row}"
    assert scp_col == expected_col, f"SCPPixel/Col {scp_col} != expected {expected_col}"

    sicdcheck_bin = os.path.join(os.path.dirname(sys.executable), "sicdcheck")
    if os.path.exists(sicdcheck_bin):
        res = subprocess.run([sicdcheck_bin, out_nitf], capture_output=True, text=True)
        errs = [l.strip() for l in res.stdout.splitlines() if "[Error]" in l]
        assert not errs, f"sicdcheck reported errors on asymmetric SICD:\n" + "\n".join(errs)

