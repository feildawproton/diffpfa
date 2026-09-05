import sarkit.cphd as sc
import sarkit.sicd as ss
from pathlib import Path

DATA_DIR = Path("/home/feildaw/data")

for cphd_path in sorted(DATA_DIR.glob("*_CPHD.cphd")):
    stem = cphd_path.name.replace("_CPHD.cphd", "")
    with open(cphd_path, "rb") as f:
        r = sc.Reader(f)
        xml = r.metadata.xmltree
        xh = sc.XmlHelper(xml)
        
        collector = xh.load("./{*}CollectionID/{*}CollectorName")
        radar_mode = xh.load("./{*}CollectionID/{*}RadarMode/{*}ModeType")
        domain_type = xh.load("./{*}Global/{*}DomainType")
        fx_min = xh.load("./{*}Global/{*}FxBand/{*}FxMin")
        fx_max = xh.load("./{*}Global/{*}FxBand/{*}FxMax")
        bw = (fx_max - fx_min) / 1e6 if fx_min and fx_max else None
        
        # Channels
        channels = xml.findall(".//{*}Data/{*}Channel")
        ch_ids = [c.find("./{*}Identifier").text for c in channels]
        num_vects = [c.find("./{*}NumVectors").text for c in channels]
        num_samp = [c.find("./{*}NumSamples").text for c in channels]
        
        # PVP parameters present in first channel
        ch0_pvp = r.read_pvps(ch_ids[0])
        pvp_names = ch0_pvp.dtype.names
        has_tx_fm = "TxFMRate" in pvp_names
        tx_fm_sample = ch0_pvp["TxFMRate"][0] if has_tx_fm else None
        
        # Polarizations
        tx_pols = []
        rcv_pols = []
        for p in xml.findall(".//{*}Channel/{*}Parameters"):
            tx_pols.append(p.find("./{*}Polarization/{*}TxPol").text if p.find("./{*}Polarization/{*}TxPol") is not None else "?")
            rcv_pols.append(p.find("./{*}Polarization/{*}RcvPol").text if p.find("./{*}Polarization/{*}RcvPol") is not None else "?")
            
        print(f"=== {stem} ===")
        print(f"  Collector: {collector}, Mode: {radar_mode}, Domain: {domain_type}")
        print(f"  Channels ({len(ch_ids)}): {ch_ids}")
        print(f"  Vectors x Samples: {num_vects[0]} x {num_samp[0]}")
        print(f"  TxPol: {set(tx_pols)}, RcvPol: {set(rcv_pols)}")
        print(f"  FxBand: {bw:.2f} MHz ({fx_min/1e9:.3f} to {fx_max/1e9:.3f} GHz)")
        print(f"  Has TxFMRate: {has_tx_fm}, sample: {tx_fm_sample}")
        
    umbra_sicd = DATA_DIR / f"{stem}_SICD.nitf"
    with open(umbra_sicd, "rb") as f, ss.NitfReader(f) as r_s:
        xh_s = ss.XmlHelper(r_s.metadata.xmltree)
        algo = xh_s.load("./{*}ImageFormation/{*}ImageFormAlgo")
        tx_pol = xh_s.load("./{*}RadarCollection/{*}TxPolarization")
        tx_rcv = xh_s.load("./{*}ImageFormation/{*}TxRcvPolarizationProc")
        app = xh_s.load("./{*}ImageCreation/{*}Application")
        print(f"  Umbra SICD: Algo={algo}, TxPol={tx_pol}, TxRcv={tx_rcv}, App={app}")
