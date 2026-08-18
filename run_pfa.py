import os
import time
import argparse
from glob import glob

from diffpfa.algo import PFAConfig, PFAEngine
from diffpfa.io import CPHDReader, SICDWriter
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tools.convert2png import visualize

def run_pfa(cphd_path, mode, output_dir):
    print(f"\nProcessing {os.path.basename(cphd_path)} with {mode}...")
    reader = CPHDReader(cphd_path)
    writer = SICDWriter()
    
    # Create mode-specific output subdirectory
    mode_output_dir = os.path.join(output_dir, mode)
    png_output_dir = os.path.join(mode_output_dir, "pngs")
    os.makedirs(mode_output_dir, exist_ok=True)
    os.makedirs(png_output_dir, exist_ok=True)

    cfg = PFAConfig(
        mode=mode,
        image_area_mode="ImageArea",
        output_dir=mode_output_dir,
        device="cuda",
        num_subpatches=1
    )
    engine = PFAEngine(reader, writer, cfg)
    
    start_time = time.time()
    outs = engine.run()
    end_time = time.time()
    
    # Create PNGs for each generated SICD
    for out_nitf in outs:
        base_name = os.path.splitext(os.path.basename(out_nitf))[0]
        png_path = os.path.join(png_output_dir, f"{base_name}.png")
        print(f"Creating PNG for {base_name}...")
        visualize(out_nitf, png_path)
    
    print(f"[TIMING] {mode} Mode took {end_time - start_time:.2f} seconds")
    return outs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PFA processing on a directory of CPHD files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .cphd files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output SICD files")
    
    args = parser.parse_args()
    
    # Find all .cphd files in the input directory
    cphd_files = glob(os.path.join(args.input_dir, "*.cphd"))
    
    if not cphd_files:
        print(f"No .cphd files found in {args.input_dir}")
        exit(1)
        
    print(f"Found {len(cphd_files)} .cphd files to process.")
    
    for cphd_path in cphd_files:
        run_pfa(cphd_path, "cztnufft", args.output_dir)
        run_pfa(cphd_path, "nufft", args.output_dir)
