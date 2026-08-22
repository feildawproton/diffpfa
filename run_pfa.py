import argparse
import os
import sys
import time
from glob import glob
import numpy as np

from diffpfa.IFP import IFAProcessor

def run_pfa(cphd_path, output_dir):
    print(f"\nProcessing {os.path.basename(cphd_path)}...")
    
    os.makedirs(output_dir, exist_ok=True)

    t0_setup = time.perf_counter()
    processor = IFAProcessor(
        cphd_path=cphd_path,
        output_dir=output_dir,
        image_area_mode="ImageArea",
        device="cuda"
    )
    out_paths, internal_read_time, proc_time, write_time = processor.run()
    
    # Combine the class instantiation and all internal XML/disk reading into one bucket
    setup_and_read_time = (time.perf_counter() - t0_setup) - proc_time - write_time
    
    total_time = setup_and_read_time + proc_time + write_time
    
    print(f"[TIMING] Processing took {total_time:.4f} seconds")
    
    return {
        "setup_and_read_time": setup_and_read_time,
        "proc_time": proc_time,
        "write_time": write_time,
        "total_time": total_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PFA processing on a directory of CPHD files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .cphd files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output SICD files")
    
    args = parser.parse_args()
    
    cphd_files = glob(os.path.join(args.input_dir, "*.cphd"))
    
    if not cphd_files:
        print(f"No .cphd files found in {args.input_dir}")
        exit(1)
        
    print(f"Found {len(cphd_files)} .cphd files to process.")
    
    all_timings = {
        "setup_and_read_time": [],
        "proc_time": [],
        "write_time": [],
        "total_time": []
    }
    
    for cphd_path in cphd_files:
        timings = run_pfa(cphd_path, args.output_dir)
        for k, v in timings.items():
            all_timings[k].append(v)
            
    print("\n" + "="*60)
    print("TIMING STATISTICS (seconds) - N=" + str(len(cphd_files)))
    print("="*60)
    
    def print_stats(name, data):
        if not data: return
        arr = np.array(data)
        print(f"{name.upper()}:")
        print(f"  Mean:   {np.mean(arr):.4f}")
        print(f"  Median: {np.median(arr):.4f}")
        print(f"  StdDev: {np.std(arr):.4f}")
        print(f"  Min:    {np.min(arr):.4f}")
        print(f"  Max:    {np.max(arr):.4f}")
        print(f"  25th %: {np.percentile(arr, 25):.4f}")
        print(f"  75th %: {np.percentile(arr, 75):.4f}")
        print("-" * 40)

    for k in all_timings.keys():
        print_stats(k, all_timings[k])
