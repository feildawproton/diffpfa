import argparse
import os
import sys
from glob import glob

# Add the current directory to sys.path so we can import from tools
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tools.convert2png import visualize

def process_directory(sicd_dir):
    # Find all nitf files in the directory
    sicd_files = glob(os.path.join(sicd_dir, "*.nitf"))
    
    if not sicd_files:
        print(f"No .nitf files found in {sicd_dir}")
        return
        
    print(f"Found {len(sicd_files)} SICD files to visualize.")
    
    # Create the png subdirectory
    png_dir = os.path.join(sicd_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)
    
    for i, sicd_path in enumerate(sicd_files, 1):
        base_name = os.path.splitext(os.path.basename(sicd_path))[0]
        png_path = os.path.join(png_dir, f"{base_name}.png")
        
        print(f"[{i}/{len(sicd_files)}] Visualizing {base_name}...")
        try:
            visualize(sicd_path, png_path)
        except Exception as e:
            print(f"Error visualizing {sicd_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a directory of SICD NITF files to PNGs.")
    parser.add_argument("sicd_dir", type=str, help="Directory containing the output SICD (.nitf) files")
    
    args = parser.parse_args()
    
    process_directory(args.sicd_dir)
