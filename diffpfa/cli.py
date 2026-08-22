import argparse
import os
import sys

from diffpfa.algo.ifa_processor import IFAProcessor

def main():
    parser = argparse.ArgumentParser(description="diffpfa - Differentiable Polar Format Algorithm")
    parser.add_argument("cphd_path", type=str, help="Path to input CPHD file")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Directory to save output SICD files")
    parser.add_argument("-b", "--image_area_mode", type=str, choices=["ImageArea", "ExtendedArea", "InscribedRectangle", "TargetPixelSpacing"], default="ImageArea", help="Spatial image bounds mode")
    parser.add_argument("-s", "--spacing", type=float, nargs=2, metavar=("DU", "DR"), default=None, help="Custom pixel spacing (du dr) in meters")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cpu', 'cuda')")

    args = parser.parse_args()

    if not os.path.exists(args.cphd_path):
        print(f"Error: CPHD file not found at {args.cphd_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening CPHD file: {args.cphd_path}")

    custom_spacing = tuple(args.spacing) if args.spacing else None

    processor = IFAProcessor(
        cphd_path=args.cphd_path,
        output_dir=args.output_dir,
        image_area_mode=args.image_area_mode,
        custom_pixel_spacing=custom_spacing,
        device=args.device,
    )
    
    output_files = processor.run()
    
    print("\nProcessing complete. Generated files:")
    for f in output_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
