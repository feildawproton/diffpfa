import argparse
import os
import sys

from diffpfa.algo import PFAConfig, PFAEngine
from diffpfa.io import CPHDReader, SICDWriter


def main():
    parser = argparse.ArgumentParser(description="PyTorch Polar Format Algorithm (PFA) CPHD Image Processor.")
    parser.add_argument("cphd_path", type=str, help="Path to input CPHD file")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Directory to save output SICD NITF files")
    parser.add_argument("-m", "--mode", type=str, choices=["czt", "nufft", "hybrid"], default="hybrid", help="PFA interpolation algorithm mode")
    parser.add_argument("-b", "--image_area_mode", type=str, choices=["ImageArea", "ExtendedArea", "InscribedRectangle", "TargetPixelSpacing"], default="ImageArea", help="Spatial image bounds mode")
    parser.add_argument("-s", "--spacing", type=float, nargs=2, metavar=("DU", "DR"), default=None, help="Custom pixel spacing (du dr) in meters")
    parser.add_argument("--debug_channels", action="store_true", help="Save intermediate uncombined channel SICD files")
    parser.add_argument("--no_align", action="store_true", help="Disable relative phase alignment between sub-channels")
    parser.add_argument("--backend", type=str, choices=["auto", "sarkit", "sarpy"], default="auto", help="I/O backend to use")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', 'cuda')")

    args = parser.parse_args()

    if not os.path.exists(args.cphd_path):
        print(f"Error: CPHD file not found at {args.cphd_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening CPHD file: {args.cphd_path} with backend {args.backend}")
    reader = CPHDReader(args.cphd_path, backend=args.backend)
    writer = SICDWriter(backend=args.backend)

    custom_spacing = tuple(args.spacing) if args.spacing else None

    config = PFAConfig(
        mode=args.mode,
        image_area_mode=args.image_area_mode,
        custom_pixel_spacing=custom_spacing,
        align_subchannels=not args.no_align,
        debug_save_channels=args.debug_channels,
        output_dir=args.output_dir,
        device=args.device,
    )

    print(f"Starting PFA Engine (mode={config.mode}, area_mode={config.image_area_mode}, device={config.device})...")
    engine = PFAEngine(reader, writer, config)
    output_files = engine.run()

    print("\nProcessing complete! Generated SICD-U files:")
    for f in output_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
