#!/usr/bin/env python3
"""
list_latest_umbra.py: Lists and downloads the newest open-data SAR collections
(CPHD and SICD pairs) from the Umbra Open Data Catalog on AWS S3, ordered by newness.
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta

DEFAULT_DEST_DIR = "/home/feildaw/data"
S3_BUCKET_URI = "s3://umbra-open-data-catalog/"
CACHE_FILE = "/tmp/umbra_s3_catalog_cache.json"
CACHE_TTL_HOURS = 12


def sizeof_fmt(num, suffix="B"):
    if num is None:
        return "N/A"
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Y{suffix}"


def parse_timestamp_from_path(path_str, s3_date_str=""):
    """
    Extracts collection datetime from the filename 'YYYY-MM-DD-HH-MM-SS'
    or falls back to S3 modification datetime.
    """
    base_name = os.path.basename(path_str)
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", base_name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S"), match.group(1)
        except ValueError:
            pass
    if s3_date_str:
        try:
            return datetime.strptime(s3_date_str, "%Y-%m-%d %H:%M:%S"), s3_date_str
        except ValueError:
            pass
    return datetime.min, "Unknown"


def fetch_raw_s3_list(force_refresh=False):
    if not force_refresh and os.path.exists(CACHE_FILE):
        mtime = os.path.getmtime(CACHE_FILE)
        if time.time() - mtime < CACHE_TTL_HOURS * 3600:
            print(f"[*] Loading cached catalog list from {CACHE_FILE} (use --refresh to update)...")
            with open(CACHE_FILE, "r") as fp:
                return json.load(fp)

    print(f"[*] Querying S3 catalog from {S3_BUCKET_URI}... (this may take ~30s)")
    cmd = ["aws", "s3", "ls", "--recursive", "--no-sign-request", S3_BUCKET_URI]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    except Exception as e:
        print(f"[!] Error running aws cli: {e}")
        sys.exit(1)

    raw_items = []
    for line in process.stdout:
        parts = line.split(None, 3)
        if len(parts) >= 4:
            raw_items.append({
                "s3_date": f"{parts[0]} {parts[1]}",
                "size": int(parts[2]) if parts[2].isdigit() else 0,
                "path": parts[3].strip(),
            })

    process.wait()

    try:
        with open(CACHE_FILE, "w") as fp:
            json.dump(raw_items, fp)
    except Exception:
        pass

    return raw_items


def get_latest_collections(force_refresh=False, pair_only=True, max_cphd_gb=None, include_future=False):
    raw_items = fetch_raw_s3_list(force_refresh=force_refresh)
    files = {}

    for item in raw_items:
        path = item["path"]
        size = item["size"]
        s3_date_str = item["s3_date"]

        if path.endswith("_CPHD.cphd"):
            ext = "cphd"
            base = path[:-10]
        elif path.endswith("_SICD.nitf"):
            ext = "nitf"
            base = path[:-10]
        else:
            continue

        if base not in files:
            dt, dt_str = parse_timestamp_from_path(base, s3_date_str)
            files[base] = {
                "datetime": dt,
                "datetime_str": dt_str,
                "sizes": {},
            }
        files[base]["sizes"][ext] = size

    now = datetime.now() + timedelta(days=2)
    collections = []

    for base, data in files.items():
        dt = data["datetime"]
        if not include_future and dt > now:
            continue

        cphd_size = data["sizes"].get("cphd")
        nitf_size = data["sizes"].get("nitf")

        if pair_only and (cphd_size is None or nitf_size is None):
            continue

        if max_cphd_gb is not None and cphd_size is not None:
            if cphd_size > max_cphd_gb * 1024 * 1024 * 1024:
                continue

        collections.append({
            "base": base,
            "datetime": dt,
            "datetime_str": data["datetime_str"],
            "cphd_size": cphd_size,
            "nitf_size": nitf_size,
        })

    collections.sort(key=lambda x: x["datetime"], reverse=True)
    return collections


def download_collection(item, dest_dir=DEFAULT_DEST_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    base = item["base"]
    base_name = os.path.basename(base)

    print("\n" + "=" * 80)
    print(f"  Downloading Umbra SAR Collection: {base_name}")
    print(f"  Destination: {dest_dir}")
    print("=" * 80)

    if item["cphd_size"] is not None:
        cphd_s3 = f"{S3_BUCKET_URI}{base}_CPHD.cphd"
        print(f"[*] Downloading CPHD ({sizeof_fmt(item['cphd_size'])})...")
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", cphd_s3, dest_dir])

    if item["nitf_size"] is not None:
        nitf_s3 = f"{S3_BUCKET_URI}{base}_SICD.nitf"
        print(f"[*] Downloading SICD ({sizeof_fmt(item['nitf_size'])})...")
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", nitf_s3, dest_dir])

    print(f"\n[+] Download complete! Files saved in: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="List and download the newest UMBRA Open SAR collections (CPHD & SICD pairs)."
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of newest collections to display (default: 20).")
    parser.add_argument("--dest", default=DEFAULT_DEST_DIR, help=f"Destination directory (default: {DEFAULT_DEST_DIR}).")
    parser.add_argument("--max-cphd-gb", type=float, default=None, help="Optional maximum CPHD size filter in GB.")
    parser.add_argument("--all-files", action="store_true", help="Include collections missing either CPHD or SICD.")
    parser.add_argument("--include-future", action="store_true", help="Include test files with simulated future timestamps.")
    parser.add_argument("--refresh", action="store_true", help="Force refresh catalog cache from S3.")
    parser.add_argument("--download", type=int, default=None, help="Non-interactive: 1-indexed number to download.")
    args = parser.parse_args()

    collections = get_latest_collections(
        force_refresh=args.refresh,
        pair_only=not args.all_files,
        max_cphd_gb=args.max_cphd_gb,
        include_future=args.include_future,
    )

    if not collections:
        print("[!] No matching collections found.")
        return

    n_show = min(args.limit, len(collections))
    print(f"\nFound {len(collections)} total matching collections. Showing top {n_show} NEWEST:")
    print("=" * 125)
    print(f"{'#':<3} | {'Collect Datetime':<19} | {'CPHD Size':<10} | {'SICD Size':<10} | {'Collection Base Name'}")
    print("-" * 125)

    for i, item in enumerate(collections[:n_show], 1):
        dt_str = item["datetime_str"]
        cphd_str = sizeof_fmt(item["cphd_size"])
        nitf_str = sizeof_fmt(item["nitf_size"])
        base_name = os.path.basename(item["base"])
        print(f"{i:<3} | {dt_str:<19} | {cphd_str:<10} | {nitf_str:<10} | {base_name}")
    print("=" * 125)

    if args.download is not None:
        idx = args.download - 1
        if 0 <= idx < len(collections):
            download_collection(collections[idx], dest_dir=args.dest)
        else:
            print(f"[!] Invalid selection #{args.download}. Must be between 1 and {len(collections)}.")
        return

    # Interactive prompt
    while True:
        choice = input("\nEnter number of the pair to download (or 'q' to quit): ").strip()
        if choice.lower() in ["q", "quit", "exit"]:
            print("Exiting.")
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(collections):
                download_collection(collections[idx], dest_dir=args.dest)
                break
            else:
                print(f"[!] Invalid number. Please enter 1-{n_show}.")
        except ValueError:
            print("[!] Please enter a valid number or 'q'.")


if __name__ == "__main__":
    main()
