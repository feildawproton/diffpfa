#!/home/feildaw/mypyenv/bin/python

import os
import subprocess
import sys

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Y{suffix}"

print("Fetching file list from S3... (this may take a moment)")
cmd = ["aws", "s3", "ls", "--recursive", "--no-sign-request", "s3://umbra-open-data-catalog/"]
try:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
except Exception as e:
    print(f"Error running aws cli: {e}")
    sys.exit(1)

files = {}

for line in process.stdout:
    parts = line.split(None, 3)
    if len(parts) >= 4:
        size_str = parts[2]
        path = parts[3].strip()
        
        if path.endswith('_CPHD.cphd'):
            ext = 'cphd'
            base = path[:-10]
        elif path.endswith('_SICD.nitf'):
            ext = 'nitf'
            base = path[:-10]
        else:
            continue
            
        try:
            size = int(size_str)
        except ValueError:
            continue
            
        if base not in files:
            files[base] = {}
        files[base][ext] = size

process.wait()

# Keep only those with a .cphd and sort by .cphd size
cphd_files = [(base, data['cphd'], data.get('nitf')) for base, data in files.items() if 'cphd' in data]
cphd_files.sort(key=lambda x: x[1])

print(f"{'#':<3} | {'Base Path':<73} | {'CPHD Size':<12} | {'SICD Size':<12}")
print("-" * 110)
for i, (base, cphd_size, nitf_size) in enumerate(cphd_files[:20], 1):
    nitf_str = sizeof_fmt(nitf_size) if nitf_size is not None else "N/A"
    cphd_str = sizeof_fmt(cphd_size)
    
    # Truncate base path if it's too long, adding ellipsis
    if len(base) > 73:
        display_base = "..." + base[-70:]
    else:
        display_base = base
        
    print(f"{i:<3} | {display_base:<73} | {cphd_str:<12} | {nitf_str:<12}")

print()
while True:
    choice = input("Enter the number of the pair to download (or 'q' to quit): ").strip()
    if choice.lower() == 'q':
        break
    try:
        idx = int(choice) - 1
        if 0 <= idx < min(20, len(cphd_files)):
            selected_base = cphd_files[idx][0]
            dest_dir = "/home/feildaw/data"
            os.makedirs(dest_dir, exist_ok=True)
            
            cphd_s3_path = f"s3://umbra-open-data-catalog/{selected_base}_CPHD.cphd"
            print(f"Downloading {cphd_s3_path}...")
            subprocess.run(["aws", "s3", "cp", "--no-sign-request", cphd_s3_path, dest_dir])
            
            has_nitf = cphd_files[idx][2] is not None
            if has_nitf:
                nitf_s3_path = f"s3://umbra-open-data-catalog/{selected_base}_SICD.nitf"
                print(f"Downloading {nitf_s3_path}...")
                subprocess.run(["aws", "s3", "cp", "--no-sign-request", nitf_s3_path, dest_dir])
            else:
                print("No SICD file associated with this pair.")
            
            print(f"Download complete! Files saved to {dest_dir}")
            break
        else:
            print("Invalid selection. Please try again.")
    except ValueError:
        print("Please enter a valid number or 'q'.")
