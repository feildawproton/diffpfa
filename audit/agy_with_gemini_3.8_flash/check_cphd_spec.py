import pypdf
from pathlib import Path

pdf_path = Path("/home/feildaw/diffpfa/references/NGA.STND.0068-1_1.1.0_CPHD_DIDD_FINAL.pdf")
reader = pypdf.PdfReader(str(pdf_path))

print(f"Total pages: {len(reader.pages)}")

found_pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if "ImageArea" in text and "ReferenceSurface" in text:
        found_pages.append((i + 1, text))

print(f"Found {len(found_pages)} pages mentioning ImageArea and ReferenceSurface:")
for pnum, text in found_pages[:5]:
    print(f"\n--- PAGE {pnum} ---")
    for line in text.split("\n"):
        if any(w in line for w in ["ImageArea", "ReferenceSurface", "Planar", "uIAX", "uIAY", "X1Y1", "X2Y2", "X1", "Y1"]):
            print("  ", line)
