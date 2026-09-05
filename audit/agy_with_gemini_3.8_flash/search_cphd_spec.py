import pypdf
from pathlib import Path

pdf_path = Path("/home/feildaw/diffpfa/references/NGA.STND.0068-1_1.1.0_CPHD_DIDD_FINAL.pdf")
reader = pypdf.PdfReader(str(pdf_path))

for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if "SceneCoordinates" in text:
        print(f"Page {i+1}:")
        for line in text.split("\n"):
            if any(k in line.lower() for k in ["scenecoordinates", "imagearea", "referencesurface", "iarp"]):
                print(" ", line[:120])
