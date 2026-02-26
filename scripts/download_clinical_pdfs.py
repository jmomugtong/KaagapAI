"""Download real open-source clinical PDFs for ingestion testing.

Downloads publicly available clinical documents from WHO and CDC to
datasets/clinical_docs/. These are NOT committed to git (too large).

Run: python scripts/download_clinical_pdfs.py
"""

import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "clinical_docs"

DOCUMENTS = [
    {
        "filename": "who_newborn_care_pocket_guide.pdf",
        "url": "https://iris.who.int/bitstream/handle/10665/361145/9789290619659-eng.pdf",
        "description": "WHO Early Essential Newborn Care Pocket Guide",
        "license": "CC BY-NC-SA 3.0 IGO",
        "type": "protocol",
    },
    {
        "filename": "cdc_dengue_clinical_management.pdf",
        "url": "https://www.cdc.gov/dengue/media/pdfs/2024/05/20240521_342849-B_PRESS_READY_PocketGuideDCMC_UPDATE.pdf",
        "description": "CDC Dengue Clinical Management Pocket Guide",
        "license": "Public Domain (US Government)",
        "type": "guideline",
    },
]


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": "KaagapAI/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response:
        total = response.headers.get("Content-Length")
        data = response.read()

    dest.write_bytes(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"  Downloaded {size_mb:.1f} MB")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading clinical PDFs to {OUTPUT_DIR}\n")

    for doc in DOCUMENTS:
        dest = OUTPUT_DIR / doc["filename"]
        print(f"[{doc['type'].upper()}] {doc['description']}")
        print(f"  License: {doc['license']}")

        if dest.exists():
            print(f"  Already exists, skipping.\n")
            continue

        try:
            download_file(doc["url"], dest)
            print(f"  OK\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            print("  You can download manually from the URL above.\n")

    print("Done!")
    print(
        "\nNote: These files are in .gitignore (datasets/clinical_docs/) "
        "and should not be committed."
    )


if __name__ == "__main__":
    main()
