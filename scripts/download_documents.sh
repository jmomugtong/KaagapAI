#!/usr/bin/env bash
# Download open-source clinical documents for MedQuery RAG corpus
# All documents are public domain or openly licensed from WHO, CDC, NIH, NICE
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "$0")/.." && pwd)/documents"
mkdir -p "$DOCS_DIR/clinical_protocols" "$DOCS_DIR/clinical_references"

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "  SKIP (exists): $(basename "$dest")"
        return
    fi
    echo "  Downloading: $(basename "$dest")"
    curl -fSL --retry 2 --max-time 120 -o "$dest" "$url" || echo "  FAILED: $url"
}

echo "=== Clinical Protocols ==="

download \
    "https://www.ncbi.nlm.nih.gov/books/NBK588130/pdf/Bookshelf_NBK588130.pdf" \
    "$DOCS_DIR/clinical_protocols/WHO_Malaria_Guidelines_2024.pdf"

download \
    "https://www.ncbi.nlm.nih.gov/books/NBK582440/pdf/Bookshelf_NBK582440.pdf" \
    "$DOCS_DIR/clinical_protocols/WHO_COVID19_Clinical_Mgmt_2023.pdf"

download \
    "https://www.ncbi.nlm.nih.gov/books/NBK570371/pdf/Bookshelf_NBK570371.pdf" \
    "$DOCS_DIR/clinical_protocols/NIH_COVID19_Treatment_2024.pdf"

download \
    "https://stacks.cdc.gov/view/cdc/122248/cdc_122248_DS1.pdf" \
    "$DOCS_DIR/clinical_protocols/CDC_Opioid_Prescribing_2022.pdf"

download \
    "https://www.cdc.gov/std/treatment-guidelines/STI-Guidelines-2021.pdf" \
    "$DOCS_DIR/clinical_protocols/CDC_STI_Treatment_2021.pdf"

download \
    "https://www.ncbi.nlm.nih.gov/books/NBK592432/pdf/Bookshelf_NBK592432.pdf" \
    "$DOCS_DIR/clinical_protocols/NICE_Head_Injury_2023.pdf"

echo ""
echo "=== Clinical References ==="

download \
    "https://iris.who.int/server/api/core/bitstreams/289a875c-cc89-4914-90ad-eb3c578ebaf6/content" \
    "$DOCS_DIR/clinical_references/WHO_Essential_Medicines_2023.pdf"

download \
    "https://www.cdc.gov/vaccines/hcp/imz-schedules/downloads/child/0-18yrs-child-combined-schedule.pdf" \
    "$DOCS_DIR/clinical_references/CDC_Child_Immunization_Schedule_2025.pdf"

download \
    "https://www.cdc.gov/vaccines/hcp/imz-schedules/downloads/adult/adult-combined-schedule.pdf" \
    "$DOCS_DIR/clinical_references/CDC_Adult_Immunization_Schedule_2025.pdf"

download \
    "https://www.cdc.gov/antibiotic-use/media/pdfs/Core-Elements-Outpatient-508.pdf" \
    "$DOCS_DIR/clinical_references/CDC_Antibiotic_Stewardship.pdf"

download \
    "https://www.cdc.gov/std/treatment-guidelines/wall-chart.pdf" \
    "$DOCS_DIR/clinical_references/CDC_STI_Treatment_WallChart_2021.pdf"

echo ""
echo "=== Done ==="
echo "Protocols:  $(ls "$DOCS_DIR/clinical_protocols/"*.pdf 2>/dev/null | wc -l) files"
echo "References: $(ls "$DOCS_DIR/clinical_references/"*.pdf 2>/dev/null | wc -l) files"
echo "Guidelines: $(ls "$DOCS_DIR/clinical_guidelines/"*.pdf 2>/dev/null | wc -l) files"
