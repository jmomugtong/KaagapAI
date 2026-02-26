#!/usr/bin/env bash
# Download open-source clinical documents for KaagapAI RAG corpus
# Philippine clinical guidelines from DOH, PSMID, PIDSP, PhilHealth + WHO international references
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "$0")/.." && pwd)/documents"
mkdir -p "$DOCS_DIR/clinical_guidelines" "$DOCS_DIR/clinical_protocols" "$DOCS_DIR/clinical_references"

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "  SKIP (exists): $(basename "$dest")"
        return
    fi
    echo "  Downloading: $(basename "$dest")"
    curl -fSL --retry 2 --max-time 120 -o "$dest" "$url" || echo "  FAILED: $url"
}

echo "=== Clinical Guidelines ==="

download \
    "https://philippinesocietyofhypertension.org.ph/ClinicalPracticeGuidelines.pdf" \
    "$DOCS_DIR/clinical_guidelines/PH_Hypertension_CPG_2020.pdf"

download \
    "http://diabetesphilippines.org/HOME/Forms/clinical_practice_guidelines_draft.pdf" \
    "$DOCS_DIR/clinical_guidelines/PH_Diabetes_UNITE_CPG.pdf"

download \
    "https://www.strokesocietyphilippines.org/wp-content/uploads/2024/07/CPG2024.pdf" \
    "$DOCS_DIR/clinical_guidelines/PH_Stroke_CPG_2024.pdf"

download \
    "http://www.pidsphil.org/home/wp-content/uploads/2017/06/2017_Dengue_CPG_Final.pdf" \
    "$DOCS_DIR/clinical_guidelines/PH_Dengue_CPG_2017.pdf"

download \
    "https://itis.doh.gov.ph/assets/img/downloads/mop/NTP_MOP_6th_Edition.pdf" \
    "$DOCS_DIR/clinical_guidelines/PH_TB_NTP_MOP_6th_Ed.pdf"

download \
    "https://www.ncbi.nlm.nih.gov/books/NBK588130/pdf/Bookshelf_NBK588130.pdf" \
    "$DOCS_DIR/clinical_guidelines/WHO_Malaria_Guidelines_2024.pdf"

echo ""
echo "=== Clinical Protocols ==="

download \
    "https://psmid.org/wp-content/uploads/2025/09/CPG-CAP-2016.pdf" \
    "$DOCS_DIR/clinical_protocols/PH_CAP_Adults_CPG_2016.pdf"

download \
    "http://www.pidsphil.org/home/wp-content/uploads/2022/03/1646542268113574.pdf" \
    "$DOCS_DIR/clinical_protocols/PH_CAP_Pediatric_CPG_2021.pdf"

download \
    "https://psmid.org/wp-content/uploads/2025/09/CPG-Leptospirosis-2010.pdf" \
    "$DOCS_DIR/clinical_protocols/PH_Leptospirosis_CPG_2010.pdf"

download \
    "http://www.pidsphil.org/home/wp-content/uploads/2022/03/CPG-OF-LEPTOSPIROSIS-FINAL-VERSION.pdf" \
    "$DOCS_DIR/clinical_protocols/PH_Leptospirosis_Children_2019.pdf"

download \
    "https://www.ncbi.nlm.nih.gov/books/NBK582435/pdf/Bookshelf_NBK582435.pdf" \
    "$DOCS_DIR/clinical_protocols/WHO_COVID19_Clinical_Mgmt_2023.pdf"

download \
    "https://nast.dost.gov.ph/images/pdf%20files/Publications/Monograph%20Series/NAST%20Monograph%20Series%207.pdf" \
    "$DOCS_DIR/clinical_protocols/PH_Rabies_Prevention_2020.pdf"

echo ""
echo "=== Clinical References ==="

download \
    "https://www.philhealth.gov.ph/partners/providers/pdf/PNF-EML_11022022.pdf" \
    "$DOCS_DIR/clinical_references/PH_National_Formulary_EML_2022.pdf"

download \
    "https://www.philhealth.gov.ph/partners/providers/pdf/PNF-EML_11022022.pdf" \
    "$DOCS_DIR/clinical_references/PH_Formulary_Primary_Healthcare.pdf"

download \
    "https://www.pidsphil.org/home/wp-content/uploads/2024/11/2025-PIDSP-Immunization-Calendar.pdf" \
    "$DOCS_DIR/clinical_references/PH_Immunization_Schedule_2024.pdf"

download \
    "https://platform.who.int/docs/default-source/mca-documents/policy-documents/plan-strategy/PHL-CC-10-01-PLAN-STRATEGY-2011-eng-MNCHN-Strategy-MOP.pdf" \
    "$DOCS_DIR/clinical_references/PH_MNCHN_Strategy_MOP.pdf"

download \
    "https://iris.who.int/bitstream/handle/10665/371090/WHO-MHP-HPS-EML-2023.02-eng.pdf" \
    "$DOCS_DIR/clinical_references/WHO_Essential_Medicines_2023.pdf"

echo ""
echo "=== Done ==="
echo "Guidelines: $(ls "$DOCS_DIR/clinical_guidelines/"*.pdf 2>/dev/null | wc -l) files"
echo "Protocols:  $(ls "$DOCS_DIR/clinical_protocols/"*.pdf 2>/dev/null | wc -l) files"
echo "References: $(ls "$DOCS_DIR/clinical_references/"*.pdf 2>/dev/null | wc -l) files"
