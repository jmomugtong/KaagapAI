"""Generate sample clinical PDF documents for testing MedQuery."""

from pathlib import Path

from fpdf import FPDF


def create_pdf(filename: str, title: str, sections: list[tuple[str, str]]):
    """Create a PDF with a title and list of (heading, body) sections."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    for heading, body in sections:
        # Section heading
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Section body
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(4)

    output_dir = Path(__file__).resolve().parent.parent / "datasets" / "sample_docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_dir / filename))
    print(f"Created {output_dir / filename}")


# ============================================================
# 1. PROTOCOL — Post-Operative Pain Management Protocol
# ============================================================
create_pdf(
    "sample_protocol.pdf",
    "Post-Operative Pain Management Protocol v3.2",
    [
        (
            "1. Purpose and Scope",
            "This protocol establishes standardized pain management procedures for all "
            "post-surgical patients at Memorial Clinical Center. It applies to all surgical "
            "departments including orthopedics, general surgery, cardiothoracic surgery, and "
            "neurosurgery. The goal is to achieve adequate pain control (NRS score below 4) "
            "within 2 hours of recovery while minimizing opioid-related adverse events.",
        ),
        (
            "2. Initial Assessment",
            "All post-operative patients must receive a pain assessment using the Numeric "
            "Rating Scale (NRS 0-10) within 15 minutes of arrival in the post-anesthesia "
            "care unit (PACU). Reassessment is required every 30 minutes until pain is "
            "controlled (NRS < 4), then every 4 hours during the first 24 hours. Document "
            "pain location, quality, intensity, and aggravating or relieving factors. For "
            "patients unable to self-report, use the Behavioral Pain Scale (BPS).",
        ),
        (
            "3. Multimodal Analgesia - First Line",
            "Administer acetaminophen 1000mg intravenously every 6 hours as the foundation "
            "of multimodal therapy. Begin within 1 hour post-operatively unless contraindicated "
            "by hepatic impairment (ALT > 3x ULN) or documented allergy. Supplement with "
            "ketorolac 15-30mg IV every 6 hours for 48 hours maximum in patients without renal "
            "impairment (eGFR > 60), active bleeding risk, or history of peptic ulcer disease. "
            "This combination reduces opioid requirements by 30-40%.",
        ),
        (
            "4. Opioid Management",
            "For moderate to severe pain (NRS 5-7), administer morphine 2-4mg IV every "
            "3-4 hours as needed, or hydromorphone 0.2-0.6mg IV every 3-4 hours for patients "
            "with morphine intolerance. For severe pain (NRS 8-10), initiate patient-controlled "
            "analgesia (PCA) with morphine: demand dose 1mg, lockout interval 8 minutes, "
            "4-hour maximum 30mg. Monitor respiratory rate, sedation level (Pasero Opioid "
            "Sedation Scale), and oxygen saturation continuously for the first 24 hours. "
            "Naloxone 0.04mg IV must be available at bedside.",
        ),
        (
            "5. Knee Replacement Specific Protocol",
            "For total knee arthroplasty patients, add the following to the standard multimodal "
            "regimen: Adacel nerve block with ropivacaine 0.2% continuous infusion at 6mL/hr "
            "for 48 hours. Begin cryotherapy within 2 hours post-operatively, applying for 20 "
            "minutes every 2 hours during waking hours. Initiate continuous passive motion (CPM) "
            "machine on post-operative day 1, starting at 0-40 degrees and advancing 10 degrees "
            "daily as tolerated. Target range of motion: 0-90 degrees by discharge.",
        ),
        (
            "6. Transition to Oral Medications",
            "Transition from IV to oral analgesics when the patient tolerates oral intake and "
            "pain is controlled (NRS < 4) on current regimen. Standard oral regimen: "
            "acetaminophen 650mg every 6 hours plus ibuprofen 400mg every 8 hours with food. "
            "For breakthrough pain: oxycodone 5-10mg every 4-6 hours as needed. Provide no "
            "more than a 3-day supply of opioids at discharge with clear tapering instructions. "
            "Schedule follow-up pain assessment within 7 days of discharge.",
        ),
        (
            "7. Monitoring and Escalation",
            "If pain remains uncontrolled (NRS >= 7) after 2 hours of standard therapy, consult "
            "the Acute Pain Service. Red flags requiring immediate escalation: respiratory rate "
            "below 8 breaths per minute, oxygen saturation below 90%, Pasero sedation scale "
            "score of 3 or higher, signs of compartment syndrome (increasing pain despite "
            "adequate analgesia, paresthesias, pallor), or suspected allergic reaction. All "
            "adverse events must be reported in the electronic health record within 1 hour.",
        ),
    ],
)


# ============================================================
# 2. GUIDELINE — Hypertension Management Clinical Guideline
# ============================================================
create_pdf(
    "sample_guideline.pdf",
    "Clinical Guideline: Hypertension Management in Adults",
    [
        (
            "1. Introduction",
            "This clinical guideline provides evidence-based recommendations for the diagnosis, "
            "evaluation, and management of hypertension in adults aged 18 and older. It is "
            "based on the 2023 ACC/AHA guidelines and adapted for our institution. Hypertension "
            "affects approximately 47% of US adults and is the leading modifiable risk factor "
            "for cardiovascular disease, stroke, chronic kidney disease, and heart failure.",
        ),
        (
            "2. Diagnosis and Classification",
            "Blood pressure should be measured on at least 2 separate occasions before "
            "establishing a diagnosis. Use validated automated office blood pressure devices. "
            "Classification: Normal BP is less than 120/80 mmHg. Elevated BP is 120-129 "
            "systolic and less than 80 diastolic. Stage 1 hypertension is 130-139 systolic "
            "or 80-89 diastolic. Stage 2 hypertension is 140 or higher systolic or 90 or "
            "higher diastolic. Hypertensive crisis is greater than 180 systolic and/or "
            "greater than 120 diastolic with or without target organ damage.",
        ),
        (
            "3. Initial Evaluation",
            "All newly diagnosed hypertensive patients require: comprehensive metabolic panel "
            "(electrolytes, creatinine, eGFR, glucose, calcium), complete blood count, lipid "
            "panel, thyroid stimulating hormone, urinalysis with albumin-to-creatinine ratio, "
            "and 12-lead electrocardiogram. Consider echocardiography for patients with Stage 2 "
            "hypertension or evidence of target organ damage. Screen for secondary causes if "
            "age of onset is under 30, resistant hypertension, or sudden worsening of control.",
        ),
        (
            "4. Lifestyle Modifications",
            "Recommend lifestyle modifications for all patients with elevated BP or hypertension. "
            "DASH diet: rich in fruits, vegetables, whole grains, low-fat dairy, with reduced "
            "saturated fat and sodium. Sodium restriction to less than 2300mg per day, ideally "
            "less than 1500mg. Regular aerobic exercise: 150 minutes per week of moderate "
            "intensity or 75 minutes of vigorous intensity. Weight management: target BMI "
            "18.5-24.9 kg/m2. Limit alcohol to 2 drinks per day for men and 1 for women. "
            "Smoking cessation with pharmacotherapy support when indicated. Expected BP "
            "reduction with comprehensive lifestyle changes: 5-15 mmHg systolic.",
        ),
        (
            "5. Pharmacologic Treatment - First Line Agents",
            "Initiate pharmacologic therapy for Stage 1 hypertension with 10-year ASCVD risk "
            "of 10% or greater, or for all Stage 2 hypertension. First-line agents include: "
            "ACE inhibitors (lisinopril 10-40mg daily, enalapril 5-40mg daily) - preferred for "
            "patients with diabetes, CKD, or heart failure with reduced ejection fraction. "
            "ARBs (losartan 50-100mg daily, valsartan 80-320mg daily) - alternative for ACE "
            "inhibitor intolerance due to cough. Calcium channel blockers (amlodipine 2.5-10mg "
            "daily) - preferred for elderly patients and African American patients. Thiazide "
            "diuretics (chlorthalidone 12.5-25mg daily, hydrochlorothiazide 25-50mg daily). "
            "Do NOT combine ACE inhibitors with ARBs due to increased risk of hyperkalemia "
            "and renal impairment.",
        ),
        (
            "6. Treatment Targets and Monitoring",
            "Target BP for most adults: less than 130/80 mmHg. For patients aged 65 and older "
            "with significant comorbidities or frailty: less than 140/90 mmHg may be acceptable. "
            "Follow-up within 4 weeks of initiating or changing therapy. Check electrolytes and "
            "renal function 2-4 weeks after starting ACE inhibitor, ARB, or diuretic. If BP "
            "remains above target on single agent at optimal dose, add a second agent from a "
            "different class rather than maximizing single agent dose. Consider fixed-dose "
            "combination pills to improve adherence.",
        ),
        (
            "7. Resistant Hypertension",
            "Defined as BP above target despite optimal doses of 3 antihypertensive agents "
            "from different classes, including a diuretic. Before diagnosing resistant "
            "hypertension: confirm adherence using pharmacy refill records, rule out white coat "
            "effect with ambulatory BP monitoring, assess for interfering substances (NSAIDs, "
            "decongestants, oral contraceptives, stimulants). Fourth-line agent: spironolactone "
            "25-50mg daily. Refer to hypertension specialist if BP remains uncontrolled on "
            "4 agents.",
        ),
        (
            "8. Special Populations",
            "Pregnancy: discontinue ACE inhibitors and ARBs immediately. Use labetalol, "
            "nifedipine, or methyldopa. Target BP less than 140/90 mmHg. Diabetes mellitus: "
            "ACE inhibitor or ARB is first-line, especially with albuminuria. Chronic kidney "
            "disease: ACE inhibitor or ARB for patients with albuminuria. Monitor potassium "
            "closely. Heart failure with reduced ejection fraction: ACE inhibitor (or ARB), "
            "beta-blocker, and mineralocorticoid receptor antagonist. Coronary artery disease: "
            "beta-blocker and ACE inhibitor preferred. African American patients: calcium "
            "channel blocker or thiazide diuretic as initial therapy.",
        ),
    ],
)


# ============================================================
# 3. REFERENCE — Diabetes Mellitus Type 2 Clinical Reference
# ============================================================
create_pdf(
    "sample_reference.pdf",
    "Clinical Reference: Diabetes Mellitus Type 2 - Diagnosis and Management",
    [
        (
            "1. Epidemiology and Pathophysiology",
            "Type 2 diabetes mellitus (T2DM) affects over 37 million Americans and accounts "
            "for 90-95% of all diabetes cases. The disease is characterized by progressive "
            "insulin resistance and relative insulin deficiency due to beta-cell dysfunction. "
            "Major risk factors include obesity (BMI >= 25), family history, physical inactivity, "
            "history of gestational diabetes, polycystic ovary syndrome, and certain ethnicities "
            "(African American, Hispanic, Native American, Asian American). Complications include "
            "cardiovascular disease, nephropathy, retinopathy, neuropathy, and increased "
            "infection risk.",
        ),
        (
            "2. Diagnostic Criteria",
            "Diagnosis requires one of the following confirmed on two separate occasions unless "
            "unequivocal hyperglycemia is present: Fasting plasma glucose (FPG) of 126 mg/dL "
            "or higher. Oral glucose tolerance test (OGTT) 2-hour value of 200 mg/dL or higher. "
            "Hemoglobin A1c of 6.5% or higher using a NGSP-certified method. Random plasma "
            "glucose of 200 mg/dL or higher with classic symptoms (polyuria, polydipsia, "
            "unexplained weight loss). Prediabetes: FPG 100-125 mg/dL, OGTT 140-199 mg/dL, "
            "or A1c 5.7-6.4%.",
        ),
        (
            "3. Glycemic Targets",
            "Standard A1c target: less than 7.0% for most non-pregnant adults. This has been "
            "shown to reduce microvascular complications. More stringent target (A1c < 6.5%): "
            "consider for patients with short disease duration, long life expectancy, and no "
            "significant cardiovascular disease, if achievable without significant hypoglycemia. "
            "Less stringent target (A1c < 8.0%): consider for patients with history of severe "
            "hypoglycemia, limited life expectancy, advanced complications, extensive "
            "comorbidities, or long-standing diabetes. Fasting glucose target: 80-130 mg/dL. "
            "Post-prandial glucose target: less than 180 mg/dL (measured 1-2 hours after meal).",
        ),
        (
            "4. First-Line Therapy - Metformin",
            "Metformin is the preferred initial pharmacologic agent for T2DM. Starting dose: "
            "500mg once daily with the evening meal, titrate by 500mg weekly to target dose "
            "of 1000mg twice daily. Maximum dose: 2550mg daily in divided doses. Mechanism: "
            "reduces hepatic glucose production, improves insulin sensitivity, modestly reduces "
            "weight. Contraindications: eGFR below 30 mL/min (do not initiate), eGFR 30-45 "
            "(reduce dose to 1000mg daily, monitor renal function every 3 months). Common side "
            "effects: gastrointestinal (nausea, diarrhea, abdominal discomfort) - usually "
            "self-limited with dose titration. Extended-release formulation may improve GI "
            "tolerability. Monitor vitamin B12 levels annually.",
        ),
        (
            "5. Second-Line and Add-On Therapies",
            "If A1c remains above target after 3 months of metformin at maximum tolerated dose, "
            "add a second agent based on patient factors: SGLT2 inhibitors (empagliflozin 10-25mg, "
            "dapagliflozin 5-10mg) - preferred for patients with established cardiovascular "
            "disease, heart failure, or CKD. Provide cardiovascular and renal protection "
            "independent of glucose lowering. GLP-1 receptor agonists (semaglutide 0.25-1mg "
            "weekly, liraglutide 0.6-1.8mg daily, dulaglutide 0.75-4.5mg weekly) - preferred "
            "for patients needing significant A1c reduction or weight loss. DPP-4 inhibitors "
            "(sitagliptin 100mg daily) - well tolerated, weight neutral, but less potent. "
            "Sulfonylureas (glimepiride 1-4mg daily, glipizide 5-20mg daily) - effective and "
            "low cost, but risk of hypoglycemia and weight gain. Insulin: consider early if "
            "A1c above 10%, symptomatic hyperglycemia, or evidence of catabolism.",
        ),
        (
            "6. Insulin Therapy",
            "Basal insulin initiation: start with 10 units or 0.1-0.2 units/kg daily of "
            "insulin glargine, detemir, or degludec. Titrate by 2-4 units every 3 days to "
            "achieve fasting glucose target of 80-130 mg/dL. If A1c remains above target on "
            "basal insulin with fasting glucose at target, add prandial insulin: rapid-acting "
            "insulin (lispro, aspart, or glulisine) 4 units or 10% of basal dose before the "
            "largest meal. Adjust by 1-2 units every 3 days based on post-prandial glucose. "
            "Educate patients on injection technique, hypoglycemia recognition and treatment "
            "(15g fast-acting carbohydrate rule), sick day management, and blood glucose "
            "monitoring frequency (minimum fasting daily, pre-prandial if on prandial insulin).",
        ),
        (
            "7. Complication Screening Schedule",
            "Annual screening requirements for all T2DM patients: dilated eye exam by "
            "ophthalmologist (retinopathy screening), foot exam with monofilament testing and "
            "pedal pulse assessment (neuropathy and peripheral vascular disease), urine "
            "albumin-to-creatinine ratio and eGFR (nephropathy), lipid panel, comprehensive "
            "metabolic panel. Every visit: blood pressure measurement (target < 130/80 mmHg), "
            "weight and BMI, medication adherence and side effect review, hypoglycemia "
            "assessment. A1c measurement every 3 months if not at target, every 6 months "
            "if stable and at target.",
        ),
        (
            "8. Cardiovascular Risk Reduction",
            "Diabetes is a cardiovascular disease equivalent. All T2DM patients require "
            "aggressive cardiovascular risk management: statin therapy for all patients aged "
            "40-75 (moderate intensity: atorvastatin 10-20mg or rosuvastatin 5-10mg; high "
            "intensity for those with ASCVD or 10-year risk > 20%: atorvastatin 40-80mg or "
            "rosuvastatin 20-40mg). Aspirin 75-162mg daily for secondary prevention in patients "
            "with established ASCVD. For primary prevention, aspirin may be considered in "
            "patients with high cardiovascular risk after discussing bleeding risk. Blood "
            "pressure target: less than 130/80 mmHg. Smoking cessation counseling at every "
            "visit with pharmacotherapy (varenicline, bupropion, or nicotine replacement) "
            "offered to all current smokers.",
        ),
    ],
)

print("\nDone! Three sample PDFs created in datasets/sample_docs/")
