"""
Agentic RAG Prompt Templates

Prompt templates for the agent's classification, decomposition,
self-reflection, and synthesis stages.
"""

CLASSIFY_PROMPT = """Classify this clinical query into exactly one category:

- SIMPLE: Direct factual lookup (e.g., "What is the dosage for amoxicillin?")
- COMPARATIVE: Comparing two or more things (e.g., "Compare pain protocols for knee vs hip")
- MULTI_STEP: Requires combining info from multiple sources (e.g., "What are the contraindications for patients with both diabetes and hypertension?")
- TEMPORAL: About changes over time (e.g., "How has the antibiotic protocol changed?")
- GENERAL: General medical knowledge that doesn't need specific clinical documents (e.g., "What is hypertension?", "Explain the mechanism of action of metformin")

Query: {question}

Respond with ONLY the category name (SIMPLE, COMPARATIVE, MULTI_STEP, TEMPORAL, or GENERAL)."""

VALID_QUERY_TYPES = {"SIMPLE", "COMPARATIVE", "MULTI_STEP", "TEMPORAL", "GENERAL"}

DECOMPOSE_PROMPT = """Break this clinical query into {n} focused sub-queries that can each be answered independently.

Original query: {question}
Query type: {query_type}

Return each sub-query on its own line, numbered 1-{n}. No other text."""

# How many sub-queries to generate per query type
DECOMPOSE_COUNTS = {
    "SIMPLE": 1,
    "COMPARATIVE": 2,
    "MULTI_STEP": 3,
    "TEMPORAL": 2,
    "GENERAL": 0,  # No retrieval needed
}

MAX_SUB_QUERIES = 4

# Direct-answer prompt for GENERAL queries (no retrieval needed)
GENERAL_ANSWER_PROMPT = """You are a medical information assistant. Answer this general medical knowledge question concisely and accurately.

Question: {question}

IMPORTANT DISCLAIMERS:
- Clearly state this is general medical knowledge, NOT from specific clinical documents
- Recommend consulting specific institutional protocols for clinical decisions
- Do not provide specific dosages or treatment plans — only general educational information
- Assign a confidence score on the last line as "Confidence: X.XX"

RESPONSE:
"""

REFLECT_PROMPT = """You answered a clinical question. Evaluate your answer:

ORIGINAL QUESTION: {question}
QUERY TYPE: {query_type}
YOUR ANSWER: {answer}
CONFIDENCE: {confidence}

Does this answer fully address the original question?
- If YES: respond "SUFFICIENT"
- If NO: respond "INSUFFICIENT: <what's missing>" followed by a refined search query on the next line

Respond with ONLY "SUFFICIENT" or "INSUFFICIENT: ..." and optionally the refined query."""


def build_synthesis_prompt(
    question: str,
    context: str,
    query_type: str,
) -> str:
    """Build a synthesis prompt with type-specific instructions."""
    type_instructions = {
        "COMPARATIVE": (
            "Structure your answer as a clear comparison. "
            "Address each item separately, then summarize key differences."
        ),
        "MULTI_STEP": (
            "Connect the information from different sources to form a complete answer."
        ),
        "TEMPORAL": (
            "Present the information chronologically. "
            "Highlight what changed and when."
        ),
        "SIMPLE": "",
    }

    specific = type_instructions.get(query_type, "")
    if specific:
        specific = f"\nSPECIAL INSTRUCTIONS:\n{specific}\n"

    return f"""You are a medical information assistant with strict grounding rules. You have been given context retrieved through multiple targeted searches to answer a {query_type} question. You MUST NOT use any knowledge outside the provided context.

CONTEXT:
{context}

ORIGINAL QUESTION:
{question}
{specific}
STRICT GROUNDING RULES:
1. Answer ONLY using information explicitly stated in the CONTEXT above
2. Quote or closely paraphrase exact phrases from the sources — do not invent medical facts
3. Cite EVERY factual claim using [Document Name, Section, p. Page] format
4. If the context does NOT contain enough information, say: "The indexed documents do not contain sufficient information to fully answer this question." Then share what IS available.
5. Do NOT add medical advice, dosages, drug interactions, or treatment recommendations not in the context
6. For comparative questions, use clear structure (bullet points or table)
7. If sources conflict, explicitly note the discrepancy
8. Assign a confidence score (0.0-1.0) on the last line as "Confidence: X.XX"
   - 0.90-1.0: Direct, explicit answer found in context
   - 0.70-0.89: Partially supported, some inference
   - Below 0.70: Weak support

RESPONSE:
"""
