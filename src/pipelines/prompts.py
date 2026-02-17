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

Query: {question}

Respond with ONLY the category name (SIMPLE, COMPARATIVE, MULTI_STEP, or TEMPORAL)."""

VALID_QUERY_TYPES = {"SIMPLE", "COMPARATIVE", "MULTI_STEP", "TEMPORAL"}

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
}

MAX_SUB_QUERIES = 4

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

    return f"""You are a medical information assistant. You have been given context retrieved through multiple targeted searches to answer a {query_type} question.

CONTEXT:
{context}

ORIGINAL QUESTION:
{question}
{specific}
INSTRUCTIONS:
1. Synthesize information from ALL provided context sections
2. Cite sources using [Document Name, Section, p. Page] format
3. For comparative questions, use clear structure (e.g., bullet points or a table)
4. If uncertain, state your limitations clearly
5. Assign a confidence score (0.0-1.0) on the last line as "Confidence: X.XX"

RESPONSE:
"""
