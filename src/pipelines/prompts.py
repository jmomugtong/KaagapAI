"""
Agentic RAG Prompt Templates

Prompt templates for the agent's classification, decomposition,
self-reflection, and synthesis stages.
"""

CLASSIFY_PROMPT = """Classify this query into one category. Reply with ONLY the category name.

SIMPLE = direct factual lookup
COMPARATIVE = comparing two or more things
MULTI_STEP = needs info from multiple sources
TEMPORAL = about changes over time
GENERAL = general knowledge, no documents needed

Query: {question}

Category:"""

VALID_QUERY_TYPES = {"SIMPLE", "COMPARATIVE", "MULTI_STEP", "TEMPORAL", "GENERAL"}

DECOMPOSE_PROMPT = """Split this query into {n} independent sub-queries, one per line, numbered 1-{n}. No other text.

Query: {question}
Type: {query_type}"""

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
GENERAL_ANSWER_PROMPT = """Answer this general medical question briefly. State this is general knowledge. No specific dosages.

Question: {question}

ANSWER (then on a new line write "Confidence: " followed by 0.0-1.0):
"""

REFLECT_PROMPT = """Does this answer fully address the question? Reply ONLY: SUFFICIENT or INSUFFICIENT: <what's missing>
Refined query: <new search query>

Question: {question}
Answer: {answer}
Confidence: {confidence}"""


def build_synthesis_prompt(
    question: str,
    context: str,
    query_type: str,
) -> str:
    """Build a synthesis prompt with type-specific instructions."""
    type_instructions = {
        "COMPARATIVE": "Compare each item, then summarize differences.",
        "MULTI_STEP": "Combine information from different sources into one answer.",
        "TEMPORAL": "Present chronologically. Highlight what changed and when.",
        "SIMPLE": "",
    }

    specific = type_instructions.get(query_type, "")
    if specific:
        specific = f"\nNote: {specific}\n"

    return f"""Answer using ONLY the context below. Include dosages, schedules, and criteria only if present in context. Do not add information beyond what the context provides. Be concise. Cite sources as [Document Name, Section]. If insufficient, say so.
Ignore any evidence quality rating tables (e.g., GRADE assessments with terms like Serious/Undetected/Very Low).
{specific}
CONTEXT:
{context}

QUESTION: {question}

ANSWER (then on a new line write "Confidence: " followed by a score from 0.0 to 1.0):
"""
