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
GENERAL_ANSWER_PROMPT = """Answer this general medical knowledge question briefly. State that this is general knowledge, not from clinical documents. Do not give specific dosages. End with "Confidence: X.XX"

Question: {question}

Answer:
"""

REFLECT_PROMPT = """Does this answer fully address the question? Reply ONLY "SUFFICIENT" or "INSUFFICIENT: <what's missing>" with a refined search query on the next line.

Question: {question}
Type: {query_type}
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

    return f"""Answer the question using ONLY the context below. Do not add outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}
{specific}
RULES:
- Use only facts from the context
- Cite sources as [Document Name, Section, p. Page]
- If the context lacks enough information, say so and share what IS available
- Do not invent dosages, drugs, or treatments not in the context
- End with "Confidence: X.XX" (0.0-1.0)

ANSWER:
"""
