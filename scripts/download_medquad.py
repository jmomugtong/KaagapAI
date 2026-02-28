"""Download MedQuAD dataset from HuggingFace and convert to evaluation format.

Produces datasets/clinical_qa_50.json with 50 curated Q&A pairs across
diverse medical specialties, matching the V1 evaluation dataset format
defined in the evaluation framework.

Requires: pip install datasets
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

# Reproducible selection
random.seed(42)

# Target output
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "clinical_qa_50.json"

# Map MedQuAD document_source → our category names
# Actual source names from lavita/MedQuAD dataset
SOURCE_TO_CATEGORY = {
    "GHR": "genetics",
    "GARD": "rare_diseases",
    "NIDDK": "endocrine",
    "NINDS": "neurology",
    "MPlusHealthTopics": "general_medicine",
    "NIHSeniorHealth": "geriatrics",
    "CancerGov": "oncology",
    "NHLBI": "cardiology",
    "CDC": "infectious_disease",
}

# Difficulty heuristic based on answer length (characters)
DIFFICULTY_THRESHOLDS = {
    "easy": 500,  # short, factual answers
    "medium": 1500,  # moderate detail
    # anything longer → "hard"
}


def assign_difficulty(answer_len: int) -> str:
    """Assign difficulty based on answer length."""
    if answer_len <= DIFFICULTY_THRESHOLDS["easy"]:
        return "easy"
    elif answer_len <= DIFFICULTY_THRESHOLDS["medium"]:
        return "medium"
    return "hard"


def main() -> None:
    print("Loading MedQuAD dataset from HuggingFace (lavita/MedQuAD)...")
    ds = load_dataset("lavita/MedQuAD", split="train")
    print(f"Loaded {len(ds)} total Q&A pairs")

    # Group by document_source so we can sample across specialties
    by_source: dict[str, list] = {}
    for row in ds:
        source = row["document_source"]
        # Filter: require both question and answer, skip very short answers
        if (
            row["question"]
            and row["answer"]
            and len(row["answer"].strip()) >= 50
            and len(row["question"].strip()) >= 15
        ):
            by_source.setdefault(source, []).append(row)

    print(f"Found {len(by_source)} document sources:")
    for src, rows in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"  {src}: {len(rows)} valid Q&A pairs")

    # Sample evenly across sources, then fill remaining from largest sources
    target_count = 50
    per_source = max(1, target_count // len(by_source))
    selected: list = []
    remaining_sources: list[tuple[str, list]] = []

    for source, rows in by_source.items():
        sample_n = min(per_source, len(rows))
        sampled = random.sample(rows, sample_n)
        selected.extend(sampled)
        leftover = [r for r in rows if r not in sampled]
        if leftover:
            remaining_sources.append((source, leftover))

    # Fill to exactly 50 from remaining pools
    random.shuffle(remaining_sources)
    while len(selected) < target_count and remaining_sources:
        source, leftovers = remaining_sources.pop(0)
        pick = random.choice(leftovers)
        selected.append(pick)
        leftovers.remove(pick)
        if leftovers:
            remaining_sources.append((source, leftovers))

    selected = selected[:target_count]
    random.shuffle(selected)

    # Convert to our evaluation format
    questions = []
    for i, row in enumerate(selected, start=1):
        qid = f"q{i:03d}"
        source = row["document_source"]
        answer_text = row["answer"].strip()
        # Truncate very long ground truths to first 500 chars for eval
        if len(answer_text) > 500:
            # Find a sentence boundary near 500 chars
            cutoff = answer_text[:500].rfind(". ")
            if cutoff > 200:
                answer_text = answer_text[: cutoff + 1]
            else:
                answer_text = answer_text[:500] + "..."

        questions.append(
            {
                "id": qid,
                "query": row["question"].strip(),
                "ground_truth": answer_text,
                "expected_sources": [source],
                "category": SOURCE_TO_CATEGORY.get(source, "general_medicine"),
                "difficulty": assign_difficulty(len(row["answer"])),
            }
        )

    # Build output JSON
    output = {
        "version": "1.0",
        "description": "Clinical QA evaluation dataset (50 questions) - sourced from MedQuAD (NIH)",
        "source": "lavita/MedQuAD on HuggingFace (NIH Medical Question-Answer pairs)",
        "questions": questions,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(questions)} Q&A pairs to {OUTPUT_FILE}")

    # Summary by category
    cats: dict[str, int] = {}
    diffs: dict[str, int] = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
        diffs[q["difficulty"]] = diffs.get(q["difficulty"], 0) + 1

    print("\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print("\nBy difficulty:")
    for diff, count in sorted(diffs.items()):
        print(f"  {diff}: {count}")


if __name__ == "__main__":
    main()
