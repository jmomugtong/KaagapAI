#!/usr/bin/env python
"""
Check Evaluation Thresholds

Usage: python scripts/check_thresholds.py
"""

import sys

from src.evaluation.runner import (
    HALLUCINATION_THRESHOLD,
    RETRIEVAL_RECALL_THRESHOLD,
    ROUGE_L_THRESHOLD,
    EvaluationRunner,
)


def main() -> int:
    runner = EvaluationRunner()
    results = runner.run()

    if results["status"] != "completed":
        print(f"Evaluation did not complete: {results.get('message', 'unknown')}")
        return 1

    metrics = results["metrics"]
    passed = True

    # ROUGE-L check
    rouge = metrics["rouge_l_avg"]
    if rouge >= ROUGE_L_THRESHOLD:
        print(f"ROUGE-L: {rouge:.4f} >= {ROUGE_L_THRESHOLD} PASS")
    else:
        print(f"ROUGE-L: {rouge:.4f} < {ROUGE_L_THRESHOLD} FAIL")
        passed = False

    # Hallucination rate check
    halluc = metrics["hallucination_rate"]
    if halluc < HALLUCINATION_THRESHOLD:
        print(f"Hallucination Rate: {halluc:.4f} < {HALLUCINATION_THRESHOLD} PASS")
    else:
        print(f"Hallucination Rate: {halluc:.4f} >= {HALLUCINATION_THRESHOLD} FAIL")
        passed = False

    # Retrieval recall check
    recall = metrics["retrieval_recall"]
    if recall >= RETRIEVAL_RECALL_THRESHOLD:
        print(f"Retrieval Recall: {recall:.4f} >= {RETRIEVAL_RECALL_THRESHOLD} PASS")
    else:
        print(f"Retrieval Recall: {recall:.4f} < {RETRIEVAL_RECALL_THRESHOLD} FAIL")
        passed = False

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
