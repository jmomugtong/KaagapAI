#!/usr/bin/env python
"""
Run MedQuery Evaluation Suite

Usage: python scripts/run_evals.py
"""

import json
import sys

from src.evaluation.runner import EvaluationRunner


def main() -> int:
    print("Running MedQuery Evaluation Suite...")
    runner = EvaluationRunner()
    results = runner.run()

    print(json.dumps(results, indent=2))

    if results.get("pass"):
        print("\nEvaluation PASSED")
        return 0
    else:
        print("\nEvaluation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
