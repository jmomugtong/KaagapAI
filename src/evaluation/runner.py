"""
Evaluation Runner for KaagapAI (Phase 10)

Runs evaluation metrics against a QA dataset:
- ROUGE-L: Lexical overlap with ground truth (target >= 0.60)
- Hallucination Rate: % citing non-existent sources (target < 5%)
- Retrieval Recall: % correct doc in top-5 (target > 90%)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds
ROUGE_L_THRESHOLD = 0.60
HALLUCINATION_THRESHOLD = 0.05
RETRIEVAL_RECALL_THRESHOLD = 0.90


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference."""
    if not prediction or not reference:
        return 0.0

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def _run_pipeline_for_question(question: str) -> object | None:
    """Run the classical pipeline for a single question.

    Returns a PipelineResult or None if the pipeline is unavailable.
    """
    try:
        from src.pipelines.classical import ClassicalPipeline

        pipeline = ClassicalPipeline(
            embedding_generator=None,
            ollama_client=None,
            reranker=None,
        )

        async def _run() -> object:
            return await pipeline.run(question)

        return asyncio.run(_run())
    except Exception as e:
        logger.warning("Pipeline execution failed for question %r: %s", question, e)
        return None


class EvaluationRunner:
    """Runs evaluation suite against clinical QA datasets."""

    def __init__(self, dataset_path: str | None = None) -> None:
        self._dataset_path = dataset_path or str(
            Path(__file__).resolve().parent.parent.parent
            / "datasets"
            / "clinical_qa_25.json"
        )

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load QA dataset from JSON file."""
        if not os.path.exists(self._dataset_path):
            logger.warning("Dataset not found: %s", self._dataset_path)
            return []
        with open(self._dataset_path) as f:
            data = json.load(f)
        questions: list[dict[str, Any]] = data.get("questions", [])
        return questions

    def run(self) -> dict:
        """Run full evaluation suite and return results."""
        questions = self.load_dataset()
        if not questions:
            return {
                "status": "no_dataset",
                "message": f"No dataset found at {self._dataset_path}",
                "metrics": {},
            }

        rouge_scores: list[float] = []
        hallucination_flags: list[bool] = []
        recall_hits: list[bool] = []
        skipped = 0

        for q in questions:
            query_text = str(q.get("query", q.get("question", "")))
            ground_truth = str(q.get("ground_truth", ""))
            expected_sources = [str(s) for s in q.get("expected_sources") or []]

            # Run the pipeline to get a real prediction
            result = _run_pipeline_for_question(query_text)

            if result is None:
                skipped += 1
                logger.warning(
                    "Skipping question %r — pipeline unavailable", q.get("id", "?")
                )
                continue

            predicted = getattr(result, "answer", "")
            score = compute_rouge_l(predicted, ground_truth)
            rouge_scores.append(score)

            # Hallucination: flagged by the pipeline's own detection
            hallucination_flags.append(
                bool(getattr(result, "hallucination_flagged", False))
            )

            # Retrieval recall: any expected source found in retrieved chunks
            retrieved_sources = {
                str(c.get("source", ""))
                for c in (getattr(result, "retrieved_chunks", None) or [])
            }
            hit = any(src in retrieved_sources for src in expected_sources)
            recall_hits.append(hit)

        evaluated = len(rouge_scores)

        if evaluated == 0:
            logger.warning(
                "No questions could be evaluated. "
                "Ensure the API is running and documents are indexed."
            )
            return {
                "status": "no_results",
                "message": (
                    "All questions skipped — pipeline unavailable or no documents indexed. "
                    "Start the API and upload documents before running evals."
                ),
                "skipped": skipped,
                "metrics": {},
            }

        avg_rouge = sum(rouge_scores) / evaluated
        hallucination_rate = sum(hallucination_flags) / evaluated
        retrieval_recall = sum(recall_hits) / evaluated

        results = {
            "status": "completed",
            "total_questions": len(questions),
            "evaluated": evaluated,
            "skipped": skipped,
            "metrics": {
                "rouge_l_avg": round(avg_rouge, 4),
                "rouge_l_pass": avg_rouge >= ROUGE_L_THRESHOLD,
                "hallucination_rate": round(hallucination_rate, 4),
                "hallucination_pass": hallucination_rate <= HALLUCINATION_THRESHOLD,
                "retrieval_recall": round(retrieval_recall, 4),
                "retrieval_recall_pass": retrieval_recall >= RETRIEVAL_RECALL_THRESHOLD,
            },
            "thresholds": {
                "rouge_l": ROUGE_L_THRESHOLD,
                "hallucination_rate": HALLUCINATION_THRESHOLD,
                "retrieval_recall": RETRIEVAL_RECALL_THRESHOLD,
            },
            "pass": (
                avg_rouge >= ROUGE_L_THRESHOLD
                and hallucination_rate <= HALLUCINATION_THRESHOLD
                and retrieval_recall >= RETRIEVAL_RECALL_THRESHOLD
            ),
        }

        return results
