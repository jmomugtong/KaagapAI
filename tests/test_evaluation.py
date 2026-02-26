"""
Tests for KaagapAI Evaluation Framework (Phase 10)
"""

import pytest

from src.evaluation.runner import ROUGE_L_THRESHOLD, EvaluationRunner, compute_rouge_l


class TestRougeL:
    """Tests for ROUGE-L computation."""

    @pytest.mark.unit
    def test_identical_strings(self):
        score = compute_rouge_l("hello world", "hello world")
        assert score == 1.0

    @pytest.mark.unit
    def test_no_overlap(self):
        score = compute_rouge_l("hello world", "foo bar baz")
        assert score == 0.0

    @pytest.mark.unit
    def test_partial_overlap(self):
        score = compute_rouge_l(
            "acetaminophen 1000mg for pain",
            "acetaminophen 1000mg every 6 hours for pain management",
        )
        assert 0.0 < score < 1.0

    @pytest.mark.unit
    def test_empty_prediction(self):
        score = compute_rouge_l("", "some reference text")
        assert score == 0.0

    @pytest.mark.unit
    def test_empty_reference(self):
        score = compute_rouge_l("some prediction", "")
        assert score == 0.0

    @pytest.mark.unit
    def test_both_empty(self):
        score = compute_rouge_l("", "")
        assert score == 0.0


class TestEvaluationRunner:
    """Tests for the evaluation runner."""

    @pytest.mark.unit
    def test_run_with_dataset(self):
        runner = EvaluationRunner()
        results = runner.run()
        assert results["status"] in ("completed", "no_dataset")

    @pytest.mark.unit
    def test_run_with_missing_dataset(self):
        runner = EvaluationRunner(dataset_path="/nonexistent/path.json")
        results = runner.run()
        assert results["status"] == "no_dataset"

    @pytest.mark.unit
    def test_load_dataset(self):
        runner = EvaluationRunner()
        questions = runner.load_dataset()
        # Should load the 25-question dataset
        if questions:
            assert len(questions) == 25
            assert "query" in questions[0]
            assert "ground_truth" in questions[0]

    @pytest.mark.unit
    def test_results_contain_metrics(self):
        runner = EvaluationRunner()
        results = runner.run()
        if results["status"] == "completed":
            assert "metrics" in results
            assert "rouge_l_avg" in results["metrics"]
            assert "hallucination_rate" in results["metrics"]
            assert "retrieval_recall" in results["metrics"]

    @pytest.mark.unit
    def test_results_contain_thresholds(self):
        runner = EvaluationRunner()
        results = runner.run()
        if results["status"] == "completed":
            assert "thresholds" in results
            assert results["thresholds"]["rouge_l"] == ROUGE_L_THRESHOLD
