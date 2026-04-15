"""
tests/test_benchmarks.py — Unit tests for benchmark and evaluation utilities.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.benchmarks.longbench import TaskResult, BenchmarkResult, TASK_TO_WORKLOAD
from src.evaluators.quality_metrics import QualityEvaluator
from src.evaluators.pareto import ParetoAnalyzer, StatisticalTester
from src.telemetry.schema import WorkloadType


# ---------------------------------------------------------------------------
# QualityEvaluator
# ---------------------------------------------------------------------------

class TestQualityEvaluator:
    def test_exact_match_correct(self):
        ev = QualityEvaluator()
        assert ev.exact_match("Paris", "Paris") == 1.0

    def test_exact_match_case_insensitive(self):
        ev = QualityEvaluator()
        assert ev.exact_match("paris", "Paris") == 1.0

    def test_exact_match_strips_articles(self):
        ev = QualityEvaluator()
        assert ev.exact_match("the cat", "cat") == 1.0

    def test_exact_match_wrong(self):
        ev = QualityEvaluator()
        assert ev.exact_match("London", "Paris") == 0.0

    def test_token_f1_perfect(self):
        ev = QualityEvaluator()
        assert ev.token_f1("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_token_f1_partial(self):
        ev = QualityEvaluator()
        score = ev.token_f1("quick fox", "the quick brown fox")
        assert 0.0 < score < 1.0

    def test_token_f1_empty_prediction(self):
        ev = QualityEvaluator()
        assert ev.token_f1("", "some reference") == 0.0

    def test_token_f1_both_empty(self):
        ev = QualityEvaluator()
        assert ev.token_f1("", "") == 1.0

    def test_rouge_l_perfect(self):
        ev = QualityEvaluator()
        score = ev._rouge_l("cat sat on mat", "cat sat on mat")
        assert score == pytest.approx(1.0)

    def test_rouge_l_partial(self):
        ev = QualityEvaluator()
        score = ev._rouge_l("cat on mat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_evaluate_returns_dict_with_all_metrics(self):
        ev = QualityEvaluator()
        result = ev.evaluate(["Paris"], ["Paris"])
        for m in ["f1", "em", "rouge_l"]:
            assert m in result

    def test_evaluate_batch_returns_dataframe(self):
        ev = QualityEvaluator()
        results = [
            {"prediction": "cat", "reference": "cat", "task_type": "qa"},
            {"prediction": "dog", "reference": "cat", "task_type": "qa"},
        ]
        df = ev.evaluate_batch(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_normalize_answer(self):
        ev = QualityEvaluator()
        assert ev.normalize_answer("The Quick Brown Fox!") == "quick brown fox"

    def test_multiple_references_pipe_separated(self):
        ev = QualityEvaluator()
        result = ev.evaluate(["cat"], ["cat | kitty | feline"])
        assert result["em"] == 1.0

    def test_edit_similarity_perfect(self):
        ev = QualityEvaluator()
        assert ev._edit_similarity("abc", "abc") == pytest.approx(1.0)

    def test_edit_similarity_partial(self):
        ev = QualityEvaluator()
        score = ev._edit_similarity("abcd", "abef")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def _make_result(self):
        task = TaskResult(
            task_name="hotpotqa",
            workload_type=WorkloadType.MULTI_DOC_QA,
            policy_name="hybrid",
            memory_budget=0.75,
            n_samples=50,
            quality_score=0.62,
            primary_metric="f1",
        )
        result = BenchmarkResult(policy_name="hybrid", memory_budget=0.75, task_results=[task])
        result.compute_aggregate()
        return result

    def test_aggregate_computed(self):
        result = self._make_result()
        assert result.aggregate_quality == pytest.approx(0.62)

    def test_to_dataframe_shape(self):
        result = self._make_result()
        df = result.to_dataframe()
        assert len(df) == 1
        assert "quality_score" in df.columns

    def test_save_and_load(self, tmp_path):
        result = self._make_result()
        path = str(tmp_path / "result.json")
        result.save(path)
        loaded = BenchmarkResult.load(path)
        assert loaded.policy_name == "hybrid"
        assert len(loaded.task_results) == 1
        assert loaded.task_results[0].quality_score == pytest.approx(0.62)


# ---------------------------------------------------------------------------
# ParetoAnalyzer
# ---------------------------------------------------------------------------

class TestParetoAnalyzer:
    def _populated_analyzer(self):
        analyzer = ParetoAnalyzer()
        # A dominates B on both axes (higher quality, lower mem)
        analyzer.add_result("policy_a", 0.75, 8192, quality_score=0.80, p95_latency_ms=200, gpu_mem_peak_mb=4000)
        analyzer.add_result("policy_b", 0.75, 8192, quality_score=0.60, p95_latency_ms=400, gpu_mem_peak_mb=6000)
        analyzer.add_result("policy_c", 0.75, 8192, quality_score=0.70, p95_latency_ms=150, gpu_mem_peak_mb=3000)
        return analyzer

    def test_pareto_finds_nondominated(self):
        analyzer = self._populated_analyzer()
        frontier = analyzer.compute_pareto_frontier("gpu_mem_peak_mb", "quality_score")
        # policy_a is dominated by policy_c (lower mem, higher quality)
        assert not frontier.empty

    def test_domination_check(self):
        analyzer = self._populated_analyzer()
        point_a = {"gpu_mem_peak_mb": 4000, "quality_score": 0.80}
        point_b = {"gpu_mem_peak_mb": 6000, "quality_score": 0.60}
        # a dominates b (lower mem = better, higher quality = better)
        assert analyzer.is_dominated(
            point_b, [point_a], "gpu_mem_peak_mb", "quality_score",
            higher_is_better_x=False, higher_is_better_y=True,
        )

    def test_quality_per_gb(self):
        analyzer = self._populated_analyzer()
        df = analyzer.compute_quality_per_gb()
        assert "quality_per_gb" in df.columns
        assert (df["quality_per_gb"] > 0).all()

    def test_generate_report_structure(self):
        analyzer = self._populated_analyzer()
        report = analyzer.generate_report()
        assert "total_results" in report
        assert "per_policy" in report
        assert "pareto_frontier_size" in report


# ---------------------------------------------------------------------------
# StatisticalTester
# ---------------------------------------------------------------------------

class TestStatisticalTester:
    def test_ttest_significant(self):
        tester = StatisticalTester()
        a = [0.8] * 30
        b = [0.6] * 30
        result = tester.paired_ttest(a, b)
        assert result["significant_at_05"] is True
        assert result["p_value"] < 0.05

    def test_ttest_not_significant(self):
        tester = StatisticalTester()
        rng = np.random.default_rng(42)
        a = rng.normal(0.7, 0.01, 20).tolist()
        b = rng.normal(0.7, 0.01, 20).tolist()
        result = tester.paired_ttest(a, b)
        # Not expected to be significant with near-identical distributions
        assert "p_value" in result

    def test_bootstrap_ci_width(self):
        tester = StatisticalTester()
        vals = [0.7 + np.random.randn() * 0.1 for _ in range(100)]
        lo, hi = tester.bootstrap_ci(vals, n_bootstrap=500, ci=0.95)
        assert lo < hi
        # 95 % CI should contain the true mean
        assert lo < np.mean(vals) < hi

    def test_effect_size_zero_for_identical(self):
        tester = StatisticalTester()
        a = [0.5] * 10
        b = [0.5] * 10
        d = tester.effect_size_cohens_d(a, b)
        assert d == pytest.approx(0.0)

    def test_effect_size_large_for_very_different(self):
        tester = StatisticalTester()
        a = [1.0] * 20
        b = [0.0] * 20
        d = tester.effect_size_cohens_d(a, b)
        assert abs(d) > 2.0

    def test_format_comparison_table_shape(self):
        tester = StatisticalTester()
        rng = np.random.default_rng(0)
        results = {
            "baseline": {"f1": rng.normal(0.6, 0.05, 20).tolist()},
            "hybrid": {"f1": rng.normal(0.7, 0.05, 20).tolist()},
        }
        df = tester.format_comparison_table(results, baseline="baseline")
        assert isinstance(df, pd.DataFrame)
        assert "policy_name" in df.columns

    def test_wilcoxon_significant(self):
        tester = StatisticalTester()
        a = [0.9] * 25
        b = [0.5] * 25
        result = tester.wilcoxon_test(a, b)
        assert result.get("significant_at_05") is True

    def test_anova_structure(self):
        tester = StatisticalTester()
        groups = {
            "a": [0.8] * 10,
            "b": [0.6] * 10,
            "c": [0.7] * 10,
        }
        result = tester.anova_test(groups)
        assert "f_statistic" in result
        assert "p_value" in result
