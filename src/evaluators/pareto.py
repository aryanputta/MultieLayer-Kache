"""
pareto.py — Pareto frontier analysis and statistical hypothesis testing.

:class:`ParetoAnalyzer` builds non-dominated frontiers for quality vs. memory
and latency vs. memory tradeoffs across policies.

:class:`StatisticalTester` provides paired tests, bootstrap CIs, and effect
size metrics for rigorous policy comparisons.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pareto analysis
# ---------------------------------------------------------------------------

class ParetoAnalyzer:
    """Accumulates per-policy results and computes Pareto frontiers.

    A point P dominates Q (for a maximisation problem) if P is at least as
    good as Q on all objectives and strictly better on at least one.

    For ``(quality, -memory)`` both dimensions should be maximised, so
    "better memory" means lower GPU usage.
    """

    def __init__(self) -> None:
        self._results: List[dict] = []

    def add_result(
        self,
        policy_name: str,
        memory_budget: float,
        context_length: int,
        quality_score: float,
        p95_latency_ms: float,
        gpu_mem_peak_mb: float,
        **kwargs,
    ) -> None:
        """Add a single benchmark result point."""
        self._results.append({
            "policy_name": policy_name,
            "memory_budget": memory_budget,
            "context_length": context_length,
            "quality_score": quality_score,
            "p95_latency_ms": p95_latency_ms,
            "gpu_mem_peak_mb": gpu_mem_peak_mb,
            **kwargs,
        })

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._results)

    def compute_pareto_frontier(
        self,
        x_metric: str = "gpu_mem_peak_mb",
        y_metric: str = "quality_score",
        higher_is_better_x: bool = False,
        higher_is_better_y: bool = True,
    ) -> pd.DataFrame:
        """Return the non-dominated set of points.

        Parameters
        ----------
        x_metric, y_metric:
            Columns from the results DataFrame.
        higher_is_better_x / higher_is_better_y:
            Whether larger values are preferred for each axis.
        """
        df = self.get_results_df()
        if df.empty:
            return df

        points = df[[x_metric, y_metric, "policy_name"]].dropna().to_dict("records")

        def dominates(a: dict, b: dict) -> bool:
            """Return True if point *a* dominates *b*."""
            ax, ay = a[x_metric], a[y_metric]
            bx, by = b[x_metric], b[y_metric]
            # Convert to maximisation
            ax_ = ax if higher_is_better_x else -ax
            bx_ = bx if higher_is_better_x else -bx
            ay_ = ay if higher_is_better_y else -ay
            by_ = by if higher_is_better_y else -by
            return (ax_ >= bx_ and ay_ >= by_) and (ax_ > bx_ or ay_ > by_)

        frontier = []
        for p in points:
            dominated = any(dominates(q, p) for q in points if q is not p)
            if not dominated:
                frontier.append(p)

        frontier_df = pd.DataFrame(frontier).sort_values(x_metric).reset_index(drop=True)
        return frontier_df

    def is_dominated(
        self,
        point: dict,
        frontier: List[dict],
        x_metric: str,
        y_metric: str,
        higher_is_better_x: bool = False,
        higher_is_better_y: bool = True,
    ) -> bool:
        """Check whether *point* is dominated by any point in *frontier*."""
        for f in frontier:
            ax = f[x_metric] if higher_is_better_x else -f[x_metric]
            bx = point[x_metric] if higher_is_better_x else -point[x_metric]
            ay = f[y_metric] if higher_is_better_y else -f[y_metric]
            by = point[y_metric] if higher_is_better_y else -point[y_metric]
            if (ax >= bx and ay >= by) and (ax > bx or ay > by):
                return True
        return False

    def compute_quality_per_gb(self, results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute quality / (GPU memory in GB) ratio per policy."""
        df = results if results is not None else self.get_results_df()
        df = df.copy()
        df["quality_per_gb"] = df["quality_score"] / (df["gpu_mem_peak_mb"] / 1024.0 + 1e-9)
        return df[["policy_name", "quality_score", "gpu_mem_peak_mb", "quality_per_gb"]]

    def generate_report(self) -> dict:
        """Produce a summary report across all accumulated results."""
        df = self.get_results_df()
        if df.empty:
            return {"error": "No results accumulated."}

        frontier = self.compute_pareto_frontier()
        policies = df["policy_name"].unique().tolist()
        per_policy = {}
        for p in policies:
            sub = df[df["policy_name"] == p]
            per_policy[p] = {
                "mean_quality": round(float(sub["quality_score"].mean()), 4),
                "mean_p95_ms":  round(float(sub["p95_latency_ms"].mean()), 2),
                "mean_gpu_mb":  round(float(sub["gpu_mem_peak_mb"].mean()), 1),
                "n_results":    int(len(sub)),
            }
        return {
            "total_results": len(df),
            "policies": policies,
            "pareto_frontier_size": len(frontier),
            "pareto_frontier_policies": frontier["policy_name"].tolist() if not frontier.empty else [],
            "per_policy": per_policy,
        }


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

class StatisticalTester:
    """Paired statistical tests for comparing policy evaluation scores."""

    # ------------------------------------------------------------------
    # Parametric
    # ------------------------------------------------------------------

    def paired_ttest(
        self, scores_a: List[float], scores_b: List[float]
    ) -> dict:
        """Two-sided paired t-test (assumes normal distribution of differences)."""
        a, b = np.array(scores_a), np.array(scores_b)
        t_stat, p_val = scipy_stats.ttest_rel(a, b)
        d = self.effect_size_cohens_d(scores_a, scores_b)
        return {
            "test": "paired_ttest",
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "effect_size_d": round(d, 4),
            "significant_at_05": bool(p_val < 0.05),
            "n": len(a),
        }

    # ------------------------------------------------------------------
    # Non-parametric
    # ------------------------------------------------------------------

    def wilcoxon_test(
        self, scores_a: List[float], scores_b: List[float]
    ) -> dict:
        """Wilcoxon signed-rank test (distribution-free paired comparison)."""
        a, b = np.array(scores_a), np.array(scores_b)
        try:
            stat, p_val = scipy_stats.wilcoxon(a, b)
        except ValueError as exc:
            return {"test": "wilcoxon", "error": str(exc)}
        d = self.effect_size_cohens_d(scores_a, scores_b)
        return {
            "test": "wilcoxon",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "effect_size_d": round(d, 4),
            "significant_at_05": bool(p_val < 0.05),
            "n": len(a),
        }

    def anova_test(self, groups: Dict[str, List[float]]) -> dict:
        """One-way ANOVA across multiple policy groups."""
        arrays = [np.array(v) for v in groups.values()]
        f_stat, p_val = scipy_stats.f_oneway(*arrays)
        return {
            "test": "one_way_anova",
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_at_05": bool(p_val < 0.05),
            "groups": list(groups.keys()),
        }

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        values: List[float],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean.

        Returns (lower, upper) bounds.
        """
        rng = np.random.default_rng(seed)
        arr = np.array(values)
        boot_means = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        alpha = 1.0 - ci
        lo = float(np.percentile(boot_means, 100 * alpha / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return lo, hi

    # ------------------------------------------------------------------
    # Effect size
    # ------------------------------------------------------------------

    def effect_size_cohens_d(
        self, a: List[float], b: List[float]
    ) -> float:
        """Cohen's d = (mean_a - mean_b) / pooled_std."""
        a_arr, b_arr = np.array(a), np.array(b)
        mean_diff = a_arr.mean() - b_arr.mean()
        pooled_std = np.sqrt((a_arr.std(ddof=1) ** 2 + b_arr.std(ddof=1) ** 2) / 2)
        if pooled_std == 0:
            return 0.0
        return float(mean_diff / pooled_std)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_comparison_table(
        self,
        results: Dict[str, Dict],
        baseline: str = "full_cache",
    ) -> pd.DataFrame:
        """Build a comparison table: policy × metric with significance flags.

        Parameters
        ----------
        results:
            Dict mapping policy_name → {metric_name → list_of_scores}.
        baseline:
            Policy to compare all others against.

        Returns
        -------
        pd.DataFrame with columns:
            policy_name, metric, mean, std, vs_baseline_delta,
            p_value, effect_size_d, significant
        """
        rows = []
        base_scores = results.get(baseline, {})

        for policy, metrics in results.items():
            for metric, scores in metrics.items():
                base = base_scores.get(metric, [])
                row = {
                    "policy_name": policy,
                    "metric": metric,
                    "mean": round(float(np.mean(scores)), 4),
                    "std": round(float(np.std(scores)), 4),
                    "n": len(scores),
                }
                if base and policy != baseline:
                    min_len = min(len(scores), len(base))
                    test = self.paired_ttest(scores[:min_len], base[:min_len])
                    lo, hi = self.bootstrap_ci(scores)
                    row.update({
                        "vs_baseline_delta": round(row["mean"] - float(np.mean(base)), 4),
                        "p_value": test["p_value"],
                        "effect_size_d": test["effect_size_d"],
                        "significant": test["significant_at_05"],
                        "ci_95_lo": round(lo, 4),
                        "ci_95_hi": round(hi, 4),
                    })
                rows.append(row)
        return pd.DataFrame(rows)
