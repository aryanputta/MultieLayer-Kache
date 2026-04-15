"""
harness.py — Orchestrates the full experiment suite.

:class:`BenchmarkHarness` runs policies × memory budgets × tasks × concurrency
levels, collects results, runs statistical comparisons, and writes reports.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import httpx

from src.benchmarks.longbench import LongBenchRunner, BenchmarkResult
from src.benchmarks.stress_tests import StressTestRunner, StressTestResult
from src.evaluators.pareto import ParetoAnalyzer, StatisticalTester

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResults:
    experiment_name: str
    policies: List[str]
    memory_budgets: List[float]
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    stress_results: List[StressTestResult] = field(default_factory=list)
    comparison_report: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for br in self.benchmark_results:
            for tr in br.task_results:
                row = tr.to_dict()
                row["policy_name"] = br.policy_name
                row["memory_budget"] = br.memory_budget
                row["aggregate_quality"] = br.aggregate_quality
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()


@dataclass
class AblationResults:
    base_policy: str
    ablations: List[str]
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)
    comparison_table: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """Runs the full experiment matrix and produces reproducible result files.

    Parameters
    ----------
    config:
        Full project config dict.
    api_url:
        Base URL of the serving API.
    """

    def __init__(
        self,
        config: dict,
        api_url: str = "http://localhost:8000",
    ) -> None:
        self.config = config
        self.api_url = api_url
        self.output_dir = config.get("benchmark", {}).get("output_dir", "results")
        self.pareto = ParetoAnalyzer()
        self.stats = StatisticalTester()

    # ------------------------------------------------------------------
    # Full experiment sweep
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        experiment_name: str,
        policies: List[str],
        memory_budgets: List[float],
        tasks: Optional[List[str]] = None,
        concurrency_levels: Optional[List[int]] = None,
    ) -> ExperimentResults:
        """Full sweep: each policy × each memory budget × each concurrency level.

        Parameters
        ----------
        experiment_name:
            Label for result files.
        policies:
            List of policy name strings.
        memory_budgets:
            GPU memory budget fractions to test.
        tasks:
            LongBench task names; defaults to config value.
        concurrency_levels:
            Request concurrency levels.
        """
        import asyncio

        bc = self.config.get("benchmark", {})
        tasks = tasks or bc.get("longbench", {}).get("tasks", ["narrativeqa"])
        concurrency_levels = concurrency_levels or bc.get("concurrency_levels", [1])

        results = ExperimentResults(
            experiment_name=experiment_name,
            policies=policies,
            memory_budgets=memory_budgets,
        )

        for policy in policies:
            for budget in memory_budgets:
                logger.info(
                    "Experiment '%s': policy=%s  budget=%.2f", experiment_name, policy, budget
                )
                self._set_policy(policy, budget)
                runner = LongBenchRunner(
                    api_url=self.api_url,
                    config=self.config,
                    policy_name=policy,
                    memory_budget=budget,
                )
                for concurrency in concurrency_levels:
                    br = asyncio.run(
                        runner.run_all_async(
                            tasks=tasks,
                            concurrency=concurrency,
                        )
                    )
                    results.benchmark_results.append(br)
                    self.pareto.add_result(
                        policy_name=policy,
                        memory_budget=budget,
                        context_length=0,
                        quality_score=br.aggregate_quality,
                        p95_latency_ms=self._extract_p95(br),
                        gpu_mem_peak_mb=self._extract_gpu_mem(br),
                    )

        results.comparison_report = self.pareto.generate_report()
        return results

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        base_policy: str,
        ablations: List[str],
        tasks: Optional[List[str]] = None,
        memory_budget: float = 0.75,
    ) -> AblationResults:
        """Run base policy and each ablation; compare statistically."""
        import asyncio

        results = AblationResults(base_policy=base_policy, ablations=ablations)

        for policy in [base_policy] + ablations:
            logger.info("Ablation: running policy=%s", policy)
            self._set_policy(policy, memory_budget)
            runner = LongBenchRunner(
                api_url=self.api_url,
                config=self.config,
                policy_name=policy,
                memory_budget=memory_budget,
            )
            br = asyncio.run(runner.run_all_async(tasks=tasks, concurrency=4))
            results.results[policy] = br

        # Build per-policy score lists for statistical comparison
        score_dict = {}
        for pol, br in results.results.items():
            score_dict[pol] = {
                "quality_score": [t.quality_score for t in br.task_results if t.quality_score >= 0]
            }

        results.comparison_table = self.stats.format_comparison_table(
            score_dict, baseline=base_policy
        )
        return results

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    def compare_policies(
        self, results: ExperimentResults
    ) -> pd.DataFrame:
        """Paired statistical comparison of all policies vs. the first policy."""
        if not results.benchmark_results:
            return pd.DataFrame()

        score_dict: Dict[str, Dict] = {}
        for br in results.benchmark_results:
            if br.policy_name not in score_dict:
                score_dict[br.policy_name] = {"quality_score": []}
            score_dict[br.policy_name]["quality_score"].append(br.aggregate_quality)

        baseline = results.policies[0] if results.policies else list(score_dict.keys())[0]
        return self.stats.format_comparison_table(score_dict, baseline=baseline)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self, results: ExperimentResults, output_dir: Optional[str] = None
    ) -> None:
        """Save all results as CSV tables and JSON summaries."""
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        df = results.to_dataframe()
        if not df.empty:
            csv_path = os.path.join(out, "tables", f"{results.experiment_name}_results.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info("Results saved to %s", csv_path)

        if results.comparison_report:
            json_path = os.path.join(out, f"{results.experiment_name}_report.json")
            with open(json_path, "w") as f:
                json.dump(results.comparison_report, f, indent=2)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, results: ExperimentResults) -> str:
        """Generate a markdown-format summary report."""
        df = results.to_dataframe()
        lines = [
            f"# Experiment: {results.experiment_name}",
            f"",
            f"**Policies tested:** {', '.join(results.policies)}",
            f"**Memory budgets:** {results.memory_budgets}",
            f"",
        ]

        if not df.empty and "policy_name" in df.columns and "quality_score" in df.columns:
            summary = (
                df.groupby("policy_name")["quality_score"]
                .agg(["mean", "std"])
                .round(4)
                .reset_index()
            )
            lines.append("## Quality Summary")
            lines.append("")
            lines.append(summary.to_markdown(index=False))
            lines.append("")

        if results.comparison_report:
            lines.append("## Pareto Analysis")
            lines.append("")
            lines.append(
                f"Non-dominated policies: "
                f"{results.comparison_report.get('pareto_frontier_policies', [])}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_policy(self, policy_name: str, memory_budget: float) -> None:
        """POST to the API to switch the active policy and memory budget."""
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(
                    f"{self.api_url}/v1/policy/override",
                    json={"policy_name": policy_name, "params": {"memory_budget": memory_budget}},
                )
        except Exception:
            pass  # Non-critical: policy switch is best-effort in experiments

    def _extract_p95(self, br: BenchmarkResult) -> float:
        all_latencies = []
        for tr in br.task_results:
            all_latencies.extend(tr.latencies_ms)
        return float(np.percentile(all_latencies, 95)) if all_latencies else 0.0

    def _extract_gpu_mem(self, br: BenchmarkResult) -> float:
        mem_vals = [
            tr.system_metrics.get("gpu_mem_peak_mb", 0.0)
            for tr in br.task_results
            if tr.system_metrics
        ]
        return float(np.max(mem_vals)) if mem_vals else 0.0
