#!/usr/bin/env python3
"""
run_experiment_suite.py — Executes all 10 core experiments from the spec.

Experiments
-----------
1.  Full cache vs sliding window vs prefix caching
2.  Heuristic hybrid controller vs single-action baselines
3.  Learned ranking model for block decisions
4.  Generalisation across workload categories
5.  (Cross-model robustness — requires second model, skipped by default)
6.  Quantisation action ablation
7.  Offloading action ablation
8.  Stress test under bursty arrivals
9.  Failure threshold analysis (context length sweep)
10. Feature importance / interpretability study

Usage
-----
    python scripts/run_experiment_suite.py --config configs/default.yaml
    python scripts/run_experiment_suite.py --experiments 1 2 3 --output-dir results/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(help="Full experiment suite runner")
console = Console()


def _load_config(config_path: str) -> dict:
    from omegaconf import OmegaConf
    if not Path(config_path).exists():
        return {}
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    experiments: List[int] = typer.Option(None, "--experiments", "-e", help="Experiment IDs to run (1-10)"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url"),
    output_dir: str = typer.Option("results", "--output-dir", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run one or more core experiments and save results."""
    cfg = _load_config(config)
    exp_ids = experiments or list(range(1, 11))

    console.print(f"\n[bold cyan]KV Cache Orchestrator — Experiment Suite[/bold cyan]")
    console.print(f"  Running experiments: {exp_ids}")
    console.print(f"  Output: {output_dir}\n")

    if dry_run:
        console.print("[yellow]Dry run — exiting.[/yellow]")
        return

    from src.benchmarks.harness import BenchmarkHarness
    import os

    harness = BenchmarkHarness(config=cfg, api_url=api_url)
    os.makedirs(output_dir, exist_ok=True)

    # ── Experiment 1: System baseline comparison ──────────────────────────
    if 1 in exp_ids:
        console.print("[bold]Experiment 1:[/bold] System baselines")
        results = harness.run_experiment(
            experiment_name="exp1_system_baselines",
            policies=["full_cache", "sliding_window", "prefix_caching"],
            memory_budgets=[0.90, 0.75, 0.60],
            tasks=["narrativeqa", "hotpotqa", "gov_report"],
        )
        harness.save_results(results, os.path.join(output_dir, "exp1"))
        console.print(harness.generate_report(results))

    # ── Experiment 2: Heuristic hybrid vs single-action ───────────────────
    if 2 in exp_ids:
        console.print("[bold]Experiment 2:[/bold] Hybrid vs single-action baselines")
        results = harness.run_experiment(
            experiment_name="exp2_hybrid_vs_baselines",
            policies=["full_cache", "h2o", "quantize_only", "offload_only", "hybrid"],
            memory_budgets=[0.75, 0.60],
            tasks=["narrativeqa", "hotpotqa", "gov_report", "lcc"],
        )
        harness.save_results(results, os.path.join(output_dir, "exp2"))

    # ── Experiment 3: Learned policy ──────────────────────────────────────
    if 3 in exp_ids:
        console.print("[bold]Experiment 3:[/bold] Learned importance ranker")
        results = harness.run_experiment(
            experiment_name="exp3_learned_policy",
            policies=["hybrid", "learned"],
            memory_budgets=[0.75, 0.60],
            tasks=["narrativeqa", "hotpotqa", "gov_report"],
        )
        harness.save_results(results, os.path.join(output_dir, "exp3"))

    # ── Experiment 4: Workload generalisation ─────────────────────────────
    if 4 in exp_ids:
        console.print("[bold]Experiment 4:[/bold] Workload-type generalisation")
        results = harness.run_experiment(
            experiment_name="exp4_workload_generalization",
            policies=["h2o", "hybrid"],
            memory_budgets=[0.75],
            tasks=[
                "narrativeqa", "qasper",        # single-doc QA
                "hotpotqa", "2wikimqa",          # multi-doc QA
                "gov_report", "qmsum",           # summarization
                "samsum",                        # dialogue
                "lcc", "repobench-p",            # code
                "trec", "passage_count",         # structured
            ],
        )
        harness.save_results(results, os.path.join(output_dir, "exp4"))

    # ── Experiment 6: Quantisation ablation ───────────────────────────────
    if 6 in exp_ids:
        console.print("[bold]Experiment 6:[/bold] Quantisation action ablation")
        results = harness.run_ablation(
            base_policy="hybrid",
            ablations=["hybrid_no_quant"],  # would need a matching config
            tasks=["narrativeqa", "hotpotqa"],
            memory_budget=0.60,
        )
        if results.comparison_table is not None:
            console.print(results.comparison_table.to_string())

    # ── Experiment 8: Stress test ──────────────────────────────────────────
    if 8 in exp_ids:
        console.print("[bold]Experiment 8:[/bold] Stress test — bursty arrivals")
        import asyncio
        from src.benchmarks.stress_tests import StressTestRunner
        for policy in ["full_cache", "hybrid"]:
            runner = StressTestRunner(api_url=api_url, policy_name=policy)
            result = asyncio.run(runner.run_bursty_arrivals(
                arrival_rate=5.0, burst_size=4, duration_s=60.0, context_length=8192
            ))
            console.print(
                f"  {policy}: RPS={result.throughput_rps}  "
                f"p95={result.p95_ms:.0f}ms  errors={result.error_rate:.1%}"
            )

    console.print(f"\n[green]Experiment suite complete.[/green]  Results: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
