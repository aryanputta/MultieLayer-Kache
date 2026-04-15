#!/usr/bin/env python3
"""
run_benchmark.py — CLI driver for the LongBench evaluation suite.

Usage
-----
    python scripts/run_benchmark.py --config configs/default.yaml
    python scripts/run_benchmark.py --policy hybrid --memory-budget 0.75
    python scripts/run_benchmark.py --tasks narrativeqa hotpotqa gov_report
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(help="KV Cache Orchestrator — Benchmark Runner", add_completion=False)
console = Console()


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Path to config YAML"),
    policy: str = typer.Option(None, "--policy", "-p", help="Override active policy"),
    tasks: list[str] = typer.Option(None, "--tasks", "-t", help="LongBench tasks to run"),
    memory_budget: float = typer.Option(None, "--memory-budget", "-m", help="GPU memory budget [0,1]"),
    concurrency: int = typer.Option(4, "--concurrency", "-j", help="Request concurrency"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Results output directory"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url", help="Serving API URL"),
    num_runs: int = typer.Option(1, "--num-runs", help="Number of repeated runs (for variance)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running"),
) -> None:
    """Run the LongBench evaluation suite against the serving API."""
    from omegaconf import OmegaConf
    import os

    # ── Load config ──────────────────────────────────────────────────────
    if not os.path.exists(config):
        console.print(f"[red]Config not found: {config}[/red]")
        raise typer.Exit(1)

    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)

    # Apply overrides
    if policy:
        cfg.setdefault("policy", {})["name"] = policy
    if memory_budget is not None:
        cfg.setdefault("benchmark", {})["active_budget"] = memory_budget
    if output_dir:
        cfg.setdefault("benchmark", {})["output_dir"] = output_dir

    benchmark_cfg = cfg.get("benchmark", {})
    lb_cfg = benchmark_cfg.get("longbench", {})
    task_list = tasks or lb_cfg.get("tasks", ["narrativeqa", "hotpotqa"])
    budget = memory_budget or 0.75
    out_dir = cfg.get("benchmark", {}).get("output_dir", "results")

    console.print(f"\n[bold cyan]KV Cache Orchestrator — Benchmark[/bold cyan]")
    console.print(f"  Config:       {config}")
    console.print(f"  Policy:       {cfg.get('policy', {}).get('name', 'default')}")
    console.print(f"  Tasks:        {task_list}")
    console.print(f"  Budget:       {budget}")
    console.print(f"  Concurrency:  {concurrency}")
    console.print(f"  Output:       {out_dir}\n")

    if dry_run:
        console.print("[yellow]Dry run — exiting without executing.[/yellow]")
        return

    # ── Run benchmark ────────────────────────────────────────────────────
    from src.benchmarks.longbench import LongBenchRunner, BenchmarkResult
    import os

    all_results: list[BenchmarkResult] = []

    for run_idx in range(num_runs):
        if num_runs > 1:
            console.print(f"[bold]Run {run_idx + 1}/{num_runs}[/bold]")

        runner = LongBenchRunner(
            api_url=api_url,
            config=cfg,
            policy_name=cfg.get("policy", {}).get("name", "unknown"),
            memory_budget=budget,
        )
        result = runner.run_all(tasks=task_list, concurrency=concurrency)
        all_results.append(result)

    # ── Display summary table ─────────────────────────────────────────────
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Task", style="white")
    table.add_column("Workload", style="dim")
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Samples", justify="right")
    table.add_column("Failed", justify="right", style="red")

    for tr in all_results[0].task_results:
        table.add_row(
            tr.task_name,
            tr.workload_type.value,
            tr.primary_metric,
            f"{tr.quality_score:.4f}",
            str(tr.n_samples),
            str(tr.failed_requests),
        )

    console.print(table)
    console.print(
        f"\n[bold]Aggregate quality:[/bold] "
        f"[green]{all_results[0].aggregate_quality:.4f}[/green]"
    )

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(
        out_dir, f"benchmark_{cfg.get('policy', {}).get('name', 'default')}.json"
    )
    all_results[0].save(save_path)
    console.print(f"\n[dim]Results saved to {save_path}[/dim]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
