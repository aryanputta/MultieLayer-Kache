#!/usr/bin/env python3
"""
train_policy.py — Train the block importance ranking model from telemetry traces.

Usage
-----
    python scripts/train_policy.py --config configs/default.yaml
    python scripts/train_policy.py --traces data/traces/ --model-type lightgbm --n-trials 50
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(help="Train KV-cache block importance ranking model")
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def train(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    traces: str = typer.Option(None, "--traces", help="Override trace directory"),
    model_type: str = typer.Option("xgboost", "--model-type", help="xgboost | lightgbm"),
    strategy: str = typer.Option("combined", "--strategy", help="Label strategy: attention_mass | future_reuse | combined"),
    n_trials: int = typer.Option(0, "--n-trials", help="Optuna HPO trials (0 = skip)"),
    output: str = typer.Option(None, "--output", "-o", help="Model output path"),
    val_frac: float = typer.Option(0.15, "--val-frac"),
    test_frac: float = typer.Option(0.15, "--test-frac"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Load traces, generate pseudo-labels, train importance model, save."""
    from omegaconf import OmegaConf
    import os

    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True) if Path(config).exists() else {}
    trace_dir = traces or cfg.get("telemetry", {}).get("output_dir", "data/traces")
    model_output = output or "results/models/importance_ranker.pkl"

    console.print(f"\n[bold cyan]Training Importance Model[/bold cyan]")
    console.print(f"  Traces dir:  {trace_dir}")
    console.print(f"  Model type:  {model_type}")
    console.print(f"  Strategy:    {strategy}")
    console.print(f"  HPO trials:  {n_trials}\n")

    # ── Load traces ──────────────────────────────────────────────────────
    from src.telemetry.storage import TelemetryStorage
    storage = TelemetryStorage(backend="parquet", output_dir=trace_dir)
    df = storage.read_all()

    if df.empty:
        console.print(f"[red]No traces found in {trace_dir}.[/red]")
        console.print("[dim]Run a benchmark with telemetry enabled first.[/dim]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(df):,} telemetry records.")

    # ── Generate labels and features ─────────────────────────────────────
    from src.trainers.label_generator import LabelGenerator
    gen = LabelGenerator(strategy=strategy)
    X, y, feature_names = gen.generate_from_trace(df)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = gen.split_train_val_test(
        X, y, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    console.print(
        f"Split: train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}"
    )

    stats = gen.compute_label_statistics(df)
    console.print(
        f"Labels: mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
        f"pct_keep={stats['pct_keep']:.1%}"
    )

    # ── Train ─────────────────────────────────────────────────────────────
    from src.trainers.importance_model import ImportanceModelTrainer
    trainer = ImportanceModelTrainer(
        model_type=model_type,
        config={"n_trials": n_trials},
    )
    metrics = trainer.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    console.print(f"\n[green]Training complete.[/green]")
    console.print(f"  Val MSE:     {metrics['val_mse']:.6f}")
    console.print(f"  Val Pearson: {metrics['val_pearson']:.4f}")
    console.print(f"  Train time:  {metrics['train_time_s']:.1f}s")

    # ── Test evaluation ───────────────────────────────────────────────────
    import numpy as np
    test_preds = trainer.predict(X_test)
    test_mse = float(np.mean((y_test - test_preds) ** 2))
    from scipy.stats import pearsonr
    test_r, _ = pearsonr(y_test, test_preds)
    console.print(f"  Test MSE:    {test_mse:.6f}")
    console.print(f"  Test Pearson:{test_r:.4f}")

    # ── Feature importance ────────────────────────────────────────────────
    fi = trainer.get_feature_importance()
    if fi:
        top10 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        table = Table(title="Top 10 Features", show_header=True)
        table.add_column("Feature")
        table.add_column("Importance", justify="right")
        for fname, imp in top10:
            table.add_row(fname, f"{imp:.4f}")
        console.print(table)

    # ── Save model ────────────────────────────────────────────────────────
    os.makedirs(Path(model_output).parent, exist_ok=True)
    trainer.save(model_output)
    console.print(f"\n[dim]Model saved to {model_output}[/dim]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
