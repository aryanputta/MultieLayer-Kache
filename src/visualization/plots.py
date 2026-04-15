"""
plots.py — Publication-quality result plots using Plotly.

All methods return :class:`plotly.graph_objects.Figure` objects so callers
can further customise or export them.  :meth:`save_all_plots` renders to both
interactive HTML and static PNG.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False
    logger.warning("plotly not installed — visualisation disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _MPL = True
except ImportError:
    _MPL = False


class ResultPlotter:
    """Factory for result visualisation plots.

    All methods accept a DataFrame and return a Plotly Figure (or None if
    plotly is not installed).
    """

    _POLICY_COLORS = {
        "full_cache":      "#636EFA",
        "sliding_window":  "#EF553B",
        "h2o":             "#00CC96",
        "prefix_caching":  "#AB63FA",
        "quantize_only":   "#FFA15A",
        "offload_only":    "#19D3F3",
        "hybrid":          "#FF6692",
        "learned":         "#B6E880",
    }

    # ------------------------------------------------------------------
    # Core plots
    # ------------------------------------------------------------------

    def plot_pareto_frontier(
        self,
        results_df: pd.DataFrame,
        x_col: str = "gpu_mem_peak_mb",
        y_col: str = "quality_score",
        color_col: str = "policy_name",
        title: Optional[str] = None,
    ) -> Optional["go.Figure"]:
        """Scatter plot of results with Pareto frontier highlighted."""
        if not _PLOTLY or results_df.empty:
            return None

        fig = px.scatter(
            results_df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_map=self._POLICY_COLORS,
            hover_data=results_df.columns.tolist(),
            title=title or f"Pareto Frontier: {y_col} vs {x_col}",
            labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85))
        fig.update_layout(
            legend_title_text="Policy",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    def plot_quality_memory_tradeoff(
        self, results_df: pd.DataFrame
    ) -> Optional["go.Figure"]:
        """Quality vs. GPU memory scatter coloured by policy, shaped by workload."""
        if not _PLOTLY or results_df.empty:
            return None

        has_workload = "workload_type" in results_df.columns
        symbol_col = "workload_type" if has_workload else None

        fig = px.scatter(
            results_df,
            x="gpu_mem_peak_mb",
            y="quality_score",
            color="policy_name",
            symbol=symbol_col,
            color_discrete_map=self._POLICY_COLORS,
            title="Quality vs. GPU Memory Peak",
            labels={
                "gpu_mem_peak_mb": "GPU Memory Peak (MB)",
                "quality_score": "Task Quality Score",
            },
            hover_data=["policy_name", "memory_budget", "task_name"] if "task_name" in results_df.columns else None,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return fig

    def plot_latency_cdf(
        self, latencies_by_policy: Dict[str, List[float]]
    ) -> Optional["go.Figure"]:
        """CDF of end-to-end latencies for each policy."""
        if not _PLOTLY:
            return None

        fig = go.Figure()
        for policy, lats in latencies_by_policy.items():
            if not lats:
                continue
            arr = np.sort(np.array(lats))
            cdf = np.arange(1, len(arr) + 1) / len(arr)
            fig.add_trace(go.Scatter(
                x=arr,
                y=cdf,
                mode="lines",
                name=policy,
                line=dict(color=self._POLICY_COLORS.get(policy)),
            ))
        fig.update_layout(
            title="Latency CDF by Policy",
            xaxis_title="Latency (ms)",
            yaxis_title="CDF",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    def plot_memory_tier_timeline(
        self, telemetry_df: pd.DataFrame
    ) -> Optional["go.Figure"]:
        """Stacked area chart: blocks per tier over time."""
        if not _PLOTLY or telemetry_df.empty:
            return None

        df = telemetry_df.copy()
        df["minute"] = (df["timestamp"] - df["timestamp"].min()) / 60.0
        df["tier"] = df.apply(
            lambda r: (
                "evicted" if r["evicted_flag"]
                else "offloaded" if r["offloaded_flag"]
                else "quantized" if r["quantized_flag"]
                else "fp16"
            ), axis=1
        )
        pivot = df.groupby(["minute", "tier"]).size().unstack(fill_value=0).reset_index()

        fig = go.Figure()
        tier_colors = {"fp16": "#636EFA", "quantized": "#FFA15A", "offloaded": "#19D3F3", "evicted": "#EF553B"}
        for tier, color in tier_colors.items():
            if tier in pivot.columns:
                fig.add_trace(go.Scatter(
                    x=pivot["minute"],
                    y=pivot[tier],
                    mode="lines",
                    stackgroup="one",
                    name=tier,
                    line=dict(color=color),
                ))
        fig.update_layout(
            title="KV Cache Memory Tiers Over Time",
            xaxis_title="Time (minutes)",
            yaxis_title="Block Count",
            plot_bgcolor="white",
        )
        return fig

    def plot_ablation_heatmap(
        self, ablation_results: pd.DataFrame
    ) -> Optional["go.Figure"]:
        """Heatmap: ablation variant × metric value."""
        if not _PLOTLY or ablation_results.empty:
            return None

        metric_cols = [c for c in ["f1", "em", "rouge_l", "edit_sim"] if c in ablation_results.columns]
        if not metric_cols or "policy_name" not in ablation_results.columns:
            return None

        pivot = ablation_results.groupby("policy_name")[metric_cols].mean()
        fig = px.imshow(
            pivot.values,
            x=metric_cols,
            y=pivot.index.tolist(),
            color_continuous_scale="RdYlGn",
            title="Ablation Study: Metrics per Policy Variant",
            labels=dict(x="Metric", y="Policy / Ablation"),
        )
        fig.update_layout(paper_bgcolor="white")
        return fig

    def plot_workload_breakdown(
        self, results_df: pd.DataFrame
    ) -> Optional["go.Figure"]:
        """Grouped bar: quality score per workload type per policy."""
        if not _PLOTLY or results_df.empty:
            return None
        if "workload_type" not in results_df.columns:
            return None

        agg = (
            results_df.groupby(["policy_name", "workload_type"])["quality_score"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            agg,
            x="workload_type",
            y="quality_score",
            color="policy_name",
            barmode="group",
            color_discrete_map=self._POLICY_COLORS,
            title="Quality Score by Workload Type",
            labels={"quality_score": "Mean Quality Score", "workload_type": "Workload Type"},
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return fig

    def plot_feature_importance(
        self, importance_dict: Dict[str, float]
    ) -> Optional["go.Figure"]:
        """Horizontal bar chart of feature importances."""
        if not _PLOTLY or not importance_dict:
            return None

        df = pd.DataFrame(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
            columns=["feature", "importance"],
        )
        fig = px.bar(
            df, x="importance", y="feature", orientation="h",
            title="Feature Importance (Learned Policy)",
            labels={"importance": "Importance Score", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white")
        return fig

    def plot_context_length_degradation(
        self, results_df: pd.DataFrame
    ) -> Optional["go.Figure"]:
        """Line chart: quality vs. context length per policy."""
        if not _PLOTLY or results_df.empty:
            return None
        if "context_length" not in results_df.columns:
            return None

        agg = (
            results_df.groupby(["policy_name", "context_length"])["quality_score"]
            .mean()
            .reset_index()
        )
        fig = px.line(
            agg, x="context_length", y="quality_score", color="policy_name",
            markers=True,
            color_discrete_map=self._POLICY_COLORS,
            title="Quality Degradation vs. Context Length",
            labels={"context_length": "Context Length (tokens)", "quality_score": "Quality Score"},
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return fig

    # ------------------------------------------------------------------
    # Batch export
    # ------------------------------------------------------------------

    def save_all_plots(self, results: dict, output_dir: str) -> None:
        """Generate and save all standard plots.

        Parameters
        ----------
        results:
            Dict with optional keys:
            - ``"results_df"``   → pd.DataFrame of benchmark results
            - ``"telemetry_df"`` → pd.DataFrame of raw telemetry
            - ``"latencies"``    → Dict[policy_name, List[float]]
            - ``"feature_importance"`` → Dict[str, float]
        output_dir:
            Directory to write HTML and PNG files.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = results.get("results_df")
        tel = results.get("telemetry_df")
        lats = results.get("latencies", {})
        fi = results.get("feature_importance", {})

        plots = {
            "pareto_frontier":          self.plot_pareto_frontier(df) if df is not None else None,
            "quality_memory_tradeoff":  self.plot_quality_memory_tradeoff(df) if df is not None else None,
            "latency_cdf":              self.plot_latency_cdf(lats),
            "memory_tier_timeline":     self.plot_memory_tier_timeline(tel) if tel is not None else None,
            "workload_breakdown":       self.plot_workload_breakdown(df) if df is not None else None,
            "context_degradation":      self.plot_context_length_degradation(df) if df is not None else None,
            "feature_importance":       self.plot_feature_importance(fi),
        }

        saved = []
        for name, fig in plots.items():
            if fig is None:
                continue
            html_path = os.path.join(output_dir, f"{name}.html")
            fig.write_html(html_path)
            saved.append(html_path)
            try:
                png_path = os.path.join(output_dir, f"{name}.png")
                fig.write_image(png_path, width=1200, height=700, scale=2)
                saved.append(png_path)
            except Exception as exc:
                logger.debug("Could not write PNG for %s: %s", name, exc)

        logger.info("Saved %d plot files to %s", len(saved), output_dir)
