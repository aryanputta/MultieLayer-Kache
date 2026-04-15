"""
label_generator.py — Generates pseudo-labels and train/val/test splits
from offline telemetry traces for importance model training.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.feature_engineering.extractor import FeatureExtractor, FEATURE_NAMES
from src.feature_engineering.importance_scorer import ImportanceScorer

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates feature matrices and importance labels from trace data.

    Parameters
    ----------
    strategy:
        Pseudo-label strategy: ``"attention_mass"``, ``"future_reuse"``,
        or ``"combined"``.
    alpha, beta, gamma, delta:
        Oracle formula weights (passed to :class:`ImportanceScorer`).
    future_window_steps:
        Look-ahead window for future-reuse label generation.
    """

    def __init__(
        self,
        strategy: str = "combined",
        alpha: float = 0.40,
        beta: float = 0.30,
        gamma: float = 0.30,
        delta: float = 0.10,
        future_window_steps: int = 100,
    ) -> None:
        self.strategy = strategy
        self.future_window_steps = future_window_steps
        self._scorer = ImportanceScorer(alpha, beta, gamma, delta)
        self._extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_from_trace(
        self,
        trace_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build (X, y, feature_names) from a raw telemetry trace DataFrame.

        Parameters
        ----------
        trace_df:
            DataFrame with columns matching :class:`TelemetryRecord` fields.

        Returns
        -------
        X : np.ndarray, shape (N, n_features)
        y : np.ndarray, shape (N,)
        feature_names : List[str]
        """
        logger.info(
            "LabelGenerator: generating labels (strategy=%s, n=%d)",
            self.strategy, len(trace_df),
        )

        # Generate labels
        labeled = self._scorer.generate_pseudo_labels(
            trace_df, strategy=self.strategy,
            future_window_steps=self.future_window_steps,
        )
        y = labeled["importance_score"].values.astype(np.float32)

        # Build feature matrix row-by-row from the DataFrame
        # (we use simplified feature extraction from tabular data since we
        #  don't have live BlockState objects at this point)
        X = self._build_feature_matrix(labeled)

        assert X.shape[0] == len(y), "Feature rows must match label count."
        logger.info("LabelGenerator: X.shape=%s  y range=[%.3f, %.3f]", X.shape, y.min(), y.max())
        return X, y, FEATURE_NAMES

    def generate_binary_labels(
        self, df: pd.DataFrame, threshold: float = 0.50
    ) -> pd.Series:
        """Convert continuous importance scores to binary keep/evict labels."""
        if "importance_score" not in df.columns:
            labeled = self._scorer.generate_pseudo_labels(df, strategy=self.strategy)
        else:
            labeled = df
        return (labeled["importance_score"] >= threshold).astype(int)

    def compute_label_statistics(self, df: pd.DataFrame) -> dict:
        """Summarise the distribution of generated labels."""
        if "importance_score" not in df.columns:
            df = self._scorer.generate_pseudo_labels(df, strategy=self.strategy)
        s = df["importance_score"]
        return {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(np.percentile(s, 25)),
            "p50": float(np.percentile(s, 50)),
            "p75": float(np.percentile(s, 75)),
            "pct_keep": float((s >= 0.5).mean()),
        }

    def split_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Random train / val / test split.

        Returns
        -------
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        rng = np.random.default_rng(seed)
        n = len(y)
        idx = rng.permutation(n)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_idx = idx[:n_test]
        val_idx = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]
        return (
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            (X[test_idx], y[test_idx]),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from a telemetry DataFrame.

        Uses the same feature names as :class:`FeatureExtractor` but reads
        directly from DataFrame columns instead of :class:`BlockState` objects.
        """
        n = len(df)
        X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

        col = lambda c, default=0.0: df[c].values.astype(float) if c in df.columns else np.full(n, default)

        max_age   = float(col("block_age_steps").max()) or 1.0
        max_step  = float(col("last_access_step").max()) or 1.0
        max_attn  = float(col("avg_attention_score").max()) or 1.0
        total_attn = float(col("avg_attention_score").sum()) or 1.0

        X[:, 0]  = col("block_age_steps") / max_age                       # block_age_normalized
        X[:, 1]  = 1.0 / (1.0 + max_step - col("last_access_step"))       # recency_score
        X[:, 2]  = col("access_count") / max(max_step, 1)                 # access_frequency
        X[:, 3]  = col("avg_attention_score")                              # avg_attention_score
        X[:, 4]  = col("max_attention_score")                              # max_attention_score

        # attention_rank_normalized
        attn_vals = col("avg_attention_score")
        sorted_attn = np.sort(attn_vals)
        X[:, 5] = np.searchsorted(sorted_attn, attn_vals) / max(len(sorted_attn), 1)

        X[:, 6]  = (col("block_id") < 4).astype(float)                    # sink_token_flag
        X[:, 7]  = col("prefix_reuse_flag")                                # prefix_reuse_flag

        max_layer = float(df["layer_id"].max()) if "layer_id" in df.columns else 1.0
        X[:, 8]  = col("layer_id") / max(max_layer, 1)                    # layer_depth_normalized
        X[:, 9]  = col("head_group") / 8.0                                 # head_group_normalized

        gpu_total = col("gpu_mem_used_mb") + col("gpu_mem_free_mb")
        gpu_total = np.where(gpu_total > 0, gpu_total, 1.0)
        X[:, 10] = col("gpu_mem_used_mb") / gpu_total                     # gpu_mem_pressure
        X[:, 11] = np.clip(col("queue_depth") / 64.0, 0, 1)              # queue_depth_normalized
        X[:, 12] = np.clip(col("context_length") / 32768.0, 0, 1)        # context_length_normalized
        X[:, 13] = np.clip(col("generated_tokens") / 32768.0, 0, 1)      # output_length_normalized

        # workload_type_encoded
        wt_map = {"unknown": 0, "single_doc_qa": 1, "multi_doc_qa": 2,
                  "summarization": 3, "dialogue": 4, "code": 5, "structured_data": 6}
        if "workload_type" in df.columns:
            X[:, 14] = df["workload_type"].map(wt_map).fillna(0).values / 6.0
        else:
            X[:, 14] = 0.0

        # is_heavy_hitter (top 20 % by attention)
        hh_threshold = np.percentile(attn_vals, 80)
        X[:, 15] = (attn_vals >= hh_threshold).astype(float)

        # is_recent_window (top 512 block_ids)
        if "block_id" in df.columns:
            max_bid = float(df["block_id"].max()) or 1.0
            X[:, 16] = np.clip((col("block_id") - (max_bid - 512)) / 512.0, 0, 1)

        X[:, 17] = np.log1p(col("access_count")) / np.log1p(max(max_step, 1))  # access_frequency_log
        X[:, 18] = col("avg_attention_score") / total_attn                      # cum_attention_mass_rank

        # tier_encoded
        tier_map = {"gpu_fp16": 0, "gpu_quantized": 1, "cpu_offloaded": 2, "evicted": 3}
        if "quantized_flag" in df.columns and "offloaded_flag" in df.columns:
            tier = np.zeros(n)
            tier[col("quantized_flag").astype(bool)] = 1.0
            tier[col("offloaded_flag").astype(bool)] = 2.0
            tier[col("evicted_flag").astype(bool)] = 3.0
            X[:, 19] = tier / 3.0

        X[:, 20] = (max_step - col("last_access_step")) / max(max_step, 1)     # inter_access_interval

        return X
