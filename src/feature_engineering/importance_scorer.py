"""
importance_scorer.py — Oracle importance formula and pseudo-label generator.

The importance score for block i is defined as:

    importance_i = α * norm_future_attention_i
                 + β * reuse_probability_i
                 + γ * quality_drop_if_removed_i
                 - δ * memory_cost_i

where each term is normalised to [0, 1] before weighting.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.telemetry.schema import BlockState

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """Computes importance scores for KV-cache blocks using the oracle formula.

    Parameters
    ----------
    alpha:
        Weight for future-attention signal.
    beta:
        Weight for reuse-probability signal.
    gamma:
        Weight for quality-drop-if-removed signal.
    delta:
        Penalty weight for memory cost (always subtracted).
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.30,
        gamma: float = 0.30,
        delta: float = 0.10,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    # ------------------------------------------------------------------
    # Live scoring
    # ------------------------------------------------------------------

    def score_block(
        self,
        block: BlockState,
        future_attention: Optional[float] = None,
        reuse_probability: Optional[float] = None,
        quality_drop: Optional[float] = None,
        memory_cost: Optional[float] = None,
    ) -> float:
        """Compute the importance score for a single block.

        Missing signals fall back to heuristic proxies derived from the
        block's recorded statistics.
        """
        # Future attention: use provided value or fall back to rolling average
        att = future_attention if future_attention is not None else block.avg_attention_score
        att = float(np.clip(att, 0.0, 1.0))

        # Reuse probability: use prefix_reuse_flag as a hard indicator
        reu = reuse_probability if reuse_probability is not None else (
            1.0 if block.prefix_reuse_flag else
            min(1.0, block.access_count / 10.0)  # frequency heuristic
        )
        reu = float(np.clip(reu, 0.0, 1.0))

        # Quality drop: approximate by max attention score (higher = bigger drop)
        qd = quality_drop if quality_drop is not None else min(1.0, block.max_attention_score)
        qd = float(np.clip(qd, 0.0, 1.0))

        # Memory cost: uniform 1.0 for fp16 blocks, 0.5 for quantised
        from src.telemetry.schema import BlockTier
        if memory_cost is not None:
            mc = float(np.clip(memory_cost, 0.0, 1.0))
        else:
            mc = 0.5 if block.current_tier == BlockTier.GPU_QUANTIZED else 1.0

        score = (
            self.alpha * att
            + self.beta * reu
            + self.gamma * qd
            - self.delta * mc
        )
        return float(np.clip(score, 0.0, 1.0))

    def score_batch(
        self,
        blocks: List[BlockState],
        future_attentions: Optional[List[float]] = None,
        reuse_probs: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Return a (N,) importance score array for *blocks*."""
        scores = []
        for i, b in enumerate(blocks):
            fa = future_attentions[i] if future_attentions else None
            rp = reuse_probs[i] if reuse_probs else None
            scores.append(self.score_block(b, future_attention=fa, reuse_probability=rp))
        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Offline pseudo-label generation from traces
    # ------------------------------------------------------------------

    def generate_pseudo_labels(
        self,
        trace_df: pd.DataFrame,
        strategy: str = "combined",
        future_window_steps: int = 100,
    ) -> pd.DataFrame:
        """Generate block importance pseudo-labels from a telemetry trace.

        Parameters
        ----------
        trace_df:
            DataFrame with telemetry records (from :class:`TelemetryStorage`).
        strategy:
            Label generation strategy:
            - ``"attention_mass"`` — use ``avg_attention_score`` directly.
            - ``"future_reuse"`` — future access probability in next K steps.
            - ``"combined"`` — full oracle formula.
        future_window_steps:
            Steps ahead to look for future reuse (used by ``"future_reuse"``
            and ``"combined"`` strategies).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with an added ``importance_score`` column.
        """
        df = trace_df.copy()

        if strategy == "attention_mass":
            max_attn = df["avg_attention_score"].max()
            df["importance_score"] = df["avg_attention_score"] / max(max_attn, 1e-9)

        elif strategy == "future_reuse":
            df = self._label_future_reuse(df, future_window_steps)

        elif strategy == "combined":
            df = self._label_combined(df, future_window_steps)

        else:
            raise ValueError(
                f"Unknown labeling strategy: '{strategy}'. "
                "Choose from: attention_mass, future_reuse, combined."
            )

        return df

    # ------------------------------------------------------------------
    # Internal label strategies
    # ------------------------------------------------------------------

    def _label_future_reuse(
        self, df: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        """Label = 1 if the block is accessed again within *window* steps."""
        if "last_access_step" not in df.columns or "block_id" not in df.columns:
            df["importance_score"] = 0.5
            return df

        # Group by block_id; compute max future step within window
        df = df.sort_values("last_access_step")
        block_groups = df.groupby("block_id")["last_access_step"].apply(list).to_dict()

        scores = []
        for _, row in df.iterrows():
            steps = block_groups.get(row["block_id"], [])
            current_step = row["last_access_step"]
            future_accesses = sum(
                1 for s in steps
                if current_step < s <= current_step + window
            )
            scores.append(min(1.0, future_accesses / max(window / 10, 1)))

        df["importance_score"] = scores
        return df

    def _label_combined(
        self, df: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        """Full oracle: α*attention + β*reuse_prob + γ*quality_proxy - δ*mem_cost."""
        # Attention component
        max_attn = df["avg_attention_score"].max()
        att = df["avg_attention_score"] / max(max_attn, 1e-9)

        # Reuse probability from prefix_reuse_flag + access count
        reu = (
            df["prefix_reuse_flag"].astype(float)
            + (df["access_count"] / df["access_count"].clip(lower=1).max()) * 0.5
        ).clip(0, 1)

        # Quality drop proxy: max attention
        max_max_attn = df["max_attention_score"].max()
        qd = df["max_attention_score"] / max(max_max_attn, 1e-9)

        # Memory cost: normalised block size proxy (constant for now)
        mc = pd.Series(1.0, index=df.index)

        score = (
            self.alpha * att
            + self.beta * reu
            + self.gamma * qd
            - self.delta * mc
        ).clip(0, 1)

        df["importance_score"] = score
        return df
