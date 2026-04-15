"""
extractor.py — Per-block feature extraction for the importance ranker.

Feature vector (21 dimensions)
-------------------------------
 0  block_age_normalized
 1  recency_score
 2  access_frequency
 3  avg_attention_score
 4  max_attention_score
 5  attention_rank_normalized
 6  sink_token_flag
 7  prefix_reuse_flag
 8  layer_depth_normalized
 9  head_group_normalized
10  gpu_mem_pressure
11  queue_depth_normalized
12  context_length_normalized
13  output_length_so_far_normalized
14  workload_type_encoded
15  is_heavy_hitter
16  is_recent_window
17  access_frequency_log
18  cumulative_attention_mass_rank
19  time_in_current_tier_normalized
20  inter_access_interval_normalized
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from src.telemetry.schema import BlockState, BlockTier, WorkloadType

# Canonical feature names (must stay in sync with the vector positions above)
FEATURE_NAMES: List[str] = [
    "block_age_normalized",
    "recency_score",
    "access_frequency",
    "avg_attention_score",
    "max_attention_score",
    "attention_rank_normalized",
    "sink_token_flag",
    "prefix_reuse_flag",
    "layer_depth_normalized",
    "head_group_normalized",
    "gpu_mem_pressure",
    "queue_depth_normalized",
    "context_length_normalized",
    "output_length_normalized",
    "workload_type_encoded",
    "is_heavy_hitter",
    "is_recent_window",
    "access_frequency_log",
    "cumulative_attention_mass_rank",
    "tier_encoded",
    "inter_access_interval_normalized",
]

# Workload type → ordinal encoding
_WORKLOAD_ENCODING: Dict[str, float] = {
    WorkloadType.UNKNOWN.value:        0.0,
    WorkloadType.SINGLE_DOC_QA.value:  1.0,
    WorkloadType.MULTI_DOC_QA.value:   2.0,
    WorkloadType.SUMMARIZATION.value:  3.0,
    WorkloadType.DIALOGUE.value:       4.0,
    WorkloadType.CODE.value:           5.0,
    WorkloadType.STRUCTURED_DATA.value: 6.0,
}

_TIER_ENCODING: Dict[BlockTier, float] = {
    BlockTier.GPU_FP16:      0.0,
    BlockTier.GPU_QUANTIZED: 1.0,
    BlockTier.CPU_OFFLOADED: 2.0,
    BlockTier.EVICTED:       3.0,
}


class FeatureExtractor:
    """Converts :class:`BlockState` objects into fixed-length numpy feature vectors.

    Parameters
    ----------
    num_layers:
        Total number of transformer layers (used for normalisation).
    num_head_groups:
        Total number of KV-head groups.
    max_context_length:
        Maximum context length supported by the model.
    sink_token_blocks:
        Number of leading blocks treated as attention sinks.
    heavy_hitter_fraction:
        Fraction of blocks (by attention) that count as heavy hitters.
    recent_window_blocks:
        Number of trailing blocks considered "recent".
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_head_groups: int = 8,
        max_context_length: int = 32768,
        sink_token_blocks: int = 4,
        heavy_hitter_fraction: float = 0.20,
        recent_window_blocks: int = 512,
    ) -> None:
        self.num_layers = num_layers
        self.num_head_groups = num_head_groups
        self.max_context_length = max_context_length
        self.sink_token_blocks = sink_token_blocks
        self.heavy_hitter_fraction = heavy_hitter_fraction
        self.recent_window_blocks = recent_window_blocks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        block: BlockState,
        system_metrics: dict,
        batch_stats: dict,
    ) -> np.ndarray:
        """Extract a (21,) feature vector for *block*.

        Parameters
        ----------
        block:
            The KV-cache block to featurise.
        system_metrics:
            Dict with ``gpu_mem_used_mb``, ``gpu_mem_free_mb``,
            ``queue_depth``, ``context_length``, ``output_length``,
            ``workload_type``.
        batch_stats:
            Precomputed batch-level statistics from :meth:`compute_batch_stats`.
        """
        # Normalisation anchors from batch stats
        max_age     = max(batch_stats.get("max_age", 1), 1)
        max_step    = max(batch_stats.get("max_step", 1), 1)
        max_attn    = max(batch_stats.get("max_attention", 1e-9), 1e-9)
        total_blocks = max(batch_stats.get("total_blocks", 1), 1)
        attn_sorted  = batch_stats.get("attn_sorted", [])  # ascending attention values
        n_hh         = max(1, int(total_blocks * self.heavy_hitter_fraction))
        max_block_id = max(batch_stats.get("max_block_id", 1), 1)

        # ── Individual features ──────────────────────────────────────────
        f0  = block.block_age_steps / max_age
        f1  = 1.0 / (1.0 + max_step - block.last_access_step)
        f2  = block.access_count / max(max_step, 1)
        f3  = block.avg_attention_score
        f4  = block.max_attention_score

        # attention rank: fraction of blocks with LOWER attention (0=lowest, 1=highest)
        if attn_sorted:
            rank = int(np.searchsorted(attn_sorted, block.avg_attention_score))
            f5 = rank / len(attn_sorted)
        else:
            f5 = 0.5

        f6  = 1.0 if block.block_id < self.sink_token_blocks else 0.0
        f7  = 1.0 if block.prefix_reuse_flag else 0.0
        f8  = block.layer_id / max(self.num_layers - 1, 1)
        f9  = block.head_group / max(self.num_head_groups - 1, 1)

        gpu_total = block.gpu_mem_used_mb + block.gpu_mem_free_mb
        f10 = block.gpu_mem_used_mb / gpu_total if gpu_total > 0 else 0.0
        f11 = min(1.0, system_metrics.get("queue_depth", 0) / 64.0)
        f12 = min(1.0, system_metrics.get("context_length", 0) / self.max_context_length)
        f13 = min(1.0, system_metrics.get("output_length", 0) / self.max_context_length)

        wt_str = system_metrics.get("workload_type", WorkloadType.UNKNOWN.value)
        f14 = _WORKLOAD_ENCODING.get(wt_str, 0.0) / 6.0

        # heavy-hitter flag: is this block in the top hh_fraction by attention?
        if attn_sorted and len(attn_sorted) >= n_hh:
            hh_threshold = attn_sorted[-n_hh]
            f15 = 1.0 if block.avg_attention_score >= hh_threshold else 0.0
        else:
            f15 = 0.0

        # recent-window flag: is this one of the last *recent_window_blocks* by block_id?
        f16 = 1.0 if block.block_id >= max_block_id - self.recent_window_blocks else 0.0

        f17 = math.log1p(block.access_count) / math.log1p(max_step)

        # cumulative attention mass rank (fraction of total attention mass in this block)
        total_attn = batch_stats.get("total_attention", 1e-9)
        f18 = block.avg_attention_score / total_attn

        f19 = _TIER_ENCODING.get(block.current_tier, 0.0) / 3.0

        inter = max_step - block.last_access_step
        f20 = inter / max_step if max_step > 0 else 0.0

        return np.array(
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
             f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20],
            dtype=np.float32,
        )

    def extract_batch(
        self,
        blocks: List[BlockState],
        system_metrics: dict,
        batch_stats: dict,
    ) -> np.ndarray:
        """Return a (N, 21) feature matrix for *blocks*."""
        if not blocks:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        rows = [self.extract(b, system_metrics, batch_stats) for b in blocks]
        return np.stack(rows, axis=0)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names."""
        return list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Batch statistics (call once per policy cycle)
    # ------------------------------------------------------------------

    def compute_batch_stats(self, blocks: List[BlockState]) -> dict:
        """Precompute aggregates needed for normalisation.

        Returns
        -------
        dict
            Keys: ``max_age``, ``max_step``, ``max_attention``,
            ``total_blocks``, ``attn_sorted``, ``total_attention``,
            ``max_block_id``.
        """
        if not blocks:
            return {
                "max_age": 1, "max_step": 1, "max_attention": 1.0,
                "total_blocks": 0, "attn_sorted": [], "total_attention": 1.0,
                "max_block_id": 0,
            }
        attn_vals = np.array([b.avg_attention_score for b in blocks], dtype=np.float32)
        return {
            "max_age":       int(max(b.block_age_steps for b in blocks)),
            "max_step":      int(max(b.last_access_step for b in blocks)),
            "max_attention": float(attn_vals.max()),
            "total_blocks":  len(blocks),
            "attn_sorted":   sorted(attn_vals.tolist()),
            "total_attention": float(attn_vals.sum()) or 1.0,
            "max_block_id":  int(max(b.block_id for b in blocks)),
        }
