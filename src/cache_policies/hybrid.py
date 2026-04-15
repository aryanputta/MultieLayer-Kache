"""
hybrid.py — Adaptive multi-tier KV-cache controller.

:class:`HybridAdaptivePolicy` is the primary research contribution of this
project.  It combines:
  1. Attention-based heavy-hitter identification
  2. Recency tracking (LRU signal)
  3. Prefix-reuse awareness
  4. Memory-pressure-sensitive action mapping (keep / quantise / offload / evict)
  5. Per-workload weight adaptation
  6. Optional learned importance-score override

Decision logic
--------------
  pressure < low_threshold
      → keep_gpu for all blocks

  low_threshold ≤ pressure < high_threshold
      → score every block
      → high score  → keep_gpu
      → medium score → quantise (if quantisation enabled)
      → low-medium   → offload_cpu (if offload enabled)
      → very low     → evict

  pressure ≥ high_threshold
      → aggressive: evict low, offload medium, quantise medium-high,
        keep only heavy hitters + sink tokens
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from src.telemetry.schema import BlockDecision, BlockState, WorkloadType
from src.cache_policies.base import BasePolicy, PolicyConfig, PolicyDecision

logger = logging.getLogger(__name__)


# Per-workload weight adjustments Δ applied to (recency, attention, reuse)
_WORKLOAD_DELTAS: Dict[WorkloadType, tuple] = {
    #                                Δrecency  Δattention  Δreuse
    WorkloadType.SINGLE_DOC_QA:    (  0.00,    +0.10,     0.00),
    WorkloadType.MULTI_DOC_QA:     (  0.00,    +0.10,    +0.05),
    WorkloadType.SUMMARIZATION:    ( +0.10,     0.00,     0.00),
    WorkloadType.DIALOGUE:         ( +0.05,     0.00,    +0.05),
    WorkloadType.CODE:             ( -0.05,    +0.05,    +0.15),
    WorkloadType.STRUCTURED_DATA:  (  0.00,    +0.05,    +0.05),
    WorkloadType.UNKNOWN:          (  0.00,     0.00,     0.00),
}


class HybridAdaptivePolicy(BasePolicy):
    """Workload-aware, pressure-sensitive KV-cache controller.

    Parameters
    ----------
    config:
        :class:`PolicyConfig` with all heuristic / quantisation / offload knobs.
    learned_policy:
        Optional :class:`LearnedPolicy`; when provided its importance scores
        are used in place of the heuristic composite score.
    """

    def __init__(
        self,
        config: PolicyConfig,
        learned_policy: Optional["LearnedPolicy"] = None,  # type: ignore[name-defined]
    ) -> None:
        super().__init__(config)
        self.learned_policy = learned_policy
        self._pressure_history: List[float] = []

    def get_name(self) -> str:
        return "hybrid"

    # ------------------------------------------------------------------
    # Main decision method
    # ------------------------------------------------------------------

    def decide(
        self,
        blocks: List[BlockState],
        system_metrics: dict,
    ) -> List[PolicyDecision]:
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)
        self._pressure_history.append(pressure)
        if len(self._pressure_history) > 1000:
            self._pressure_history = self._pressure_history[-1000:]

        if pressure < self.config.memory_pressure_low:
            decisions = self._keep_all(blocks)
        elif pressure < self.config.memory_pressure_high:
            decisions = self._medium_pressure(blocks, pressure, system_metrics)
        else:
            decisions = self._high_pressure(blocks, pressure, system_metrics)

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions

    # ------------------------------------------------------------------
    # Pressure regimes
    # ------------------------------------------------------------------

    def _medium_pressure(
        self,
        blocks: List[BlockState],
        pressure: float,
        system_metrics: dict,
    ) -> List[PolicyDecision]:
        """Selective: quantise and/or offload low-importance blocks."""
        scores = self._score_all(blocks, system_metrics)
        sink_ids = self._sink_ids(blocks)

        decisions = []
        for b, score in zip(blocks, scores):
            if b.block_id in sink_ids:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.KEEP_GPU, 1.0, "sink"))
                continue

            if score >= self.config.quantization_min_importance:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.KEEP_GPU, score, "high_score"))
            elif score >= self.config.offload_threshold_score and self.config.quantization_enabled:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.QUANTIZE, score, "medium_score_quantize"))
            elif score >= self.config.offload_threshold_score * 0.5 and self.config.offload_enabled:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.OFFLOAD_CPU, score, "low_medium_offload"))
            else:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.EVICT, score, "very_low_evict"))

        return decisions

    def _high_pressure(
        self,
        blocks: List[BlockState],
        pressure: float,
        system_metrics: dict,
    ) -> List[PolicyDecision]:
        """Aggressive: keep only heavy-hitters + sinks; evict everything else
        (with quantise and offload as intermediate tiers)."""
        scores = self._score_all(blocks, system_metrics)
        sink_ids = self._sink_ids(blocks)

        # Heavy-hitter threshold: top heavy_hitter_fraction by score
        n_hh = max(1, int(len(blocks) * self.config.heavy_hitter_fraction))
        sorted_by_score = sorted(zip(blocks, scores), key=lambda x: x[1], reverse=True)
        hh_ids = {b.block_id for b, _ in sorted_by_score[:n_hh]}

        decisions = []
        for b, score in zip(blocks, scores):
            if b.block_id in sink_ids or b.block_id in hh_ids:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.KEEP_GPU, score, "hh_or_sink"))
            elif score >= self.config.quantization_min_importance and self.config.quantization_enabled:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.QUANTIZE, score, "medium_high_quantize"))
            elif score >= self.config.offload_threshold_score and self.config.offload_enabled:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.OFFLOAD_CPU, score, "medium_offload"))
            else:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.EVICT, score, "aggressive_evict"))

        return decisions

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_all(
        self, blocks: List[BlockState], system_metrics: dict
    ) -> List[float]:
        """Return a composite importance score in [0, 1] for each block.

        If a learned policy is attached and its model is loaded, use its
        predictions.  Otherwise fall back to the heuristic composite.
        """
        if self.learned_policy is not None and self.learned_policy.is_ready():
            try:
                return self.learned_policy.score_blocks(blocks, system_metrics)
            except Exception as exc:
                logger.warning("Learned scoring failed, falling back to heuristic: %s", exc)

        return [self._score_block(b, system_metrics) for b in blocks]

    def _score_block(self, block: BlockState, system_metrics: dict) -> float:
        """Heuristic composite importance score for a single block."""
        # Determine workload type from system_metrics if available
        wt_str = system_metrics.get("workload_type", WorkloadType.UNKNOWN.value)
        try:
            wt = WorkloadType(wt_str)
        except ValueError:
            wt = WorkloadType.UNKNOWN

        dr, da, dru = _WORKLOAD_DELTAS.get(wt, (0.0, 0.0, 0.0))
        w_rec = max(0.0, self.config.recency_weight + dr)
        w_att = max(0.0, self.config.attention_weight + da)
        w_reu = max(0.0, self.config.reuse_weight + dru)
        # Re-normalise weights
        total_w = w_rec + w_att + w_reu
        if total_w > 0:
            w_rec /= total_w
            w_att /= total_w
            w_reu /= total_w

        # Recency: blocks with more-recent last_access_step score higher
        max_step = system_metrics.get("global_step", max(block.last_access_step, 1))
        recency = 1.0 - (max_step - block.last_access_step) / max(max_step, 1)
        recency = max(0.0, recency)

        # Attention: normalise by per-batch max
        attn_max = system_metrics.get("attn_max", max(block.max_attention_score, 1e-9))
        attention = block.avg_attention_score / attn_max if attn_max > 0 else 0.0
        attention = min(1.0, attention)

        # Reuse: binary flag (prefix reuse)
        reuse = 1.0 if block.prefix_reuse_flag else 0.0

        return w_rec * recency + w_att * attention + w_reu * reuse

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sink_ids(self, blocks: List[BlockState]) -> set:
        """Return block_ids of the first *sink_token_blocks* blocks."""
        sorted_by_id = sorted(blocks, key=lambda b: b.block_id)
        return {b.block_id for b in sorted_by_id[: self.config.sink_token_blocks]}

    def get_policy_report(self) -> dict:
        """Summary of recent pressure history and last decision stats."""
        stats = self._last_stats
        return {
            "policy_name": self.get_name(),
            "eval_count": self._eval_count,
            "avg_pressure_recent_100": (
                sum(self._pressure_history[-100:]) / len(self._pressure_history[-100:])
                if self._pressure_history else 0.0
            ),
            "last_stats": stats.to_dict() if stats else None,
            "learned_policy_ready": (
                self.learned_policy.is_ready() if self.learned_policy else False
            ),
        }
