"""
heuristic.py — Rule-based KV-cache policies.

Implements the system baselines and research-inspired heuristics used in the
ablation study:
  - FullCachePolicy       (LRU-only baseline)
  - SlidingWindowPolicy   (StreamingLLM-style)
  - H2OPolicy             (Heavy-Hitter Oracle)
  - PrefixCachingPolicy   (reuse-aware)
  - QuantizeOnlyPolicy    (no eviction, only quantisation)
  - OffloadOnlyPolicy     (no eviction, only offload)
"""

from __future__ import annotations

import logging
import time
from typing import List

from src.telemetry.schema import BlockDecision, BlockState, BlockTier
from src.cache_policies.base import BasePolicy, PolicyConfig, PolicyDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(val: float, lo: float, hi: float) -> float:
    """Normalise *val* to [0, 1] in [lo, hi].  Clamps outside the range."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _recency_score(block: BlockState, max_step: int) -> float:
    """Higher score = accessed more recently."""
    if max_step == 0:
        return 1.0
    return 1.0 - _norm(block.last_access_step, 0, max_step)


# ---------------------------------------------------------------------------
# 1. Full-cache baseline (LRU eviction under OOM)
# ---------------------------------------------------------------------------

class FullCachePolicy(BasePolicy):
    """Keep all blocks on GPU; evict LRU only under memory pressure.

    This is the simplest possible policy and the primary system baseline.
    """

    def get_name(self) -> str:
        return "full_cache"

    def decide(
        self, blocks: List[BlockState], system_metrics: dict
    ) -> List[PolicyDecision]:
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)

        if pressure < self.config.memory_pressure_high:
            decisions = self._keep_all(blocks)
        else:
            # Sort by LRU (oldest access first)
            sorted_blocks = sorted(blocks, key=lambda b: b.last_access_step)
            # Evict bottom 10 % to relieve pressure
            n_evict = max(1, int(len(sorted_blocks) * 0.10))
            evict_set = {b.block_id for b in sorted_blocks[:n_evict]}
            decisions = [
                PolicyDecision(
                    block_id=b.block_id,
                    decision=BlockDecision.EVICT if b.block_id in evict_set else BlockDecision.KEEP_GPU,
                    score=0.0 if b.block_id in evict_set else 1.0,
                    reason="lru_evict" if b.block_id in evict_set else "keep",
                )
                for b in blocks
            ]

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions


# ---------------------------------------------------------------------------
# 2. Sliding-window (StreamingLLM-inspired)
# ---------------------------------------------------------------------------

class SlidingWindowPolicy(BasePolicy):
    """Keep the last *window_size* blocks plus the first *sink_blocks*.

    Inspired by StreamingLLM (Xiao et al., 2023): attention sinks at position
    0 receive disproportionate weight, so they are always retained.
    """

    def get_name(self) -> str:
        return "sliding_window"

    def decide(
        self, blocks: List[BlockState], system_metrics: dict
    ) -> List[PolicyDecision]:
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)

        if pressure < self.config.memory_pressure_low:
            decisions = self._keep_all(blocks)
            self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
            return decisions

        window = self.config.sliding_window_blocks
        sink = self.config.sink_token_blocks

        # Sort by block_id as a proxy for position in the context
        sorted_by_id = sorted(blocks, key=lambda b: b.block_id)
        total = len(sorted_by_id)

        keep_ids = set()
        # Sink tokens (earliest block_ids)
        for b in sorted_by_id[:sink]:
            keep_ids.add(b.block_id)
        # Recent window (latest block_ids)
        for b in sorted_by_id[max(0, total - window):]:
            keep_ids.add(b.block_id)

        decisions = []
        for b in blocks:
            if b.block_id in keep_ids:
                decisions.append(PolicyDecision(
                    block_id=b.block_id,
                    decision=BlockDecision.KEEP_GPU,
                    score=1.0,
                    reason="sink_or_window",
                ))
            else:
                decisions.append(PolicyDecision(
                    block_id=b.block_id,
                    decision=BlockDecision.EVICT,
                    score=0.0,
                    reason="outside_window",
                ))

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions


# ---------------------------------------------------------------------------
# 3. H2O — Heavy-Hitter Oracle
# ---------------------------------------------------------------------------

class H2OPolicy(BasePolicy):
    """Retain the top *heavy_hitter_fraction* blocks by cumulative attention
    mass, plus sink tokens.  Evict the rest under memory pressure.

    Inspired by H2O (Zhang et al., 2023).
    """

    def get_name(self) -> str:
        return "h2o"

    def decide(
        self, blocks: List[BlockState], system_metrics: dict
    ) -> List[PolicyDecision]:
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)

        if pressure < self.config.memory_pressure_low:
            decisions = self._keep_all(blocks)
            self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
            return decisions

        sink = self.config.sink_token_blocks
        hh_frac = self.config.heavy_hitter_fraction
        sorted_by_id = sorted(blocks, key=lambda b: b.block_id)

        # Sink token block_ids (first *sink* by position)
        sink_ids = {b.block_id for b in sorted_by_id[:sink]}

        # Heavy-hitter block_ids (top hh_frac by avg attention)
        n_hh = max(1, int(len(blocks) * hh_frac))
        heavy = sorted(blocks, key=lambda b: b.avg_attention_score, reverse=True)
        hh_ids = {b.block_id for b in heavy[:n_hh]}

        keep_ids = sink_ids | hh_ids

        decisions = []
        for b in blocks:
            if b.block_id in keep_ids:
                decisions.append(PolicyDecision(
                    block_id=b.block_id,
                    decision=BlockDecision.KEEP_GPU,
                    score=b.avg_attention_score,
                    reason="heavy_hitter" if b.block_id in hh_ids else "sink",
                ))
            else:
                decisions.append(PolicyDecision(
                    block_id=b.block_id,
                    decision=BlockDecision.EVICT,
                    score=b.avg_attention_score,
                    reason="low_attention",
                ))

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions


# ---------------------------------------------------------------------------
# 4. Prefix-caching policy
# ---------------------------------------------------------------------------

class PrefixCachingPolicy(BasePolicy):
    """Prioritise blocks with ``prefix_reuse_flag=True``; evict non-reuse
    blocks first under pressure.
    """

    def get_name(self) -> str:
        return "prefix_caching"

    def decide(
        self, blocks: List[BlockState], system_metrics: dict
    ) -> List[PolicyDecision]:
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)

        if pressure < self.config.memory_pressure_low:
            decisions = self._keep_all(blocks)
            self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
            return decisions

        # Score: prefix-reuse gets 1.0, non-reuse gets attention-based score
        max_attn = max((b.avg_attention_score for b in blocks), default=1.0)

        decisions = []
        for b in blocks:
            if b.prefix_reuse_flag:
                score = 1.0
                decision = BlockDecision.KEEP_GPU
                reason = "prefix_reuse"
            else:
                score = b.avg_attention_score / max_attn if max_attn > 0 else 0.0
                if pressure >= self.config.memory_pressure_high:
                    decision = BlockDecision.EVICT if score < 0.20 else BlockDecision.KEEP_GPU
                else:
                    decision = BlockDecision.EVICT if score < 0.10 else BlockDecision.KEEP_GPU
                reason = "low_score" if decision == BlockDecision.EVICT else "keep"
            decisions.append(PolicyDecision(
                block_id=b.block_id,
                decision=decision,
                score=score,
                reason=reason,
            ))

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions


# ---------------------------------------------------------------------------
# 5. Quantise-only (no eviction)
# ---------------------------------------------------------------------------

class QuantizeOnlyPolicy(BasePolicy):
    """Never evict or offload blocks.  Quantise low-attention blocks to int8;
    keep high-attention blocks as fp16.

    Useful for isolating the effect of quantisation in ablations.
    """

    def get_name(self) -> str:
        return "quantize_only"

    def decide(
        self, blocks: List[BlockState], system_metrics: dict
    ) -> List[PolicyDecision]:
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)

        if not self.config.quantization_enabled or pressure < self.config.memory_pressure_low:
            decisions = self._keep_all(blocks)
            self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
            return decisions

        sorted_by_id = sorted(blocks, key=lambda b: b.block_id)
        sink_ids = {b.block_id for b in sorted_by_id[:self.config.sink_token_blocks]}
        max_attn = max((b.avg_attention_score for b in blocks), default=1.0)

        decisions = []
        for b in blocks:
            if b.block_id in sink_ids or b.current_tier == BlockTier.GPU_QUANTIZED:
                decision = BlockDecision.KEEP_GPU
                score = 1.0
                reason = "sink_or_already_quantized"
            else:
                norm_attn = b.avg_attention_score / max_attn if max_attn > 0 else 0.0
                if norm_attn < self.config.quantization_min_importance:
                    decision = BlockDecision.QUANTIZE
                    score = norm_attn
                    reason = "low_attention_quantize"
                else:
                    decision = BlockDecision.KEEP_GPU
                    score = norm_attn
                    reason = "high_attention_keep"
            decisions.append(PolicyDecision(
                block_id=b.block_id, decision=decision, score=score, reason=reason
            ))

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions


# ---------------------------------------------------------------------------
# 6. Offload-only (no eviction, only CPU offload)
# ---------------------------------------------------------------------------

class OffloadOnlyPolicy(BasePolicy):
    """Never evict.  Offload medium-importance blocks to CPU host memory.
    Useful for isolating the offload action in ablations.
    """

    def get_name(self) -> str:
        return "offload_only"

    def decide(
        self, blocks: List[BlockState], system_metrics: dict
    ) -> List[PolicyDecision]:
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)

        if not self.config.offload_enabled or pressure < self.config.memory_pressure_low:
            decisions = self._keep_all(blocks)
            self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
            return decisions

        sorted_by_id = sorted(blocks, key=lambda b: b.block_id)
        sink_ids = {b.block_id for b in sorted_by_id[:self.config.sink_token_blocks]}
        max_attn = max((b.avg_attention_score for b in blocks), default=1.0)

        # How many blocks can we offload?
        max_offload = int(len(blocks) * self.config.offload_max_fraction)
        offload_count = 0

        scored = [
            (b, b.avg_attention_score / max_attn if max_attn > 0 else 0.0)
            for b in blocks
        ]
        scored.sort(key=lambda x: x[1])  # lowest attention first

        offload_set = set()
        for b, score in scored:
            if b.block_id in sink_ids:
                continue
            if score <= self.config.offload_threshold_score and offload_count < max_offload:
                offload_set.add(b.block_id)
                offload_count += 1

        decisions = []
        for b, score in scored:
            if b.block_id in offload_set:
                dec = BlockDecision.OFFLOAD_CPU
                reason = "low_score_offload"
            else:
                dec = BlockDecision.KEEP_GPU
                reason = "keep"
            decisions.append(PolicyDecision(
                block_id=b.block_id, decision=dec, score=score, reason=reason
            ))

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions
