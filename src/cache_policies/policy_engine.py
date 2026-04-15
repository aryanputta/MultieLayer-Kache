"""
policy_engine.py — Async scheduling loop that runs the active policy.

:class:`PolicyEngine` is the glue between the serving engine, the block
tracker, and the active policy.  It:
  - runs ``policy.decide()`` on a configurable interval
  - applies decisions by updating :class:`BlockTracker` tier labels
  - emits :class:`TelemetryRecord` events for each block decision
  - maintains a ring buffer of recent :class:`PolicyStats` for debugging
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Deque, List, Optional

from src.telemetry.schema import BlockDecision, BlockTier
from src.cache_policies.base import BasePolicy, PolicyDecision, PolicyStats

logger = logging.getLogger(__name__)

# Map from BlockDecision → new BlockTier (for tracker updates)
_DECISION_TO_TIER = {
    BlockDecision.KEEP_GPU:    BlockTier.GPU_FP16,
    BlockDecision.QUANTIZE:    BlockTier.GPU_QUANTIZED,
    BlockDecision.OFFLOAD_CPU: BlockTier.CPU_OFFLOADED,
    BlockDecision.EVICT:       BlockTier.EVICTED,
}


class PolicyEngine:
    """Runs the active KV-cache policy on a background asyncio loop.

    Parameters
    ----------
    policy:
        Any :class:`BasePolicy` subclass.
    block_tracker:
        :class:`~src.telemetry.BlockTracker` — single source of truth for
        block state.
    telemetry_collector:
        Optional :class:`~src.telemetry.TelemetryCollector` — receives one
        record per block decision.
    interval_ms:
        How often (ms) to run one full policy evaluation cycle.
    stats_buffer_size:
        Number of past :class:`PolicyStats` snapshots to keep.
    """

    def __init__(
        self,
        policy: BasePolicy,
        block_tracker,
        telemetry_collector=None,
        interval_ms: float = 50.0,
        stats_buffer_size: int = 500,
    ) -> None:
        self.policy = policy
        self.block_tracker = block_tracker
        self.telemetry_collector = telemetry_collector
        self.interval_ms = interval_ms

        self._history: Deque[PolicyStats] = deque(maxlen=stats_buffer_size)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background scheduling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="policy-engine")
        logger.info(
            "PolicyEngine started (policy=%s, interval=%.0fms)",
            self.policy.get_name(),
            self.interval_ms,
        )

    async def stop(self) -> None:
        """Stop the scheduling loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("PolicyEngine stopped after %d cycles.", self._cycle_count)

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._run_policy_cycle()
            except Exception as exc:
                logger.error("PolicyEngine cycle error: %s", exc, exc_info=True)
            await asyncio.sleep(self.interval_ms / 1000.0)

    async def _run_policy_cycle(self) -> Optional[PolicyStats]:
        """One full evaluation: snapshot blocks → decide → apply → record."""
        blocks = self.block_tracker.get_all_blocks()
        if not blocks:
            return None

        # Build system_metrics snapshot
        mem_stats = self.block_tracker.get_memory_stats()
        # Approximate GPU memory from block tracker estimates
        system_metrics = {
            "gpu_mem_used_mb": mem_stats.get("estimated_gpu_mb", 0.0),
            "gpu_mem_free_mb": max(
                0.0,
                self.block_tracker._blocks.__len__() * 0.0  # placeholder until engine provides real values
            ),
            "global_step": max((b.last_access_step for b in blocks), default=0),
            "attn_max": max((b.max_attention_score for b in blocks), default=1.0),
        }

        # Run policy
        decisions = self.policy.decide(blocks, system_metrics)

        # Apply decisions
        await self.apply_decisions(decisions)

        # Record telemetry
        if self.telemetry_collector:
            for d in decisions:
                block = self.block_tracker.get_block(d.block_id)
                if block:
                    self.telemetry_collector.record_block_event(
                        block=block,
                        decision=d.decision,
                    )

        stats = self.policy.get_last_stats()
        if stats:
            self._history.append(stats)

        self._cycle_count += 1
        return stats

    async def apply_decisions(self, decisions: List[PolicyDecision]) -> None:
        """Apply policy decisions to the block tracker.

        For EVICT decisions the block is removed from the tracker.
        For all other decisions the tier label is updated.
        """
        for d in decisions:
            new_tier = _DECISION_TO_TIER.get(d.decision)
            if new_tier is None:
                continue

            if d.decision == BlockDecision.EVICT:
                self.block_tracker.evict_block(d.block_id)
            else:
                self.block_tracker.update_tier(d.block_id, new_tier)
                self.block_tracker.set_importance_score(d.block_id, d.score)

    # ------------------------------------------------------------------
    # Runtime management
    # ------------------------------------------------------------------

    def override_policy(self, new_policy: BasePolicy) -> None:
        """Swap the active policy without restarting the loop."""
        old_name = self.policy.get_name()
        self.policy = new_policy
        logger.info(
            "PolicyEngine: policy switched %s → %s",
            old_name,
            new_policy.get_name(),
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_decision_history(self, n: int = 100) -> List[PolicyStats]:
        """Return the *n* most-recent :class:`PolicyStats` snapshots."""
        history = list(self._history)
        return history[-n:]

    def get_current_stats(self) -> dict:
        """Return a summary of the engine's recent activity."""
        last = list(self._history)[-1] if self._history else None
        return {
            "policy_name": self.policy.get_name(),
            "cycle_count": self._cycle_count,
            "running": self._running,
            "interval_ms": self.interval_ms,
            "last_stats": last.to_dict() if last else None,
        }
