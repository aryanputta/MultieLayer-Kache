"""
block_tracker.py — Thread-safe registry of every live KV-cache block.

:class:`BlockTracker` is the single authoritative source of truth for block
placement, attention statistics, and lifecycle counters.  The policy engine
reads from it; the serving engine writes to it.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import numpy as np

from src.telemetry.schema import BlockState, BlockTier


# ---------------------------------------------------------------------------
# Memory estimates (bytes) per block element, used for get_memory_stats
# ---------------------------------------------------------------------------
_BYTES_PER_ELEMENT_FP16 = 2
_BYTES_PER_ELEMENT_INT8 = 1          # rough post-quantisation estimate


class BlockTracker:
    """Maintains the runtime state of all KV-cache blocks.

    All public methods are thread-safe; they acquire a single reentrant lock
    before mutating or reading shared state.

    Parameters
    ----------
    num_layers:
        Number of transformer layers in the served model.
    num_heads:
        Number of attention heads (or head groups) per layer.
    block_size:
        Number of tokens packed into one KV block (default: 16).
    head_dim:
        Dimension of each attention head (used for memory estimation).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        block_size: int = 16,
        head_dim: int = 128,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim

        # block_id → BlockState
        self._blocks: Dict[int, BlockState] = {}
        self._lock = threading.RLock()

        # Bytes in one FP16 KV block (K + V, both heads, all tokens)
        self._block_bytes_fp16: int = (
            2 * num_heads * block_size * head_dim * _BYTES_PER_ELEMENT_FP16
        )
        self._block_bytes_int8: int = (
            2 * num_heads * block_size * head_dim * _BYTES_PER_ELEMENT_INT8
        )

    # ------------------------------------------------------------------
    # Block lifecycle
    # ------------------------------------------------------------------

    def register_block(
        self,
        block_id: int,
        layer_id: int,
        head_group: int,
        request_id: str,
    ) -> BlockState:
        """Create and register a new block, defaulting to GPU_FP16 tier.

        If a block with the same *block_id* already exists it is overwritten —
        callers should evict the old block first if that matters.

        Returns
        -------
        BlockState
            The freshly-created state object.
        """
        state = BlockState(
            block_id=block_id,
            layer_id=layer_id,
            head_group=head_group,
            request_id=request_id,
            current_tier=BlockTier.GPU_FP16,
        )
        with self._lock:
            self._blocks[block_id] = state
        return state

    def evict_block(self, block_id: int) -> Optional[BlockState]:
        """Remove a block from the tracker and return its last known state.

        Returns ``None`` if the block is not found.
        """
        with self._lock:
            state = self._blocks.pop(block_id, None)
            if state is not None:
                state.current_tier = BlockTier.EVICTED
            return state

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_attention(
        self, block_id: int, attention_scores: np.ndarray
    ) -> None:
        """Update rolling attention statistics for *block_id*.

        The running average is computed as an exponential moving average with
        α = 0.1 so that recent passes carry more weight.

        Parameters
        ----------
        block_id:
            Target block.
        attention_scores:
            1-D or 2-D array of attention weights from the latest forward
            pass.  Only finite values are considered.
        """
        scores = attention_scores.ravel()
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            return

        batch_mean = float(np.mean(finite))
        batch_max = float(np.max(finite))

        with self._lock:
            state = self._blocks.get(block_id)
            if state is None:
                return
            alpha = 0.1
            if state.access_count == 0:
                state.avg_attention_score = batch_mean
                state.max_attention_score = batch_max
            else:
                state.avg_attention_score = (
                    alpha * batch_mean + (1 - alpha) * state.avg_attention_score
                )
                state.max_attention_score = max(state.max_attention_score, batch_max)

    def mark_accessed(self, block_id: int, step: int) -> None:
        """Increment access counter and update last-access step."""
        with self._lock:
            state = self._blocks.get(block_id)
            if state is None:
                return
            state.access_count += 1
            state.last_access_step = step
            state.block_age_steps = step  # age grows with global step counter

    def update_tier(self, block_id: int, tier: BlockTier) -> None:
        """Move block to a different storage tier."""
        with self._lock:
            state = self._blocks.get(block_id)
            if state is None:
                return
            state.current_tier = tier

    def update_memory_snapshot(
        self,
        block_id: int,
        gpu_mem_used_mb: float,
        gpu_mem_free_mb: float,
    ) -> None:
        """Attach the current GPU memory snapshot to a block's state."""
        with self._lock:
            state = self._blocks.get(block_id)
            if state is None:
                return
            state.gpu_mem_used_mb = gpu_mem_used_mb
            state.gpu_mem_free_mb = gpu_mem_free_mb

    def set_importance_score(self, block_id: int, score: float) -> None:
        """Write the composite importance score produced by the policy engine."""
        with self._lock:
            state = self._blocks.get(block_id)
            if state is not None:
                state.importance_score = score

    def mark_prefix_reuse(self, block_id: int, reused: bool = True) -> None:
        """Flag whether this block's prefix was reused from a prior request."""
        with self._lock:
            state = self._blocks.get(block_id)
            if state is not None:
                state.prefix_reuse_flag = reused

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def get_block(self, block_id: int) -> Optional[BlockState]:
        """Return the :class:`BlockState` for *block_id*, or ``None``."""
        with self._lock:
            return self._blocks.get(block_id)

    def get_all_blocks(self) -> List[BlockState]:
        """Return a snapshot list of all tracked blocks."""
        with self._lock:
            return list(self._blocks.values())

    def get_blocks_by_tier(self, tier: BlockTier) -> List[BlockState]:
        """Return all blocks currently residing in *tier*."""
        with self._lock:
            return [b for b in self._blocks.values() if b.current_tier == tier]

    # ------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> dict:
        """Compute estimated memory consumption per tier.

        Returns
        -------
        dict
            Keys: ``total_blocks``, ``blocks_per_tier``,
            ``estimated_gpu_mb``, ``estimated_cpu_mb``, ``block_size``.
        """
        with self._lock:
            blocks = list(self._blocks.values())

        tier_counts: Dict[str, int] = {t.value: 0 for t in BlockTier}
        for b in blocks:
            tier_counts[b.current_tier.value] += 1

        fp16_blocks = tier_counts[BlockTier.GPU_FP16.value]
        quant_blocks = tier_counts[BlockTier.GPU_QUANTIZED.value]
        cpu_blocks = tier_counts[BlockTier.CPU_OFFLOADED.value]

        gpu_bytes = (
            fp16_blocks * self._block_bytes_fp16
            + quant_blocks * self._block_bytes_int8
        )
        cpu_bytes = cpu_blocks * self._block_bytes_fp16  # stored as fp16 on host

        return {
            "total_blocks": len(blocks),
            "blocks_per_tier": tier_counts,
            "estimated_gpu_mb": gpu_bytes / (1024 ** 2),
            "estimated_cpu_mb": cpu_bytes / (1024 ** 2),
            "block_size": self.block_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }

    # ------------------------------------------------------------------
    # Candidate selection for the policy engine
    # ------------------------------------------------------------------

    def get_candidate_blocks(
        self, n: int, strategy: str = "lru"
    ) -> List[BlockState]:
        """Return up to *n* blocks most likely to benefit from migration.

        Parameters
        ----------
        n:
            Maximum number of blocks to return.
        strategy:
            Selection heuristic.

            ``"lru"``
                Least-recently-used: blocks with the smallest
                ``last_access_step`` come first.
            ``"low_attention"``
                Blocks with the lowest ``avg_attention_score`` first —
                useful for pruning semantically unimportant entries.
            ``"age"``
                Oldest blocks (largest ``block_age_steps``) first.
            ``"importance"``
                Blocks with the lowest ``importance_score`` first
                (policy-engine output).

        Returns
        -------
        List[BlockState]
            Snapshot copies; modifications do not affect the tracker.
        """
        with self._lock:
            # Only consider blocks that are still on GPU
            candidates = [
                b
                for b in self._blocks.values()
                if b.current_tier in (BlockTier.GPU_FP16, BlockTier.GPU_QUANTIZED)
            ]

        if strategy == "lru":
            candidates.sort(key=lambda b: b.last_access_step)
        elif strategy == "low_attention":
            candidates.sort(key=lambda b: b.avg_attention_score)
        elif strategy == "age":
            candidates.sort(key=lambda b: -b.block_age_steps)
        elif strategy == "importance":
            candidates.sort(key=lambda b: b.importance_score)
        else:
            raise ValueError(
                f"Unknown candidate strategy '{strategy}'. "
                "Choose from: lru, low_attention, age, importance."
            )

        return candidates[:n]

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def release_request_blocks(self, request_id: str) -> int:
        """Evict all blocks associated with a completed request.

        Returns
        -------
        int
            Number of blocks removed.
        """
        with self._lock:
            stale = [
                bid
                for bid, b in self._blocks.items()
                if b.request_id == request_id
            ]
            for bid in stale:
                self._blocks.pop(bid)
        return len(stale)

    def increment_age(self, step: int) -> None:
        """Bump ``block_age_steps`` for every tracked block to reflect a new
        global decoding step.

        Typically called once per forward pass by the serving engine.
        """
        with self._lock:
            for b in self._blocks.values():
                b.block_age_steps = step

    def __len__(self) -> int:
        with self._lock:
            return len(self._blocks)

    def __repr__(self) -> str:
        stats = self.get_memory_stats()
        return (
            f"BlockTracker(total={stats['total_blocks']}, "
            f"gpu_fp16={stats['blocks_per_tier'].get('gpu_fp16', 0)}, "
            f"gpu_q={stats['blocks_per_tier'].get('gpu_quantized', 0)}, "
            f"cpu={stats['blocks_per_tier'].get('cpu_offloaded', 0)}, "
            f"evicted={stats['blocks_per_tier'].get('evicted', 0)})"
        )
