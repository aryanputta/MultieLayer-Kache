"""
base.py — Abstract interface shared by all KV-cache policies.

Every concrete policy inherits :class:`BasePolicy` and implements
:meth:`decide`, which maps a snapshot of live blocks to per-block
:class:`PolicyDecision` objects.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.telemetry.schema import BlockDecision, BlockState


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """Unified configuration object for any policy.

    Values mirror the ``policy`` section of ``configs/default.yaml``.
    """
    name: str = "hybrid"

    # Memory pressure thresholds (GPU used / total)
    memory_pressure_low: float = 0.65
    memory_pressure_high: float = 0.85

    # How often the engine runs a policy cycle (ms)
    scheduling_interval_ms: float = 50.0

    # Heuristic sub-config
    sliding_window_blocks: int = 512
    heavy_hitter_fraction: float = 0.20
    sink_token_blocks: int = 4
    recency_weight: float = 0.40
    attention_weight: float = 0.40
    reuse_weight: float = 0.20

    # Learned sub-config
    model_path: Optional[str] = None
    feature_set: str = "full"
    decision_threshold: float = 0.50
    fallback_policy: str = "h2o"

    # Quantisation action
    quantization_enabled: bool = True
    quantization_dtype: str = "int8"
    quantization_min_importance: float = 0.30

    # Offload action
    offload_enabled: bool = True
    offload_target: str = "cpu"
    offload_max_fraction: float = 0.40
    offload_threshold_score: float = 0.15

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyConfig":
        """Build from a nested config dict (e.g. OmegaConf output)."""
        heuristic = d.get("heuristic", {})
        learned = d.get("learned", {})
        quant = d.get("quantization", {})
        offload = d.get("offload", {})
        return cls(
            name=d.get("name", "hybrid"),
            memory_pressure_low=d.get("memory_pressure_low", 0.65),
            memory_pressure_high=d.get("memory_pressure_high", 0.85),
            scheduling_interval_ms=d.get("scheduling_interval_ms", 50.0),
            sliding_window_blocks=heuristic.get("sliding_window_blocks", 512),
            heavy_hitter_fraction=heuristic.get("heavy_hitter_fraction", 0.20),
            sink_token_blocks=heuristic.get("sink_token_blocks", 4),
            recency_weight=heuristic.get("recency_weight", 0.40),
            attention_weight=heuristic.get("attention_weight", 0.40),
            reuse_weight=heuristic.get("reuse_weight", 0.20),
            model_path=learned.get("model_path"),
            feature_set=learned.get("feature_set", "full"),
            decision_threshold=learned.get("decision_threshold", 0.50),
            fallback_policy=learned.get("fallback_policy", "h2o"),
            quantization_enabled=quant.get("enabled", True),
            quantization_dtype=quant.get("dtype", "int8"),
            quantization_min_importance=quant.get("min_importance_score", 0.30),
            offload_enabled=offload.get("enabled", True),
            offload_target=offload.get("target", "cpu"),
            offload_max_fraction=offload.get("max_offload_fraction", 0.40),
            offload_threshold_score=offload.get("offload_threshold_score", 0.15),
        )


# ---------------------------------------------------------------------------
# Decision and stats containers
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """The policy's verdict for one KV-cache block."""
    block_id: int
    decision: BlockDecision
    score: float                    # importance score in [0, 1]
    reason: str = ""                # human-readable explanation
    metadata: Dict = field(default_factory=dict)


@dataclass
class PolicyStats:
    """Summary of one policy evaluation round."""
    timestamp: float
    policy_name: str
    num_blocks_evaluated: int
    decisions: Dict[str, int]       # BlockDecision.value → count
    gpu_mem_pressure: float
    evaluation_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "policy_name": self.policy_name,
            "num_blocks_evaluated": self.num_blocks_evaluated,
            "decisions": self.decisions,
            "gpu_mem_pressure": round(self.gpu_mem_pressure, 4),
            "evaluation_latency_ms": round(self.evaluation_latency_ms, 3),
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BasePolicy(ABC):
    """Interface that every KV-cache policy must implement."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self._eval_count: int = 0
        self._last_stats: Optional[PolicyStats] = None

    @abstractmethod
    def decide(
        self,
        blocks: List[BlockState],
        system_metrics: dict,
    ) -> List[PolicyDecision]:
        """Map a list of live blocks to per-block decisions.

        Parameters
        ----------
        blocks:
            Snapshot of all currently tracked KV-cache blocks.
        system_metrics:
            Dict with at least ``gpu_mem_used_mb`` and ``gpu_mem_free_mb``.

        Returns
        -------
        List[PolicyDecision]
            One entry per block in *blocks* (same ordering).
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return the canonical policy name string."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def compute_memory_pressure(
        self,
        gpu_mem_used_mb: float,
        gpu_mem_free_mb: float,
    ) -> float:
        """GPU memory pressure as a fraction [0, 1]."""
        total = gpu_mem_used_mb + gpu_mem_free_mb
        return gpu_mem_used_mb / total if total > 0 else 0.0

    def _make_stats(
        self,
        decisions: List[PolicyDecision],
        pressure: float,
        latency_ms: float,
    ) -> PolicyStats:
        counts: Dict[str, int] = {}
        for d in decisions:
            counts[d.decision.value] = counts.get(d.decision.value, 0) + 1
        stats = PolicyStats(
            timestamp=time.time(),
            policy_name=self.get_name(),
            num_blocks_evaluated=len(decisions),
            decisions=counts,
            gpu_mem_pressure=pressure,
            evaluation_latency_ms=latency_ms,
        )
        self._last_stats = stats
        self._eval_count += 1
        return stats

    def get_last_stats(self) -> Optional[PolicyStats]:
        return self._last_stats

    def _keep_all(self, blocks: List[BlockState]) -> List[PolicyDecision]:
        """Return KEEP_GPU for every block — used under low pressure."""
        return [
            PolicyDecision(
                block_id=b.block_id,
                decision=BlockDecision.KEEP_GPU,
                score=1.0,
                reason="low_pressure",
            )
            for b in blocks
        ]
