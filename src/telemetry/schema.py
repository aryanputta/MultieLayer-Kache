"""
schema.py — Shared data model for the Adaptive Multi-Tier KV Cache Orchestrator.

All telemetry records, block states, and metric structures are defined here so
that every module in the project works from a single source of truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class WorkloadType(str, Enum):
    """High-level characterisation of the request being served."""
    SINGLE_DOC_QA = "single_doc_qa"
    MULTI_DOC_QA = "multi_doc_qa"
    SUMMARIZATION = "summarization"
    DIALOGUE = "dialogue"
    CODE = "code"
    STRUCTURED_DATA = "structured_data"
    UNKNOWN = "unknown"


class BlockTier(str, Enum):
    """Storage tier that a KV-cache block currently occupies."""
    GPU_FP16 = "gpu_fp16"
    GPU_QUANTIZED = "gpu_quantized"
    CPU_OFFLOADED = "cpu_offloaded"
    EVICTED = "evicted"


class BlockDecision(str, Enum):
    """Policy decision applied to a KV-cache block."""
    KEEP_GPU = "keep_gpu"
    QUANTIZE = "quantize"
    OFFLOAD_CPU = "offload_cpu"
    EVICT = "evict"


# ---------------------------------------------------------------------------
# Block state
# ---------------------------------------------------------------------------

@dataclass
class BlockState:
    """Runtime state of a single KV-cache block.

    Instances are created by :class:`BlockTracker` and passed into the policy
    engine and telemetry collector.
    """

    block_id: int
    layer_id: int
    head_group: int
    request_id: str

    # Lifecycle counters
    block_age_steps: int = 0
    last_access_step: int = 0
    access_count: int = 0

    # Attention statistics (updated on each forward pass)
    avg_attention_score: float = 0.0
    max_attention_score: float = 0.0

    # Flags
    prefix_reuse_flag: bool = False

    # Current placement
    current_tier: BlockTier = BlockTier.GPU_FP16

    # Snapshot of system memory at last update
    gpu_mem_used_mb: float = 0.0
    gpu_mem_free_mb: float = 0.0

    # Composite score computed by the policy engine
    importance_score: float = 0.0

    # Derived helper properties -------------------------------------------

    @property
    def quantized_flag(self) -> bool:
        return self.current_tier == BlockTier.GPU_QUANTIZED

    @property
    def offloaded_flag(self) -> bool:
        return self.current_tier == BlockTier.CPU_OFFLOADED

    @property
    def evicted_flag(self) -> bool:
        return self.current_tier == BlockTier.EVICTED


# ---------------------------------------------------------------------------
# Telemetry record
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    """One row of telemetry, written to storage after every block decision.

    The field ordering mirrors the canonical 29-column schema so that
    :meth:`to_dict` / :meth:`from_block_state` stay trivially consistent.
    """

    # Request context
    request_id: str
    timestamp: float
    workload_type: str                  # WorkloadType.value
    model_name: str
    prompt_tokens: int
    generated_tokens: int
    context_length: int
    queue_depth: int

    # System memory snapshot
    gpu_mem_used_mb: float
    gpu_mem_free_mb: float
    cpu_mem_used_mb: float

    # Block identity
    block_id: int
    layer_id: int
    head_group: int

    # Block lifecycle
    block_age_steps: int
    last_access_step: int
    access_count: int

    # Attention statistics
    avg_attention_score: float
    max_attention_score: float

    # Block flags
    prefix_reuse_flag: bool
    quantized_flag: bool
    offloaded_flag: bool
    evicted_flag: bool

    # Policy output
    decision_label: str                 # BlockDecision.value

    # Latency metrics
    latency_ms: float
    ttft_ms: float
    tpot_ms: float

    # Task-level quality score (optional, default −1 when not yet scored)
    task_score: float = -1.0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dictionary suitable for pandas / pyarrow ingestion."""
        return asdict(self)

    @classmethod
    def from_block_state(
        cls,
        block: BlockState,
        decision: BlockDecision,
        *,
        model_name: str,
        workload_type: WorkloadType = WorkloadType.UNKNOWN,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        context_length: int = 0,
        queue_depth: int = 0,
        cpu_mem_used_mb: float = 0.0,
        latency_ms: float = 0.0,
        ttft_ms: float = 0.0,
        tpot_ms: float = 0.0,
        task_score: float = -1.0,
        timestamp: Optional[float] = None,
    ) -> "TelemetryRecord":
        """Construct a :class:`TelemetryRecord` from a live :class:`BlockState`.

        Parameters
        ----------
        block:
            The KV-cache block whose decision is being recorded.
        decision:
            The policy decision that was applied to this block.
        model_name:
            Identifier of the serving model (e.g. ``"mistral-7b"``).
        workload_type:
            Workload class of the originating request.
        prompt_tokens / generated_tokens / context_length:
            Token-count metadata from the request.
        queue_depth:
            Number of requests waiting in the serving queue at event time.
        cpu_mem_used_mb:
            Host-side RSS at event time.
        latency_ms / ttft_ms / tpot_ms:
            Latency measurements from the request that owns this block.
        task_score:
            Downstream evaluation score, if already known (-1 otherwise).
        timestamp:
            Unix timestamp override; defaults to ``time.time()``.
        """
        return cls(
            request_id=block.request_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            workload_type=workload_type.value,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            context_length=context_length,
            queue_depth=queue_depth,
            gpu_mem_used_mb=block.gpu_mem_used_mb,
            gpu_mem_free_mb=block.gpu_mem_free_mb,
            cpu_mem_used_mb=cpu_mem_used_mb,
            block_id=block.block_id,
            layer_id=block.layer_id,
            head_group=block.head_group,
            block_age_steps=block.block_age_steps,
            last_access_step=block.last_access_step,
            access_count=block.access_count,
            avg_attention_score=block.avg_attention_score,
            max_attention_score=block.max_attention_score,
            prefix_reuse_flag=block.prefix_reuse_flag,
            quantized_flag=block.quantized_flag,
            offloaded_flag=block.offloaded_flag,
            evicted_flag=block.evicted_flag,
            decision_label=decision.value,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            task_score=task_score,
        )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class SystemMetrics:
    """Point-in-time snapshot of system-level resource utilisation."""

    timestamp: float

    # GPU
    gpu_mem_used_mb: float
    gpu_mem_free_mb: float
    gpu_utilization_pct: float

    # CPU / host
    cpu_mem_used_mb: float
    cpu_utilization_pct: float

    # Serving
    queue_depth: int
    active_requests: int
    throughput_tps: float               # tokens per second (output)

    # Latency percentiles (rolling window)
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


@dataclass
class RequestMetrics:
    """Per-request latency and token-count telemetry."""

    request_id: str
    start_time: float
    end_time: float

    latency_ms: float
    ttft_ms: float
    tpot_ms: float

    prompt_tokens: int
    generated_tokens: int

    # Time the request spent waiting in the serving queue (ms)
    queue_wait_ms: float = 0.0

    # Task-specific quality score; populated asynchronously after eval
    task_score: Optional[float] = None
