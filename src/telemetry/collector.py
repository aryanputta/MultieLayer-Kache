"""
collector.py — Thread-safe event collector for KV-cache telemetry.

:class:`TelemetryCollector` receives block-decision and request-completion
events from the serving engine, buffers them in memory, and periodically
flushes batches to :class:`TelemetryStorage`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from src.telemetry.schema import (
    BlockDecision,
    BlockState,
    RequestMetrics,
    TelemetryRecord,
    WorkloadType,
)
from src.telemetry.storage import TelemetryStorage

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """Buffers :class:`TelemetryRecord` objects and flushes them to storage.

    Parameters
    ----------
    config:
        Telemetry section of the project config dict.  Expected keys:
        ``enabled``, ``storage_backend``, ``output_dir``,
        ``flush_interval_s``, ``max_buffer_size``.
    model_name:
        Identifier of the served model; embedded in every record.
    """

    def __init__(self, config: dict, model_name: str = "unknown") -> None:
        self.enabled: bool = config.get("enabled", True)
        self.model_name = model_name
        self.flush_interval_s: float = float(config.get("flush_interval_s", 10))
        self.max_buffer_size: int = int(config.get("max_buffer_size", 10_000))

        self._buffer: List[TelemetryRecord] = []
        self._lock = threading.Lock()

        self._storage = TelemetryStorage(
            backend=config.get("storage_backend", "parquet"),
            output_dir=config.get("output_dir", "data/traces"),
        )

        # Per-request metrics cache (keyed by request_id) for TTFT / TPOT
        self._request_cache: dict = {}

        # Aggregate counters
        self._total_records: int = 0
        self._total_flushes: int = 0

        # Background flush thread
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        if self.enabled:
            self._start_flush_thread()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_block_event(
        self,
        block: BlockState,
        decision: BlockDecision,
        request_metrics: Optional[RequestMetrics] = None,
        workload_type: WorkloadType = WorkloadType.UNKNOWN,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        context_length: int = 0,
        queue_depth: int = 0,
        cpu_mem_used_mb: float = 0.0,
    ) -> None:
        """Create and buffer a :class:`TelemetryRecord` for one block event.

        Parameters
        ----------
        block:
            Current state of the KV-cache block.
        decision:
            Policy decision applied to the block.
        request_metrics:
            Optional per-request latency snapshot.
        """
        if not self.enabled:
            return

        rm = request_metrics
        record = TelemetryRecord.from_block_state(
            block,
            decision,
            model_name=self.model_name,
            workload_type=workload_type,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            context_length=context_length,
            queue_depth=queue_depth,
            cpu_mem_used_mb=cpu_mem_used_mb,
            latency_ms=rm.latency_ms if rm else 0.0,
            ttft_ms=rm.ttft_ms if rm else 0.0,
            tpot_ms=rm.tpot_ms if rm else 0.0,
            task_score=rm.task_score if (rm and rm.task_score is not None) else -1.0,
        )

        with self._lock:
            self._buffer.append(record)
            self._total_records += 1
            if len(self._buffer) >= self.max_buffer_size:
                self._flush_locked()

    def record_request_completion(self, metrics: RequestMetrics) -> None:
        """Cache per-request metrics so future block events can reference them."""
        if not self.enabled:
            return
        with self._lock:
            self._request_cache[metrics.request_id] = metrics

    def get_request_metrics(self, request_id: str) -> Optional[RequestMetrics]:
        """Retrieve cached metrics for a given request."""
        with self._lock:
            return self._request_cache.get(request_id)

    def flush(self) -> None:
        """Manually flush the current buffer to storage."""
        with self._lock:
            self._flush_locked()

    def get_recent_records(self, n: int = 1000) -> List[TelemetryRecord]:
        """Return up to *n* most-recently-buffered records (not yet flushed)."""
        with self._lock:
            return list(self._buffer[-n:])

    def get_statistics(self) -> dict:
        """Return aggregate statistics about collected telemetry."""
        with self._lock:
            buf = list(self._buffer)

        decision_counts: dict = {}
        for r in buf:
            decision_counts[r.decision_label] = (
                decision_counts.get(r.decision_label, 0) + 1
            )

        avg_importance = (
            sum(r.avg_attention_score for r in buf) / len(buf) if buf else 0.0
        )

        return {
            "buffer_size": len(buf),
            "total_records_collected": self._total_records,
            "total_flushes": self._total_flushes,
            "decision_counts": decision_counts,
            "avg_attention_score_in_buffer": round(avg_importance, 6),
        }

    def stop(self) -> None:
        """Stop the background flush thread and perform a final flush."""
        self._stop_event.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=self.flush_interval_s + 2)
        self.flush()
        logger.info(
            "TelemetryCollector stopped. Total records: %d, flushes: %d",
            self._total_records,
            self._total_flushes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_locked(self) -> None:
        """Flush the buffer to storage — must be called with *self._lock* held."""
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        try:
            self._storage.write_batch(batch)
            self._total_flushes += 1
            logger.debug("Flushed %d telemetry records.", len(batch))
        except Exception as exc:
            logger.error("Failed to flush telemetry: %s", exc)
            # Re-buffer on failure to avoid data loss
            self._buffer.extend(batch)

    def _start_flush_thread(self) -> None:
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="telemetry-flush",
            daemon=True,
        )
        self._flush_thread.start()
        logger.debug(
            "TelemetryCollector flush thread started (interval=%.1fs).",
            self.flush_interval_s,
        )

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.flush_interval_s):
            with self._lock:
                self._flush_locked()
