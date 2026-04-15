"""
system_metrics.py — GPU/CPU telemetry and latency distribution tracking.

:class:`SystemMetricsCollector` polls hardware stats in a background thread
and accumulates per-request latency measurements for percentile computation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Optional hardware polling
# --------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _gpu_memory_mb(device: int = 0) -> Tuple[float, float]:
    """Return (used_mb, free_mb).  Falls back to RAM if no GPU."""
    if _NVML:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device)
            i = pynvml.nvmlDeviceGetMemoryInfo(h)
            return i.used / 1024**2, i.free / 1024**2
        except Exception:
            pass
    if _PSUTIL:
        vm = psutil.virtual_memory()
        return vm.used / 1024**2, vm.available / 1024**2
    return 0.0, 0.0


def _gpu_utilization(device: int = 0) -> float:
    if _NVML:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            return float(u.gpu)
        except Exception:
            pass
    return 0.0


def _cpu_memory_mb() -> float:
    if _PSUTIL:
        return psutil.virtual_memory().used / 1024**2
    return 0.0


# --------------------------------------------------------------------------
# Latency tracker
# --------------------------------------------------------------------------

class LatencyTracker:
    """Sliding-window latency distribution tracker.

    Keeps the last *window_size* observations and computes percentiles in O(n).

    Parameters
    ----------
    window_size:
        Maximum number of observations to retain.
    """

    def __init__(self, window_size: int = 10_000) -> None:
        self._window: Deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, value: float) -> None:
        with self._lock:
            self._window.append(value)

    def percentile(self, p: float) -> float:
        """Return the *p*-th percentile (e.g. 95.0 for p95)."""
        with self._lock:
            if not self._window:
                return 0.0
            return float(np.percentile(list(self._window), p))

    def mean(self) -> float:
        with self._lock:
            return float(np.mean(list(self._window))) if self._window else 0.0

    def count(self) -> int:
        with self._lock:
            return len(self._window)

    def reset(self) -> None:
        with self._lock:
            self._window.clear()

    def get_all(self) -> List[float]:
        with self._lock:
            return list(self._window)


# --------------------------------------------------------------------------
# System metrics collector
# --------------------------------------------------------------------------

class SystemMetricsCollector:
    """Polls GPU/CPU stats and tracks per-request latency metrics.

    Parameters
    ----------
    polling_interval_ms:
        How often to sample hardware metrics.
    device_index:
        CUDA device to monitor.
    """

    def __init__(
        self,
        polling_interval_ms: float = 100.0,
        device_index: int = 0,
    ) -> None:
        self.polling_interval_ms = polling_interval_ms
        self.device_index = device_index

        # Latency trackers (ms)
        self.latency = LatencyTracker()
        self.ttft = LatencyTracker()
        self.tpot = LatencyTracker()

        # Token throughput tracking
        self._token_timestamps: Deque[Tuple[float, int]] = deque(maxlen=10_000)
        self._request_timestamps: Deque[float] = deque(maxlen=10_000)

        # Snapshot history
        self._snapshots: Deque[dict] = deque(maxlen=5_000)

        # Hardware poll thread
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current: dict = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Begin background hardware polling."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="sys-metrics-poll",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the polling thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def record_request(
        self,
        request_id: str,
        latency_ms: float,
        ttft_ms: float,
        tpot_ms: float,
        tokens_generated: int,
    ) -> None:
        """Record per-request metrics."""
        now = time.time()
        self.latency.record(latency_ms)
        self.ttft.record(ttft_ms)
        self.tpot.record(tpot_ms)
        with self._lock:
            self._token_timestamps.append((now, tokens_generated))
            self._request_timestamps.append(now)

    def get_current_metrics(self) -> dict:
        with self._lock:
            return dict(self._current)

    def get_summary(self, window_s: float = 60.0) -> dict:
        """Aggregate metrics over the last *window_s* seconds."""
        now = time.time()
        cutoff = now - window_s

        # Request rate
        recent_reqs = sum(1 for t in self._request_timestamps if t >= cutoff)
        rps = recent_reqs / window_s if window_s > 0 else 0.0

        # Token throughput
        recent_tokens = sum(
            toks for ts, toks in self._token_timestamps if ts >= cutoff
        )
        tps = recent_tokens / window_s if window_s > 0 else 0.0

        # Memory
        used, free = _gpu_memory_mb(self.device_index)

        return {
            "rps": round(rps, 2),
            "throughput_tps": round(tps, 2),
            "p50_latency_ms": round(self.latency.percentile(50), 2),
            "p95_latency_ms": round(self.latency.percentile(95), 2),
            "p99_latency_ms": round(self.latency.percentile(99), 2),
            "avg_ttft_ms": round(self.ttft.mean(), 2),
            "avg_tpot_ms": round(self.tpot.mean(), 2),
            "total_requests": self.latency.count(),
            "gpu_mem_used_mb": round(used, 1),
            "gpu_mem_free_mb": round(free, 1),
        }

    def compute_percentiles(
        self, values: List[float], percentiles: List[int]
    ) -> dict:
        """Compute requested percentiles for an arbitrary list of values."""
        if not values:
            return {f"p{p}": 0.0 for p in percentiles}
        arr = np.array(values)
        return {f"p{p}": round(float(np.percentile(arr, p)), 3) for p in percentiles}

    def export_prometheus(self) -> str:
        """Generate Prometheus text format for current system metrics."""
        s = self.get_summary()
        lines = [
            f"# HELP kv_throughput_tps Tokens per second",
            f"# TYPE kv_throughput_tps gauge",
            f"kv_throughput_tps {s['throughput_tps']}",
            f"# HELP kv_p95_latency_ms P95 end-to-end latency (ms)",
            f"# TYPE kv_p95_latency_ms gauge",
            f"kv_p95_latency_ms {s['p95_latency_ms']}",
            f"# HELP kv_gpu_mem_used_mb GPU memory used (MB)",
            f"# TYPE kv_gpu_mem_used_mb gauge",
            f"kv_gpu_mem_used_mb {s['gpu_mem_used_mb']}",
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.wait(timeout=self.polling_interval_ms / 1000.0):
            used, free = _gpu_memory_mb(self.device_index)
            gpu_util = _gpu_utilization(self.device_index)
            cpu_mem = _cpu_memory_mb()
            snapshot = {
                "timestamp": time.time(),
                "gpu_mem_used_mb": used,
                "gpu_mem_free_mb": free,
                "gpu_utilization_pct": gpu_util,
                "cpu_mem_used_mb": cpu_mem,
            }
            with self._lock:
                self._current = snapshot
                self._snapshots.append(snapshot)
