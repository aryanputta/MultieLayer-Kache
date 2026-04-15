"""
dashboard.py — FastAPI-based SSE live demo dashboard.

Provides a minimal HTML/JS frontend that streams:
  - Memory tier breakdown (GPU fp16 / quantised / CPU / evicted)
  - Recent policy decisions
  - Live latency percentiles

Mount on the main FastAPI app or run standalone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standalone FastAPI sub-app
# ---------------------------------------------------------------------------

dashboard_app = FastAPI(title="KV Cache Orchestrator — Live Dashboard")
dashboard_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Module-level references injected by the main server
_block_tracker = None
_policy_engine = None
_telemetry_collector = None
_metrics_collector = None


def configure(
    block_tracker=None,
    policy_engine=None,
    telemetry_collector=None,
    metrics_collector=None,
) -> None:
    """Inject runtime dependencies from the main serving app."""
    global _block_tracker, _policy_engine, _telemetry_collector, _metrics_collector
    _block_tracker = block_tracker
    _policy_engine = policy_engine
    _telemetry_collector = telemetry_collector
    _metrics_collector = metrics_collector


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

async def _sse_stream(gen: AsyncIterator[dict]) -> AsyncIterator[bytes]:
    async for payload in gen:
        yield f"data: {json.dumps(payload)}\n\n".encode()


async def _memory_events() -> AsyncIterator[dict]:
    """Yield tier breakdown snapshots every second."""
    while True:
        if _block_tracker is not None:
            stats = _block_tracker.get_memory_stats()
            yield {
                "type": "memory_tiers",
                "timestamp": time.time(),
                "blocks_per_tier": stats.get("blocks_per_tier", {}),
                "estimated_gpu_mb": round(stats.get("estimated_gpu_mb", 0.0), 1),
                "estimated_cpu_mb": round(stats.get("estimated_cpu_mb", 0.0), 1),
            }
        else:
            yield {"type": "memory_tiers", "timestamp": time.time(), "status": "not_ready"}
        await asyncio.sleep(1.0)


async def _decision_events() -> AsyncIterator[dict]:
    """Yield recent policy decision summaries every 2 seconds."""
    while True:
        if _policy_engine is not None:
            history = _policy_engine.get_decision_history(n=5)
            for s in history[-1:]:  # most recent only
                yield {
                    "type": "policy_decision",
                    "timestamp": s.timestamp,
                    "policy_name": s.policy_name,
                    "decisions": s.decisions,
                    "pressure": round(s.gpu_mem_pressure, 4),
                    "latency_ms": round(s.evaluation_latency_ms, 2),
                }
        else:
            yield {"type": "policy_decision", "timestamp": time.time(), "status": "not_ready"}
        await asyncio.sleep(2.0)


async def _latency_events() -> AsyncIterator[dict]:
    """Yield live latency percentiles every 3 seconds."""
    while True:
        if _metrics_collector is not None:
            summary = _metrics_collector.get_summary(window_s=30.0)
            yield {"type": "latency", "timestamp": time.time(), **summary}
        else:
            yield {"type": "latency", "timestamp": time.time(), "status": "not_ready"}
        await asyncio.sleep(3.0)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@dashboard_app.get("/stream/memory")
async def stream_memory():
    """Server-Sent Events: memory tier breakdown (1 Hz)."""
    return StreamingResponse(
        _sse_stream(_memory_events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@dashboard_app.get("/stream/decisions")
async def stream_decisions():
    """Server-Sent Events: policy decisions (0.5 Hz)."""
    return StreamingResponse(
        _sse_stream(_decision_events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@dashboard_app.get("/stream/latency")
async def stream_latency():
    """Server-Sent Events: latency percentiles (0.33 Hz)."""
    return StreamingResponse(
        _sse_stream(_latency_events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@dashboard_app.get("/snapshot")
async def snapshot():
    """Single JSON snapshot of current state."""
    mem = _block_tracker.get_memory_stats() if _block_tracker else {}
    policy = _policy_engine.get_current_stats() if _policy_engine else {}
    metrics = _metrics_collector.get_summary() if _metrics_collector else {}
    telemetry = (
        {"recent_count": len(_telemetry_collector.get_recent_records(100))}
        if _telemetry_collector else {}
    )
    return {
        "timestamp": time.time(),
        "memory": mem,
        "policy": policy,
        "metrics": metrics,
        "telemetry": telemetry,
    }


@dashboard_app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    """Minimal HTML+JS live dashboard."""
    return HTMLResponse(_DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Embedded HTML template
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KV Cache Orchestrator — Live Dashboard</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }
    header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d; }
    header h1 { margin: 0; font-size: 1.2rem; color: #58a6ff; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; padding: 24px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
    .card h2 { margin: 0 0 12px; font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric { font-size: 2rem; font-weight: bold; color: #58a6ff; }
    .sub { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
    .bar-row { display: flex; align-items: center; margin: 6px 0; }
    .bar-label { width: 120px; font-size: 0.8rem; }
    .bar-track { flex: 1; background: #21262d; border-radius: 4px; height: 12px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
    .fp16 { background: #58a6ff; }
    .quantized { background: #f0883e; }
    .offloaded { background: #3fb950; }
    .evicted { background: #f85149; }
    .log { font-family: monospace; font-size: 0.75rem; background: #0d1117; border-radius: 4px; padding: 10px; max-height: 180px; overflow-y: auto; }
    .log-entry { border-bottom: 1px solid #21262d; padding: 4px 0; }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3fb950; margin-right: 6px; }
  </style>
</head>
<body>
<header>
  <h1><span class="status-dot"></span>KV Cache Orchestrator — Live Dashboard</h1>
</header>
<div class="grid">

  <!-- Memory Tiers -->
  <div class="card" id="tier-card">
    <h2>Memory Tiers</h2>
    <div class="bar-row"><span class="bar-label">GPU fp16</span><div class="bar-track"><div class="bar-fill fp16" id="bar-fp16" style="width:0%"></div></div><span id="cnt-fp16" style="margin-left:8px;font-size:.8rem">0</span></div>
    <div class="bar-row"><span class="bar-label">GPU quantized</span><div class="bar-track"><div class="bar-fill quantized" id="bar-q" style="width:0%"></div></div><span id="cnt-q" style="margin-left:8px;font-size:.8rem">0</span></div>
    <div class="bar-row"><span class="bar-label">CPU offloaded</span><div class="bar-track"><div class="bar-fill offloaded" id="bar-cpu" style="width:0%"></div></div><span id="cnt-cpu" style="margin-left:8px;font-size:.8rem">0</span></div>
    <div class="bar-row"><span class="bar-label">Evicted</span><div class="bar-track"><div class="bar-fill evicted" id="bar-ev" style="width:0%"></div></div><span id="cnt-ev" style="margin-left:8px;font-size:.8rem">0</span></div>
    <div class="sub" id="mem-sub">GPU: 0 MB</div>
  </div>

  <!-- Latency -->
  <div class="card">
    <h2>Latency</h2>
    <div class="metric" id="p95-val">—</div>
    <div class="sub">p95 end-to-end (ms)</div>
    <div class="sub" id="lat-sub" style="margin-top:12px">p50: — | p99: — | TPS: —</div>
  </div>

  <!-- Policy Decisions -->
  <div class="card">
    <h2>Policy Decisions</h2>
    <div class="log" id="decision-log"></div>
  </div>

</div>

<script>
  function connectSSE(url, handler) {
    const es = new EventSource(url);
    es.onmessage = e => handler(JSON.parse(e.data));
    es.onerror = () => setTimeout(() => connectSSE(url, handler), 3000);
  }

  connectSSE('/stream/memory', d => {
    const bpt = d.blocks_per_tier || {};
    const total = Object.values(bpt).reduce((a,b)=>a+b,0) || 1;
    const set = (id, barId, key) => {
      const v = bpt[key] || 0;
      document.getElementById(id).textContent = v;
      document.getElementById(barId).style.width = (v/total*100).toFixed(1)+'%';
    };
    set('cnt-fp16','bar-fp16','gpu_fp16');
    set('cnt-q','bar-q','gpu_quantized');
    set('cnt-cpu','bar-cpu','cpu_offloaded');
    set('cnt-ev','bar-ev','evicted');
    document.getElementById('mem-sub').textContent =
      `GPU: ${(d.estimated_gpu_mb||0).toFixed(0)} MB | CPU: ${(d.estimated_cpu_mb||0).toFixed(0)} MB`;
  });

  connectSSE('/stream/latency', d => {
    document.getElementById('p95-val').textContent = (d.p95_latency_ms||0).toFixed(0);
    document.getElementById('lat-sub').textContent =
      `p50: ${(d.p50_latency_ms||0).toFixed(0)} ms | p99: ${(d.p99_latency_ms||0).toFixed(0)} ms | TPS: ${(d.throughput_tps||0).toFixed(1)}`;
  });

  connectSSE('/stream/decisions', d => {
    const log = document.getElementById('decision-log');
    if (d.status === 'not_ready') return;
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const dec = d.decisions || {};
    const parts = Object.entries(dec).map(([k,v])=>`${k}: ${v}`).join(' | ');
    entry.textContent = `[${new Date(d.timestamp*1000).toLocaleTimeString()}] ${d.policy_name} | pressure: ${(d.pressure||0).toFixed(3)} | ${parts}`;
    log.prepend(entry);
    while (log.children.length > 50) log.removeChild(log.lastChild);
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Standalone launcher
# ---------------------------------------------------------------------------

class LiveDashboard:
    """Convenience wrapper to run the dashboard as a standalone FastAPI app."""

    def __init__(
        self,
        block_tracker=None,
        policy_engine=None,
        telemetry_collector=None,
        metrics_collector=None,
        port: int = 8080,
    ) -> None:
        configure(block_tracker, policy_engine, telemetry_collector, metrics_collector)
        self.port = port

    def run(self, host: str = "0.0.0.0") -> None:
        import uvicorn
        uvicorn.run(dashboard_app, host=host, port=self.port, log_level="info")
