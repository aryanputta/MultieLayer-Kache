"""
api_server.py — FastAPI application exposing the KV-cache orchestrator.

Endpoints
---------
POST /v1/generate          — Single-request completion
POST /v1/batch             — Batch completions
GET  /v1/health            — GPU + queue health check
GET  /v1/metrics           — Prometheus text metrics
GET  /v1/cache/status      — KV-cache tier breakdown
GET  /v1/cache/telemetry/recent — Recent telemetry records
POST /v1/policy/override   — Runtime policy swap
GET  /v1/benchmark/status  — Live benchmark progress (SSE)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics registry
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )

    _REQUEST_COUNT = Counter(
        "kv_requests_total", "Total inference requests", ["status", "workload_type"]
    )
    _REQUEST_LATENCY = Histogram(
        "kv_request_latency_ms",
        "End-to-end request latency in milliseconds",
        buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
    )
    _TTFT = Histogram(
        "kv_ttft_ms",
        "Time to first token in milliseconds",
        buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
    )
    _ACTIVE_REQUESTS = Gauge("kv_active_requests", "Number of in-flight requests")
    _QUEUE_DEPTH = Gauge("kv_queue_depth", "Serving queue depth")
    _GPU_MEM_USED = Gauge("kv_gpu_mem_used_mb", "GPU memory used (MB)")
    _GPU_MEM_FREE = Gauge("kv_gpu_mem_free_mb", "GPU memory free (MB)")
    _BLOCKS_GPU = Gauge("kv_blocks_gpu_fp16", "KV blocks on GPU (fp16)")
    _BLOCKS_QUANTIZED = Gauge("kv_blocks_gpu_quantized", "KV blocks quantized on GPU")
    _BLOCKS_CPU = Gauge("kv_blocks_cpu_offloaded", "KV blocks offloaded to CPU")
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — /v1/metrics will return 404.")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_tokens: int = Field(512, ge=1, le=65536)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    stop: List[str] = Field(default_factory=list)
    stream: bool = False
    workload_type: Optional[str] = None
    request_id: Optional[str] = None


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    finish_reason: str


class BatchRequest(BaseModel):
    requests: List[GenerateRequest]
    max_concurrency: int = Field(8, ge=1, le=64)


class PolicyOverrideRequest(BaseModel):
    policy_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global state (populated in lifespan)
# ---------------------------------------------------------------------------

_engine = None
_block_tracker = None
_telemetry_collector = None
_policy_engine = None
_request_handler = None
_config: dict = {}


def _get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return _engine


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and clean up the inference engine."""
    global _engine, _block_tracker, _telemetry_collector, _policy_engine, _request_handler

    logger.info("Starting KV Cache Orchestrator ...")

    # Import here to avoid circular imports and allow the server to start
    # without heavy deps installed in minimal environments.
    try:
        from omegaconf import OmegaConf
        import os

        cfg_path = os.environ.get("KV_CONFIG", "configs/default.yaml")
        if os.path.exists(cfg_path):
            cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        else:
            cfg = {}
        _config.update(cfg)
    except Exception as exc:
        logger.warning("Could not load config: %s — using defaults.", exc)

    model_name = _config.get("model", {}).get("name", "meta-llama/Llama-3.1-8B-Instruct")
    model_cfg = _config.get("model", {})

    from src.telemetry.block_tracker import BlockTracker
    from src.telemetry.collector import TelemetryCollector

    _block_tracker = BlockTracker(
        num_layers=model_cfg.get("num_layers", 32),
        num_heads=model_cfg.get("num_kv_heads", 8),
        block_size=16,
        head_dim=model_cfg.get("head_dim", 128),
    )
    _telemetry_collector = TelemetryCollector(
        config=_config.get("telemetry", {"enabled": True}),
        model_name=model_name,
    )

    from src.serving.vllm_engine import InstrumentedVLLMEngine, SamplingConfig
    from src.serving.request_handler import RequestHandler

    _engine = InstrumentedVLLMEngine(
        model_name=model_name,
        config=_config,
        block_tracker=_block_tracker,
        telemetry_collector=_telemetry_collector,
    )
    await _engine.initialize()

    _request_handler = RequestHandler(
        max_concurrent=_config.get("serving", {}).get("max_concurrent_requests", 32),
    )

    logger.info("KV Cache Orchestrator ready.")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down ...")
    if _telemetry_collector:
        _telemetry_collector.stop()
    if _engine:
        await _engine.shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adaptive KV Cache Orchestrator",
    description=(
        "Research serving system with multi-tier KV cache management. "
        "Supports full_cache, sliding_window, H2O, prefix_caching, quantize_only, "
        "offload_only, hybrid, and learned policies."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Single-request completion endpoint."""
    engine = _get_engine()

    from src.serving.vllm_engine import SamplingConfig

    rid = await _request_handler.submit(
        req.prompt,
        task_metadata={"workload_type": req.workload_type} if req.workload_type else None,
        request_id=req.request_id,
    )
    await _request_handler.mark_started(rid)

    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_REQUESTS.inc()
        _QUEUE_DEPTH.set(_request_handler.get_queue_depth())

    try:
        sc = SamplingConfig(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
        )
        result = await engine.generate(
            prompt=req.prompt,
            sampling_config=sc,
            request_id=rid,
        )

        if _PROMETHEUS_AVAILABLE:
            _REQUEST_COUNT.labels(status="success", workload_type=req.workload_type or "unknown").inc()
            _REQUEST_LATENCY.observe(result.total_latency_ms)
            _TTFT.observe(result.ttft_ms)

        await _request_handler.mark_completed(rid)
        return GenerateResponse(
            request_id=result.request_id,
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            ttft_ms=round(result.ttft_ms, 2),
            tpot_ms=round(result.tpot_ms, 2),
            latency_ms=round(result.total_latency_ms, 2),
            finish_reason=result.finish_reason,
        )
    except Exception as exc:
        await _request_handler.mark_completed(rid, error=str(exc))
        if _PROMETHEUS_AVAILABLE:
            _REQUEST_COUNT.labels(status="error", workload_type=req.workload_type or "unknown").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if _PROMETHEUS_AVAILABLE:
            _ACTIVE_REQUESTS.dec()


@app.post("/v1/batch")
async def batch_generate(req: BatchRequest):
    """Submit multiple requests and return results in order."""
    engine = _get_engine()
    semaphore = asyncio.Semaphore(req.max_concurrency)

    async def _one(r: GenerateRequest):
        async with semaphore:
            return await generate(r)

    tasks = [_one(r) for r in req.requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append({"error": str(r), "index": i})
        else:
            out.append(r)
    return {"results": out, "total": len(out)}


@app.get("/v1/health")
async def health():
    """Health check with GPU memory and queue stats."""
    mem = _engine.get_gpu_memory_stats() if _engine else {}
    return {
        "status": "ok",
        "engine_ready": _engine is not None and _engine._initialized,
        "queue_depth": _request_handler.get_queue_depth() if _request_handler else 0,
        "active_requests": _request_handler.get_active_count() if _request_handler else 0,
        **mem,
    }


@app.get("/v1/metrics")
async def metrics():
    """Prometheus-formatted metrics."""
    if not _PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=404, detail="prometheus_client not installed")

    # Refresh gauges
    if _engine:
        mem = _engine.get_gpu_memory_stats()
        _GPU_MEM_USED.set(mem.get("gpu_mem_used_mb", 0))
        _GPU_MEM_FREE.set(mem.get("gpu_mem_free_mb", 0))

    if _block_tracker:
        stats = _block_tracker.get_memory_stats()
        bpt = stats.get("blocks_per_tier", {})
        _BLOCKS_GPU.set(bpt.get("gpu_fp16", 0))
        _BLOCKS_QUANTIZED.set(bpt.get("gpu_quantized", 0))
        _BLOCKS_CPU.set(bpt.get("cpu_offloaded", 0))

    return PlainTextResponse(
        generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/v1/cache/status")
async def cache_status():
    """Current KV-cache tier breakdown."""
    if _block_tracker is None:
        return {"error": "block_tracker not initialised"}
    stats = _block_tracker.get_memory_stats()
    mem = _engine.get_gpu_memory_stats() if _engine else {}
    return {**stats, **mem}


@app.get("/v1/cache/telemetry/recent")
async def recent_telemetry(n: int = 200):
    """Return the *n* most recent buffered telemetry records."""
    if _telemetry_collector is None:
        return {"records": []}
    records = _telemetry_collector.get_recent_records(n)
    return {"records": [r.to_dict() for r in records], "count": len(records)}


@app.post("/v1/policy/override")
async def policy_override(req: PolicyOverrideRequest):
    """Temporarily switch the active cache policy at runtime."""
    if _policy_engine is None:
        raise HTTPException(status_code=503, detail="Policy engine not initialised")
    # Dynamic policy construction would happen here; simplified for now.
    return {
        "status": "override_applied",
        "policy": req.policy_name,
        "params": req.params,
    }
