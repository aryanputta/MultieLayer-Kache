"""
vllm_engine.py — Instrumented wrapper around vLLM's AsyncLLMEngine.

:class:`InstrumentedVLLMEngine` intercepts token generation to:
  - measure TTFT and TPOT
  - update :class:`BlockTracker` with attention statistics
  - trigger the policy engine on a configurable interval
  - emit :class:`TelemetryRecord` events via :class:`TelemetryCollector`
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import of vLLM — the engine is optional so tests can run without it
# ---------------------------------------------------------------------------
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.outputs import RequestOutput
    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False
    AsyncLLMEngine = None  # type: ignore[assignment,misc]
    AsyncEngineArgs = None  # type: ignore[assignment,misc]
    SamplingParams = None   # type: ignore[assignment,misc]
    RequestOutput = None    # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# GPU memory polling (pynvml → psutil fallback)
# ---------------------------------------------------------------------------
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _get_gpu_memory_mb(device_index: int = 0) -> tuple[float, float]:
    """Return (used_mb, free_mb) for *device_index*."""
    if _NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024**2, info.free / 1024**2
        except Exception:
            pass
    if _PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        used = vm.used / 1024**2
        free = vm.available / 1024**2
        return used, free
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    """Generation sampling parameters."""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: List[str] = field(default_factory=list)
    repetition_penalty: float = 1.0


@dataclass
class GenerationResult:
    """Output from a single generation call."""
    request_id: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class InstrumentedVLLMEngine:
    """Thin instrumentation layer on top of :class:`vllm.AsyncLLMEngine`.

    If vLLM is not installed the class degrades to a mock that returns
    placeholder text — this allows unit tests and CI to run without a GPU.

    Parameters
    ----------
    model_name:
        HuggingFace model ID (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``).
    config:
        Full project config dict; ``config["model"]`` and
        ``config["serving"]`` sections are consumed.
    block_tracker:
        :class:`~src.telemetry.BlockTracker` instance shared with the
        policy engine.
    telemetry_collector:
        :class:`~src.telemetry.TelemetryCollector` for event emission.
    policy_engine:
        Optional policy engine; if provided, ``trigger_policy_check`` will
        invoke it.
    device_index:
        CUDA device index for GPU memory polling.
    """

    def __init__(
        self,
        model_name: str,
        config: dict,
        block_tracker=None,
        telemetry_collector=None,
        policy_engine=None,
        device_index: int = 0,
    ) -> None:
        self.model_name = model_name
        self.config = config
        self.block_tracker = block_tracker
        self.telemetry_collector = telemetry_collector
        self.policy_engine = policy_engine
        self.device_index = device_index

        self._engine: Optional[AsyncLLMEngine] = None
        self._policy_interval_ms: float = float(
            config.get("policy", {}).get("scheduling_interval_ms", 50)
        )
        self._last_policy_check: float = 0.0

        # Lightweight per-request tracking: request_id → start_time
        self._in_flight: Dict[str, float] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Build the vLLM engine (must be awaited before ``generate()``)."""
        if not _VLLM_AVAILABLE:
            logger.warning(
                "vLLM not available — using mock engine. "
                "Install vllm for real inference."
            )
            self._initialized = True
            return

        model_cfg = self.config.get("model", {})
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=model_cfg.get("dtype", "float16"),
            max_model_len=model_cfg.get("max_model_len", 32768),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.85),
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
            enforce_eager=model_cfg.get("enforce_eager", False),
            swap_space=model_cfg.get("swap_space", 4),
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._initialized = True
        logger.info("vLLM engine initialised for model: %s", self.model_name)

    async def generate(
        self,
        prompt: str,
        sampling_config: Optional[SamplingConfig] = None,
        request_id: Optional[str] = None,
        workload_type=None,
    ) -> GenerationResult:
        """Generate a completion for *prompt*.

        Parameters
        ----------
        prompt:
            The full (possibly tokenised) prompt string.
        sampling_config:
            Generation parameters; defaults to greedy decode.
        request_id:
            Optional identifier; a UUID is generated if not provided.
        workload_type:
            Workload class used for telemetry annotation.

        Returns
        -------
        GenerationResult
            Final completion text with timing breakdowns.
        """
        if not self._initialized:
            await self.initialize()

        rid = request_id or str(uuid.uuid4())
        sc = sampling_config or SamplingConfig()

        start = time.perf_counter()
        self._in_flight[rid] = start

        if not _VLLM_AVAILABLE or self._engine is None:
            return await self._mock_generate(rid, prompt, sc, start)

        sp = SamplingParams(
            temperature=sc.temperature,
            top_p=sc.top_p,
            max_tokens=sc.max_tokens,
            stop=sc.stop,
            repetition_penalty=sc.repetition_penalty,
        )

        ttft_ms: float = -1.0
        output_text: str = ""
        finish_reason: str = "stop"
        prompt_tokens: int = 0
        generated_tokens: int = 0

        async for output in self._engine.generate(prompt, sp, request_id=rid):
            output: RequestOutput
            if ttft_ms < 0 and output.outputs:
                ttft_ms = (time.perf_counter() - start) * 1000

            if output.finished:
                best = output.outputs[0]
                output_text = best.text
                finish_reason = best.finish_reason or "stop"
                prompt_tokens = len(output.prompt_token_ids or [])
                generated_tokens = len(best.token_ids or [])

        total_ms = (time.perf_counter() - start) * 1000
        tpot_ms = (
            (total_ms - ttft_ms) / max(generated_tokens, 1)
            if generated_tokens > 0
            else 0.0
        )

        self._in_flight.pop(rid, None)

        # Trigger policy check if interval has elapsed
        await self._maybe_trigger_policy()

        return GenerationResult(
            request_id=rid,
            text=output_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=max(ttft_ms, 0.0),
            tpot_ms=tpot_ms,
            total_latency_ms=total_ms,
            finish_reason=finish_reason,
        )

    async def stream_generate(
        self,
        prompt: str,
        sampling_config: Optional[SamplingConfig] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Yield token strings incrementally.

        Useful for SSE / streaming API endpoints.
        """
        if not self._initialized:
            await self.initialize()

        rid = request_id or str(uuid.uuid4())
        sc = sampling_config or SamplingConfig()
        start = time.perf_counter()
        self._in_flight[rid] = start

        if not _VLLM_AVAILABLE or self._engine is None:
            # Mock streaming
            mock = f"[mock response for request {rid}]"
            for tok in mock.split():
                yield tok + " "
                await asyncio.sleep(0)
            return

        sp = SamplingParams(
            temperature=sc.temperature,
            top_p=sc.top_p,
            max_tokens=sc.max_tokens,
            stop=sc.stop,
        )

        prev_len = 0
        async for output in self._engine.generate(prompt, sp, request_id=rid):
            if output.outputs:
                text = output.outputs[0].text
                delta = text[prev_len:]
                if delta:
                    yield delta
                prev_len = len(text)

        self._in_flight.pop(rid, None)
        await self._maybe_trigger_policy()

    def get_gpu_memory_stats(self) -> dict:
        """Return current GPU memory usage as a dict."""
        used, free = _get_gpu_memory_mb(self.device_index)
        total = used + free
        return {
            "gpu_mem_used_mb": used,
            "gpu_mem_free_mb": free,
            "gpu_mem_total_mb": total,
            "gpu_mem_pressure": used / total if total > 0 else 0.0,
        }

    async def trigger_policy_check(self) -> None:
        """Manually invoke the policy engine on all current blocks."""
        if self.policy_engine is not None:
            try:
                await self.policy_engine._run_policy_cycle()
            except Exception as exc:
                logger.warning("Policy cycle error: %s", exc)

    async def shutdown(self) -> None:
        """Cleanly shut down the vLLM engine."""
        if self._engine is not None:
            await self._engine.abort_all_requests()  # type: ignore[attr-defined]
        logger.info("vLLM engine shut down.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _maybe_trigger_policy(self) -> None:
        now = time.perf_counter() * 1000
        if now - self._last_policy_check >= self._policy_interval_ms:
            self._last_policy_check = now
            await self.trigger_policy_check()

    async def _mock_generate(
        self,
        rid: str,
        prompt: str,
        sc: SamplingConfig,
        start: float,
    ) -> GenerationResult:
        """Return a placeholder result when vLLM is unavailable."""
        await asyncio.sleep(0.01)  # simulate minimal latency
        total_ms = (time.perf_counter() - start) * 1000
        self._in_flight.pop(rid, None)
        return GenerationResult(
            request_id=rid,
            text=f"[MOCK] Response to: {prompt[:80]}",
            prompt_tokens=len(prompt.split()),
            generated_tokens=8,
            ttft_ms=total_ms * 0.2,
            tpot_ms=total_ms * 0.8 / 8,
            total_latency_ms=total_ms,
            finish_reason="stop",
        )
