"""
stress_tests.py — Synthetic load and stress workloads.

Tests designed to expose policy failure modes under:
  - Bursty Poisson arrivals
  - Repeated prefix reuse
  - Long multi-turn chat sessions
  - Mixed context-length batches
  - Memory spike / rapid ramp-up scenarios
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    test_name: str
    policy_name: str
    config: Dict = field(default_factory=dict)
    duration_s: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    throughput_rps: float = 0.0
    throughput_tps: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    avg_gpu_mem_mb: float = 0.0
    blocks_evicted: int = 0
    blocks_quantized: int = 0
    blocks_offloaded: int = 0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def finalise(self) -> None:
        """Compute derived statistics from raw data."""
        if self.latencies_ms:
            arr = np.array(self.latencies_ms)
            self.p50_ms = float(np.percentile(arr, 50))
            self.p95_ms = float(np.percentile(arr, 95))
            self.p99_ms = float(np.percentile(arr, 99))
        total = self.total_requests
        if total > 0:
            self.error_rate = round(self.failed_requests / total, 4)
        if self.duration_s > 0:
            self.throughput_rps = round(self.successful_requests / self.duration_s, 2)


class StressTestRunner:
    """Runs synthetic stress workloads against the serving API.

    Parameters
    ----------
    api_url:
        Base URL of the serving API.
    policy_name:
        Name of the active policy (for labelling results).
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        policy_name: str = "unknown",
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.policy_name = policy_name

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    async def run_bursty_arrivals(
        self,
        arrival_rate: float = 2.0,
        burst_size: int = 4,
        duration_s: float = 120.0,
        context_length: int = 8192,
        max_tokens: int = 64,
    ) -> StressTestResult:
        """Simulate Poisson-distributed bursts of requests.

        Parameters
        ----------
        arrival_rate:
            Mean requests per second (Poisson parameter).
        burst_size:
            Number of requests fired simultaneously per arrival.
        duration_s:
            How long to run the test.
        context_length:
            Approximate prompt length in characters.
        """
        result = StressTestResult(
            test_name="bursty_arrivals",
            policy_name=self.policy_name,
            config={
                "arrival_rate": arrival_rate,
                "burst_size": burst_size,
                "context_length": context_length,
            },
        )
        prompt = self._random_prompt(context_length)
        t_start = time.perf_counter()
        tasks = []

        while (time.perf_counter() - t_start) < duration_s:
            # Poisson inter-arrival time
            wait = random.expovariate(arrival_rate)
            await asyncio.sleep(min(wait, duration_s))
            for _ in range(burst_size):
                tasks.append(asyncio.create_task(
                    self._call(prompt, max_tokens)
                ))

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        duration_s_actual = time.perf_counter() - t_start

        result.duration_s = duration_s_actual
        result.total_requests = len(responses)
        for r in responses:
            if isinstance(r, Exception):
                result.failed_requests += 1
            else:
                result.successful_requests += 1
                result.latencies_ms.append(r)

        result.finalise()
        await self._fetch_cache_stats(result)
        return result

    async def run_repeated_prefix_reuse(
        self,
        shared_prefix_chars: int = 4096,
        num_variations: int = 50,
        concurrency: int = 8,
        max_tokens: int = 64,
    ) -> StressTestResult:
        """Test prefix-caching effectiveness under heavy reuse.

        All requests share the same long prefix but have different suffixes.
        """
        result = StressTestResult(
            test_name="repeated_prefix_reuse",
            policy_name=self.policy_name,
            config={"shared_prefix_chars": shared_prefix_chars, "num_variations": num_variations},
        )
        prefix = self._random_prompt(shared_prefix_chars)
        prompts = [
            prefix + f"\n\nVariation {i}: What is the key idea?  Answer:"
            for i in range(num_variations)
        ]

        sem = asyncio.Semaphore(concurrency)
        t_start = time.perf_counter()

        async def _guarded(p):
            async with sem:
                return await self._call(p, max_tokens)

        responses = await asyncio.gather(*[_guarded(p) for p in prompts], return_exceptions=True)
        result.duration_s = time.perf_counter() - t_start
        result.total_requests = len(responses)
        for r in responses:
            if isinstance(r, Exception):
                result.failed_requests += 1
            else:
                result.successful_requests += 1
                result.latencies_ms.append(r)

        result.finalise()
        await self._fetch_cache_stats(result)
        return result

    async def run_long_chat_session(
        self,
        num_turns: int = 20,
        turn_tokens: int = 512,
        concurrency: int = 4,
        max_tokens: int = 128,
    ) -> StressTestResult:
        """Simulate concurrent long multi-turn dialogues."""
        result = StressTestResult(
            test_name="long_chat_session",
            policy_name=self.policy_name,
            config={"num_turns": num_turns, "concurrency": concurrency},
        )

        async def _session():
            history = ""
            lats = []
            for turn in range(num_turns):
                history += f"\nUser: Tell me more about topic {turn}.\nAssistant:"
                lat = await self._call(history, max_tokens)
                if isinstance(lat, float):
                    lats.append(lat)
                    history += f" [response {turn}]"
            return lats

        t_start = time.perf_counter()
        session_tasks = [_session() for _ in range(concurrency)]
        all_lats = await asyncio.gather(*session_tasks, return_exceptions=True)
        result.duration_s = time.perf_counter() - t_start

        for sl in all_lats:
            if isinstance(sl, Exception):
                result.failed_requests += 1
            elif isinstance(sl, list):
                result.latencies_ms.extend(sl)
                result.successful_requests += len(sl)
            result.total_requests += num_turns

        result.finalise()
        await self._fetch_cache_stats(result)
        return result

    async def run_mixed_context_batch(
        self,
        context_lengths: Optional[List[int]] = None,
        concurrency: int = 16,
        n_requests: int = 100,
        max_tokens: int = 64,
    ) -> StressTestResult:
        """Mixed batch of short and long context requests."""
        lengths = context_lengths or [512, 2048, 8192, 16384, 32768]
        result = StressTestResult(
            test_name="mixed_context_batch",
            policy_name=self.policy_name,
            config={"context_lengths": lengths, "n_requests": n_requests},
        )
        prompts = [
            self._random_prompt(random.choice(lengths))
            for _ in range(n_requests)
        ]
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(p):
            async with sem:
                return await self._call(p, max_tokens)

        t_start = time.perf_counter()
        responses = await asyncio.gather(*[_guarded(p) for p in prompts], return_exceptions=True)
        result.duration_s = time.perf_counter() - t_start
        result.total_requests = len(responses)
        for r in responses:
            if isinstance(r, Exception):
                result.failed_requests += 1
            else:
                result.successful_requests += 1
                result.latencies_ms.append(r)

        result.finalise()
        await self._fetch_cache_stats(result)
        return result

    async def run_memory_spike(
        self,
        ramp_up_s: float = 30.0,
        peak_concurrency: int = 32,
        peak_duration_s: float = 60.0,
        context_length: int = 16384,
        max_tokens: int = 64,
    ) -> StressTestResult:
        """Rapidly ramp to peak concurrency to stress memory management."""
        result = StressTestResult(
            test_name="memory_spike",
            policy_name=self.policy_name,
            config={
                "ramp_up_s": ramp_up_s,
                "peak_concurrency": peak_concurrency,
                "context_length": context_length,
            },
        )
        prompt = self._random_prompt(context_length)
        t_start = time.perf_counter()
        all_tasks = []

        # Ramp phase
        steps = max(1, int(ramp_up_s))
        for step in range(steps):
            n = max(1, int(peak_concurrency * step / steps))
            for _ in range(n):
                all_tasks.append(asyncio.create_task(self._call(prompt, max_tokens)))
            await asyncio.sleep(1.0)

        # Peak phase
        peak_end = time.perf_counter() + peak_duration_s
        while time.perf_counter() < peak_end:
            for _ in range(peak_concurrency):
                all_tasks.append(asyncio.create_task(self._call(prompt, max_tokens)))
            await asyncio.sleep(0.5)

        responses = await asyncio.gather(*all_tasks, return_exceptions=True)
        result.duration_s = time.perf_counter() - t_start
        result.total_requests = len(responses)
        for r in responses:
            if isinstance(r, Exception):
                result.failed_requests += 1
            else:
                result.successful_requests += 1
                result.latencies_ms.append(r)

        result.finalise()
        await self._fetch_cache_stats(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _call(self, prompt: str, max_tokens: int) -> float:
        """Send one request; return latency_ms or raise on error."""
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.api_url}/v1/generate",
                json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
            )
            resp.raise_for_status()
        return (time.perf_counter() - t0) * 1000

    async def _fetch_cache_stats(self, result: StressTestResult) -> None:
        """Try to fetch cache tier stats from the API and annotate the result."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.api_url}/v1/cache/status")
                if resp.status_code == 200:
                    data = resp.json()
                    bpt = data.get("blocks_per_tier", {})
                    result.blocks_evicted = bpt.get("evicted", 0)
                    result.blocks_quantized = bpt.get("gpu_quantized", 0)
                    result.blocks_offloaded = bpt.get("cpu_offloaded", 0)
                    result.peak_gpu_mem_mb = data.get("gpu_mem_used_mb", 0.0)
        except Exception:
            pass

    @staticmethod
    def _random_prompt(target_chars: int) -> str:
        """Generate a plausible-length prompt string."""
        words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "artificial", "intelligence", "machine", "learning", "language",
            "model", "inference", "context", "window", "attention", "token",
            "embedding", "transformer", "neural", "network", "performance",
        ]
        rng = random.Random(42)
        result = []
        while sum(len(w) + 1 for w in result) < target_chars:
            result.append(rng.choice(words))
        return " ".join(result)
