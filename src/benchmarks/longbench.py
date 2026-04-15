"""
longbench.py — LongBench evaluation runner.

Loads tasks from the THUDM/LongBench HuggingFace dataset, sends requests to
the serving API, evaluates quality with task-appropriate metrics, and returns
structured :class:`BenchmarkResult` objects.

Supported tasks (all 16):
    narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique,
    gov_report, qmsum, multi_news, trec, triviaqa, samsum,
    passage_count, passage_retrieval_en, lcc, repobench-p
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from src.telemetry.schema import WorkloadType
from src.evaluators.quality_metrics import QualityEvaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task metadata tables
# ---------------------------------------------------------------------------

TASK_TO_WORKLOAD: Dict[str, WorkloadType] = {
    "narrativeqa":          WorkloadType.SINGLE_DOC_QA,
    "qasper":               WorkloadType.SINGLE_DOC_QA,
    "multifieldqa_en":      WorkloadType.MULTI_DOC_QA,
    "hotpotqa":             WorkloadType.MULTI_DOC_QA,
    "2wikimqa":             WorkloadType.MULTI_DOC_QA,
    "musique":              WorkloadType.MULTI_DOC_QA,
    "gov_report":           WorkloadType.SUMMARIZATION,
    "qmsum":                WorkloadType.SUMMARIZATION,
    "multi_news":           WorkloadType.SUMMARIZATION,
    "trec":                 WorkloadType.STRUCTURED_DATA,
    "triviaqa":             WorkloadType.SINGLE_DOC_QA,
    "samsum":               WorkloadType.DIALOGUE,
    "passage_count":        WorkloadType.STRUCTURED_DATA,
    "passage_retrieval_en": WorkloadType.STRUCTURED_DATA,
    "lcc":                  WorkloadType.CODE,
    "repobench-p":          WorkloadType.CODE,
}

TASK_METRIC: Dict[str, str] = {
    "narrativeqa":          "rouge_l",
    "qasper":               "f1",
    "multifieldqa_en":      "f1",
    "hotpotqa":             "f1",
    "2wikimqa":             "f1",
    "musique":              "f1",
    "gov_report":           "rouge_l",
    "qmsum":                "rouge_l",
    "multi_news":           "rouge_l",
    "trec":                 "em",
    "triviaqa":             "f1",
    "samsum":               "rouge_l",
    "passage_count":        "em",
    "passage_retrieval_en": "em",
    "lcc":                  "edit_sim",
    "repobench-p":          "edit_sim",
}

# Prompt templates (simplified; real templates should follow LongBench paper)
_TEMPLATES: Dict[str, str] = {
    "default": "Read the following text and answer the question.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    "summarization": "Summarise the following text in a few sentences.\n\nText: {context}\n\nSummary:",
    "code": "Complete the following code:\n\n{context}\n\n{input}",
    "dialogue": "The following is a conversation. Answer the question based on it.\n\n{context}\n\nQuestion: {input}\n\nAnswer:",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_name: str
    workload_type: WorkloadType
    policy_name: str
    memory_budget: float
    n_samples: int
    quality_score: float                    # primary metric value
    primary_metric: str                     # e.g. "f1"
    quality_breakdown: Dict[str, float] = field(default_factory=dict)
    system_metrics: Dict = field(default_factory=dict)
    latencies_ms: List[float] = field(default_factory=list)
    failed_requests: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["workload_type"] = self.workload_type.value
        return d


@dataclass
class BenchmarkResult:
    policy_name: str
    memory_budget: float
    task_results: List[TaskResult] = field(default_factory=list)
    aggregate_quality: float = 0.0
    system_summary: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    config: Dict = field(default_factory=dict)

    def compute_aggregate(self) -> None:
        """Update aggregate_quality as macro-average across tasks."""
        scores = [t.quality_score for t in self.task_results if t.quality_score >= 0]
        self.aggregate_quality = float(np.mean(scores)) if scores else 0.0

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for t in self.task_results:
            row = t.to_dict()
            row["aggregate_quality"] = self.aggregate_quality
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            data = {
                "policy_name": self.policy_name,
                "memory_budget": self.memory_budget,
                "aggregate_quality": self.aggregate_quality,
                "system_summary": self.system_summary,
                "timestamp": self.timestamp,
                "config": self.config,
                "task_results": [t.to_dict() for t in self.task_results],
            }
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkResult":
        with open(path) as f:
            data = json.load(f)
        task_results = [
            TaskResult(
                **{k: (WorkloadType(v) if k == "workload_type" else v)
                   for k, v in t.items()}
            )
            for t in data.pop("task_results", [])
        ]
        return cls(task_results=task_results, **data)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class LongBenchRunner:
    """Runs LongBench evaluation against a running API server.

    Parameters
    ----------
    api_url:
        Base URL of the serving API, e.g. ``"http://localhost:8000"``.
    config:
        Benchmark config dict (``config["benchmark"]["longbench"]`` section).
    policy_name:
        Name of the policy under test (for result labelling).
    memory_budget:
        Active GPU memory budget fraction (for result labelling).
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        config: Optional[dict] = None,
        policy_name: str = "unknown",
        memory_budget: float = 1.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.config = config or {}
        self.policy_name = policy_name
        self.memory_budget = memory_budget
        self.evaluator = QualityEvaluator()

        lb_cfg = self.config.get("longbench", {})
        self.max_samples = lb_cfg.get("max_samples_per_task", 200)
        self.max_input_tokens = lb_cfg.get("max_input_tokens", 31500)
        self.dataset_name = lb_cfg.get("dataset_name", "THUDM/LongBench")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_task(
        self, task_name: str, max_samples: Optional[int] = None
    ) -> List[dict]:
        """Load samples for *task_name* from the HuggingFace dataset.

        Falls back to a small synthetic dataset if HuggingFace is unavailable.
        """
        n = max_samples or self.max_samples
        try:
            from datasets import load_dataset
            ds = load_dataset(self.dataset_name, task_name, split="test", trust_remote_code=True)
            samples = list(ds.select(range(min(n, len(ds)))))
            return [self._preprocess_sample(s, task_name) for s in samples]
        except Exception as exc:
            logger.warning(
                "Could not load %s/%s from HuggingFace (%s) — using synthetic fallback.",
                self.dataset_name, task_name, exc,
            )
            return self._synthetic_samples(task_name, n)

    def _preprocess_sample(self, sample: dict, task_name: str) -> dict:
        """Normalise a raw HuggingFace sample into {prompt, reference}."""
        ctx = sample.get("context", sample.get("input", ""))
        question = sample.get("input", "")
        answers = sample.get("answers", sample.get("answer", ""))
        if isinstance(answers, list):
            reference = " | ".join(str(a) for a in answers)
        else:
            reference = str(answers)

        # Truncate context to max_input_tokens (rough character proxy)
        max_chars = self.max_input_tokens * 4
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars]

        workload = TASK_TO_WORKLOAD.get(task_name, WorkloadType.UNKNOWN)
        if workload in (WorkloadType.SUMMARIZATION,):
            template = _TEMPLATES["summarization"]
            prompt = template.format(context=ctx, input=question)
        elif workload == WorkloadType.CODE:
            template = _TEMPLATES["code"]
            prompt = template.format(context=ctx, input=question)
        elif workload == WorkloadType.DIALOGUE:
            template = _TEMPLATES["dialogue"]
            prompt = template.format(context=ctx, input=question)
        else:
            template = _TEMPLATES["default"]
            prompt = template.format(context=ctx, input=question)

        return {
            "prompt": prompt,
            "reference": reference,
            "task_name": task_name,
            "workload_type": workload.value,
        }

    def _synthetic_samples(self, task_name: str, n: int) -> List[dict]:
        """Minimal synthetic fallback for CI / offline testing."""
        wt = TASK_TO_WORKLOAD.get(task_name, WorkloadType.UNKNOWN)
        return [
            {
                "prompt": f"This is a synthetic {task_name} prompt {i}. Answer the question.",
                "reference": f"synthetic answer {i}",
                "task_name": task_name,
                "workload_type": wt.value,
            }
            for i in range(min(n, 10))
        ]

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def run_task_async(
        self,
        task_name: str,
        max_samples: Optional[int] = None,
        concurrency: int = 4,
        max_tokens: int = 128,
    ) -> TaskResult:
        """Run *task_name* asynchronously against the serving API."""
        samples = self.load_task(task_name, max_samples)
        if not samples:
            return self._empty_result(task_name)

        sem = asyncio.Semaphore(concurrency)
        predictions: List[str] = [""] * len(samples)
        latencies: List[float] = []
        failed = 0

        async def _call(idx: int, sample: dict) -> None:
            nonlocal failed
            async with sem:
                try:
                    t0 = time.perf_counter()
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        resp = await client.post(
                            f"{self.api_url}/v1/generate",
                            json={
                                "prompt": sample["prompt"],
                                "max_tokens": max_tokens,
                                "temperature": 0.0,
                                "workload_type": sample["workload_type"],
                            },
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        predictions[idx] = data.get("text", "")
                        latencies.append((time.perf_counter() - t0) * 1000)
                except Exception as exc:
                    logger.debug("Request %d failed: %s", idx, exc)
                    failed += 1

        tasks = [_call(i, s) for i, s in enumerate(samples)]
        await asyncio.gather(*tasks)

        references = [s["reference"] for s in samples]
        primary_metric = TASK_METRIC.get(task_name, "f1")
        quality = self.evaluator.evaluate(predictions, references)
        score = quality.get(primary_metric, 0.0)

        return TaskResult(
            task_name=task_name,
            workload_type=TASK_TO_WORKLOAD.get(task_name, WorkloadType.UNKNOWN),
            policy_name=self.policy_name,
            memory_budget=self.memory_budget,
            n_samples=len(samples),
            quality_score=score,
            primary_metric=primary_metric,
            quality_breakdown=quality,
            latencies_ms=latencies,
            failed_requests=failed,
        )

    def run_task(self, task_name: str, **kwargs) -> TaskResult:
        """Synchronous wrapper around :meth:`run_task_async`."""
        return asyncio.run(self.run_task_async(task_name, **kwargs))

    async def run_all_async(
        self,
        tasks: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        concurrency: int = 4,
    ) -> BenchmarkResult:
        """Run all requested tasks sequentially and return a :class:`BenchmarkResult`."""
        task_names = tasks or list(TASK_TO_WORKLOAD.keys())
        result = BenchmarkResult(
            policy_name=self.policy_name,
            memory_budget=self.memory_budget,
            config=self.config,
        )
        for tn in task_names:
            logger.info("LongBench: running %s …", tn)
            tr = await self.run_task_async(tn, max_samples=max_samples, concurrency=concurrency)
            result.task_results.append(tr)
            logger.info("  %s: %s = %.4f", tn, tr.primary_metric, tr.quality_score)
        result.compute_aggregate()
        return result

    def run_all(self, **kwargs) -> BenchmarkResult:
        """Synchronous wrapper around :meth:`run_all_async`."""
        return asyncio.run(self.run_all_async(**kwargs))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self, task_name: str) -> TaskResult:
        return TaskResult(
            task_name=task_name,
            workload_type=TASK_TO_WORKLOAD.get(task_name, WorkloadType.UNKNOWN),
            policy_name=self.policy_name,
            memory_budget=self.memory_budget,
            n_samples=0,
            quality_score=-1.0,
            primary_metric=TASK_METRIC.get(task_name, "f1"),
        )
