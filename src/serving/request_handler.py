"""
request_handler.py — Async request queue manager and workload classifier.

:class:`RequestHandler` sits between the API layer and the vLLM engine,
tracking in-flight requests, measuring queue wait times, and classifying
prompts into :class:`WorkloadType` categories.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.telemetry.schema import WorkloadType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request status
# ---------------------------------------------------------------------------

class RequestStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RequestRecord:
    """Internal tracking object for a single serving request."""
    request_id: str
    prompt: str
    workload_type: WorkloadType
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: RequestStatus = RequestStatus.QUEUED
    error: Optional[str] = None

    @property
    def queue_wait_ms(self) -> float:
        if self.started_at is None:
            return (time.time() - self.submitted_at) * 1000
        return (self.started_at - self.submitted_at) * 1000

    @property
    def latency_ms(self) -> float:
        if self.completed_at is None or self.started_at is None:
            return 0.0
        return (self.completed_at - self.started_at) * 1000


# ---------------------------------------------------------------------------
# Workload classification heuristics
# ---------------------------------------------------------------------------

class WorkloadRouter:
    """Classify an incoming prompt into a :class:`WorkloadType` using
    lightweight keyword heuristics.  An optional sklearn classifier can be
    loaded for higher accuracy.
    """

    # Compiled regex patterns per workload type
    _QA_PATTERNS = re.compile(
        r"\b(answer the question|what is|who is|when did|where is|how many"
        r"|based on the (passage|document|context)|according to)\b",
        re.IGNORECASE,
    )
    _MULTI_DOC_PATTERNS = re.compile(
        r"\b(multiple (documents|sources|passages)|across (documents|sources)"
        r"|compare|contrast|both documents|all (three|four|five) documents)\b",
        re.IGNORECASE,
    )
    _SUMMARIZATION_PATTERNS = re.compile(
        r"\b(summarize|summary|tldr|brief overview|key points|main ideas"
        r"|in a few sentences|condense)\b",
        re.IGNORECASE,
    )
    _CODE_PATTERNS = re.compile(
        r"(def |class |import |from \w+ import|```python|```javascript"
        r"|function |public static|#include|package main)",
        re.IGNORECASE,
    )
    _DIALOGUE_PATTERNS = re.compile(
        r"\b(conversation|dialogue|chat|discussion|said|replied"
        r"|user:|assistant:|human:|bot:)\b",
        re.IGNORECASE,
    )
    _STRUCTURED_PATTERNS = re.compile(
        r"(\|\s*[-]+|^\s*\d+[\.\)]\s|\bcount\b|\blist\b|\btable\b"
        r"|\bclassif|\bcategor)",
        re.IGNORECASE,
    )

    def classify(
        self,
        prompt: str,
        task_metadata: Optional[dict] = None,
    ) -> WorkloadType:
        """Return the most likely :class:`WorkloadType` for *prompt*.

        Parameters
        ----------
        prompt:
            Raw prompt text (first 2048 chars are examined for efficiency).
        task_metadata:
            Optional dict with a ``"task"`` or ``"workload_type"`` key that
            overrides heuristic classification (used when the caller knows the
            task from a benchmark dataset).
        """
        if task_metadata:
            explicit = task_metadata.get("workload_type") or task_metadata.get("task")
            if explicit:
                try:
                    return WorkloadType(explicit)
                except ValueError:
                    pass

        excerpt = prompt[:2048]

        if self._CODE_PATTERNS.search(excerpt):
            return WorkloadType.CODE
        if self._MULTI_DOC_PATTERNS.search(excerpt):
            return WorkloadType.MULTI_DOC_QA
        if self._SUMMARIZATION_PATTERNS.search(excerpt):
            return WorkloadType.SUMMARIZATION
        if self._DIALOGUE_PATTERNS.search(excerpt):
            return WorkloadType.DIALOGUE
        if self._STRUCTURED_PATTERNS.search(excerpt):
            return WorkloadType.STRUCTURED_DATA
        if self._QA_PATTERNS.search(excerpt):
            return WorkloadType.SINGLE_DOC_QA

        # Fall back to length heuristic:
        # very long prompts without clear signal ≈ single-doc QA
        word_count = len(excerpt.split())
        return WorkloadType.SINGLE_DOC_QA if word_count > 200 else WorkloadType.UNKNOWN


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class RequestHandler:
    """Manages the request queue and in-flight request registry.

    Parameters
    ----------
    max_concurrent:
        Maximum number of requests that can be ``IN_PROGRESS`` simultaneously.
    router:
        :class:`WorkloadRouter` for prompt classification.
    """

    def __init__(
        self,
        max_concurrent: int = 32,
        router: Optional[WorkloadRouter] = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.router = router or WorkloadRouter()

        # request_id → RequestRecord
        self._records: Dict[str, RequestRecord] = {}
        self._lock = asyncio.Lock()

        # Semaphore to bound concurrency
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Completed request ring buffer (last 10k)
        self._completed: List[RequestRecord] = []
        self._max_completed = 10_000

    async def submit(
        self,
        prompt: str,
        task_metadata: Optional[dict] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Register a new request and return its ``request_id``."""
        rid = request_id or str(uuid.uuid4())
        wt = self.router.classify(prompt, task_metadata)
        record = RequestRecord(
            request_id=rid,
            prompt=prompt,
            workload_type=wt,
            submitted_at=time.time(),
        )
        async with self._lock:
            self._records[rid] = record
        logger.debug("Request %s submitted (workload=%s)", rid, wt.value)
        return rid

    async def mark_started(self, request_id: str) -> None:
        """Mark a request as IN_PROGRESS (called when dequeued by engine)."""
        async with self._lock:
            rec = self._records.get(request_id)
            if rec:
                rec.started_at = time.time()
                rec.status = RequestStatus.IN_PROGRESS

    async def mark_completed(
        self,
        request_id: str,
        error: Optional[str] = None,
    ) -> Optional[RequestRecord]:
        """Mark a request as COMPLETED or FAILED and archive it."""
        async with self._lock:
            rec = self._records.pop(request_id, None)
            if rec is None:
                return None
            rec.completed_at = time.time()
            rec.status = RequestStatus.FAILED if error else RequestStatus.COMPLETED
            rec.error = error
            self._completed.append(rec)
            if len(self._completed) > self._max_completed:
                self._completed = self._completed[-self._max_completed:]
        return rec

    def get_status(self, request_id: str) -> dict:
        """Return a status snapshot for *request_id*."""
        rec = self._records.get(request_id)
        if rec is None:
            # Check completed ring buffer
            for r in reversed(self._completed):
                if r.request_id == request_id:
                    rec = r
                    break
        if rec is None:
            return {"status": "not_found"}
        return {
            "request_id": rec.request_id,
            "status": rec.status.value,
            "workload_type": rec.workload_type.value,
            "queue_wait_ms": round(rec.queue_wait_ms, 2),
            "latency_ms": round(rec.latency_ms, 2),
        }

    def get_queue_depth(self) -> int:
        """Number of requests currently waiting or in progress."""
        return len(self._records)

    def get_active_count(self) -> int:
        """Number of requests currently IN_PROGRESS."""
        return sum(
            1 for r in self._records.values()
            if r.status == RequestStatus.IN_PROGRESS
        )

    def get_recent_completed(self, n: int = 100) -> List[RequestRecord]:
        """Return the *n* most recently completed requests."""
        return self._completed[-n:]

    def get_workload_distribution(self) -> Dict[str, int]:
        """Count active requests by workload type."""
        counts: Dict[str, int] = {}
        for r in self._records.values():
            key = r.workload_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
