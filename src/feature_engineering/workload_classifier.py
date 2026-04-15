"""
workload_classifier.py — Prompt-level workload type classifier.

Provides two modes:
- Keyword heuristic (no model required, fast, used in the serving hot path)
- Lightweight sklearn classifier (TF-IDF + LogisticRegression, optional)
"""

from __future__ import annotations

import logging
import pickle
import re
from typing import List, Optional

from src.telemetry.schema import WorkloadType

logger = logging.getLogger(__name__)


class WorkloadClassifier:
    """Classifies a prompt into a :class:`WorkloadType`.

    Parameters
    ----------
    model_path:
        Optional path to a pickled sklearn pipeline.  When ``None`` the
        classifier falls back to keyword heuristics.
    """

    _QA_RE = re.compile(
        r"\b(answer|what is|who is|when did|where|how many|based on|according to"
        r"|the question is|please answer)\b",
        re.IGNORECASE,
    )
    _MULTI_DOC_RE = re.compile(
        r"\b(multiple (documents|sources|passages|texts)|compare|contrast"
        r"|across documents|all (three|four|five) (documents|passages))\b",
        re.IGNORECASE,
    )
    _SUM_RE = re.compile(
        r"\b(summarize|summary|summarise|tldr|tl;dr|key points|brief overview"
        r"|main ideas|condense|in a few sentences)\b",
        re.IGNORECASE,
    )
    _CODE_RE = re.compile(
        r"(def |class |import |from \w+ import |```(python|javascript|java|go|cpp"
        r"|c\+\+|rust|ruby|ts)|function |public static|#include|package main|\bpip\b)",
    )
    _DIALOGUE_RE = re.compile(
        r"\b(conversation|dialogue|chat history|user:|assistant:|human:|bot:"
        r"|said|replied|asked)\b",
        re.IGNORECASE,
    )
    _STRUCT_RE = re.compile(
        r"(\|\s*[-]+|\bcount\b|\bclassif|\bcategor|\blist (all|the)|tabular"
        r"|\brank\b|\border\b.*\bby\b)",
        re.IGNORECASE,
    )

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._sklearn_model = None
        if model_path:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_from_prompt(
        self,
        prompt: str,
        task_metadata: Optional[dict] = None,
    ) -> WorkloadType:
        """Return the most likely :class:`WorkloadType` for *prompt*.

        If a trained sklearn model is loaded it takes precedence.
        Otherwise keyword heuristics are applied in priority order.
        """
        if task_metadata:
            explicit = task_metadata.get("workload_type") or task_metadata.get("task_type")
            if explicit:
                try:
                    return WorkloadType(explicit)
                except ValueError:
                    pass

        excerpt = prompt[:2048]

        if self._sklearn_model is not None:
            try:
                label = self._sklearn_model.predict([excerpt])[0]
                return WorkloadType(label)
            except Exception:
                pass

        # Keyword heuristic chain (highest precision first)
        if self._CODE_RE.search(excerpt):
            return WorkloadType.CODE
        if self._MULTI_DOC_RE.search(excerpt):
            return WorkloadType.MULTI_DOC_QA
        if self._SUM_RE.search(excerpt):
            return WorkloadType.SUMMARIZATION
        if self._DIALOGUE_RE.search(excerpt):
            return WorkloadType.DIALOGUE
        if self._STRUCT_RE.search(excerpt):
            return WorkloadType.STRUCTURED_DATA
        if self._QA_RE.search(excerpt):
            return WorkloadType.SINGLE_DOC_QA

        # Length heuristic: long prompts without a clear signal ≈ QA
        return WorkloadType.SINGLE_DOC_QA if len(excerpt.split()) > 150 else WorkloadType.UNKNOWN

    def classify_from_metrics(self, metrics: dict) -> WorkloadType:
        """Heuristic classification from request-level metrics.

        Parameters
        ----------
        metrics:
            Dict with ``prompt_tokens``, ``generated_tokens``,
            ``prefix_reuse_flag``.
        """
        prompt_tokens = metrics.get("prompt_tokens", 0)
        gen_tokens = metrics.get("generated_tokens", 0)
        reuse = metrics.get("prefix_reuse_flag", False)

        if reuse and prompt_tokens > 4000:
            return WorkloadType.CODE  # likely repo context
        if gen_tokens > 500 and prompt_tokens > 2000:
            return WorkloadType.SUMMARIZATION
        if prompt_tokens > 8000:
            return WorkloadType.MULTI_DOC_QA
        if gen_tokens < 50 and prompt_tokens > 500:
            return WorkloadType.SINGLE_DOC_QA
        return WorkloadType.UNKNOWN

    # ------------------------------------------------------------------
    # Training (optional sklearn path)
    # ------------------------------------------------------------------

    def train(
        self,
        prompts: List[str],
        labels: List[WorkloadType],
        max_features: int = 5000,
    ) -> dict:
        """Train a TF-IDF + LogisticRegression classifier.

        Parameters
        ----------
        prompts:
            List of raw prompt strings.
        labels:
            Corresponding :class:`WorkloadType` ground-truth labels.
        max_features:
            Vocabulary size for the TF-IDF vectoriser.

        Returns
        -------
        dict
            Training summary with ``train_accuracy`` and ``classes``.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder

        label_strings = [l.value for l in labels]
        excerpts = [p[:1024] for p in prompts]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
        ])
        pipeline.fit(excerpts, label_strings)
        train_preds = pipeline.predict(excerpts)
        acc = sum(p == t for p, t in zip(train_preds, label_strings)) / len(label_strings)
        self._sklearn_model = pipeline
        logger.info("WorkloadClassifier trained: accuracy=%.3f, n=%d", acc, len(prompts))
        return {"train_accuracy": acc, "classes": list(set(label_strings))}

    def save(self, path: str) -> None:
        """Pickle the sklearn model to *path*."""
        if self._sklearn_model is None:
            raise RuntimeError("No model to save — call train() first.")
        with open(path, "wb") as f:
            pickle.dump(self._sklearn_model, f)
        logger.info("WorkloadClassifier saved to %s", path)

    def load(self, path: str) -> None:
        """Load a pickled sklearn model from *path*."""
        try:
            with open(path, "rb") as f:
                self._sklearn_model = pickle.load(f)
            logger.info("WorkloadClassifier loaded from %s", path)
        except Exception as exc:
            logger.warning("Could not load WorkloadClassifier from %s: %s", path, exc)
