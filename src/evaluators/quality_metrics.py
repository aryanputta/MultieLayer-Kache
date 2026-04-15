"""
quality_metrics.py — Task-quality evaluation for LongBench and related tasks.

Implements token-level F1 (SQuAD-style), exact match, ROUGE-L, and edit
similarity — the four metrics needed to cover all 16 LongBench tasks.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class QualityEvaluator:
    """Evaluates generation quality for diverse long-context tasks.

    Parameters
    ----------
    metrics:
        Subset of ``["f1", "em", "rouge_l", "edit_sim"]`` to compute.
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.metrics = metrics or ["f1", "em", "rouge_l", "edit_sim"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str = "qa",
    ) -> Dict[str, float]:
        """Evaluate a list of predictions against references.

        Parameters
        ----------
        predictions:
            Model-generated text strings.
        references:
            Gold-standard answer strings.  For tasks with multiple valid
            answers separate them with a pipe ``"|"`` character.
        task_type:
            One of ``"qa"``, ``"summarization"``, ``"code"``.

        Returns
        -------
        dict
            Requested metric names → mean scores.
        """
        assert len(predictions) == len(references), "Length mismatch."

        scores: Dict[str, List[float]] = {m: [] for m in self.metrics}

        for pred, ref in zip(predictions, references):
            # Support pipe-separated multiple reference answers
            refs = [r.strip() for r in ref.split("|") if r.strip()]
            if not refs:
                refs = [""]

            if "f1" in self.metrics:
                scores["f1"].append(max(self.token_f1(pred, r) for r in refs))
            if "em" in self.metrics:
                scores["em"].append(max(self.exact_match(pred, r) for r in refs))
            if "rouge_l" in self.metrics:
                scores["rouge_l"].append(max(self._rouge_l(pred, r) for r in refs))
            if "edit_sim" in self.metrics:
                scores["edit_sim"].append(max(self._edit_similarity(pred, r) for r in refs))

        return {m: round(float(np.mean(v)), 4) for m, v in scores.items() if v}

    def evaluate_batch(self, results: List[dict]) -> pd.DataFrame:
        """Evaluate a list of result dicts.

        Each dict must have ``prediction``, ``reference``, and optionally
        ``task_type``, ``workload_type``, ``request_id``.

        Returns a DataFrame with one row per result and one column per metric.
        """
        rows = []
        for r in results:
            pred = r.get("prediction", "")
            ref = r.get("reference", "")
            tt = r.get("task_type", "qa")
            single = self.evaluate([pred], [ref], task_type=tt)
            row = {
                "request_id": r.get("request_id", ""),
                "workload_type": r.get("workload_type", ""),
                "task_type": tt,
                **single,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def compute_aggregate(self, df: pd.DataFrame) -> dict:
        """Compute mean ± std per metric, optionally broken down by workload_type."""
        result: dict = {}
        metric_cols = [c for c in df.columns if c in self.metrics]

        for m in metric_cols:
            vals = df[m].dropna().tolist()
            result[m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "p5": round(float(np.percentile(vals, 5)), 4),
                "p95": round(float(np.percentile(vals, 95)), 4),
            }

        if "workload_type" in df.columns:
            per_workload: dict = {}
            for wt, group in df.groupby("workload_type"):
                per_workload[wt] = {}
                for m in metric_cols:
                    vals = group[m].dropna().tolist()
                    if vals:
                        per_workload[wt][m] = round(float(np.mean(vals)), 4)
            result["per_workload"] = per_workload

        return result

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def normalize_answer(self, text: str) -> str:
        """Lower-case, strip punctuation, articles, and extra whitespace.

        Mirrors the normalisation used in SQuAD / LongBench evaluations.
        """
        text = text.lower()
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Collapse whitespace
        return " ".join(text.split())

    def token_f1(self, prediction: str, reference: str) -> float:
        """Token-level F1 score (SQuAD style)."""
        pred_tokens = self.normalize_answer(prediction).split()
        ref_tokens = self.normalize_answer(reference).split()
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        common = sum((pred_counter & ref_counter).values())

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def exact_match(self, prediction: str, reference: str) -> float:
        """1.0 if normalised strings are identical, else 0.0."""
        return float(self.normalize_answer(prediction) == self.normalize_answer(reference))

    def _rouge_l(self, prediction: str, reference: str) -> float:
        """ROUGE-L F1 using the LCS algorithm."""
        pred_tokens = self.normalize_answer(prediction).split()
        ref_tokens = self.normalize_answer(reference).split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs_len = self._lcs_length(pred_tokens, ref_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _lcs_length(self, a: list, b: list) -> int:
        """Dynamic-programming LCS length."""
        m, n = len(a), len(b)
        # Space-optimised DP (O(n) space)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def _edit_similarity(self, prediction: str, reference: str) -> float:
        """Character-level edit similarity = 1 − edit_distance / max_len."""
        a = self.normalize_answer(prediction)
        b = self.normalize_answer(reference)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        dist = self._levenshtein(a, b)
        return 1.0 - dist / max(len(a), len(b))

    def _levenshtein(self, a: str, b: str) -> int:
        """Standard Levenshtein edit distance (character level)."""
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]
