"""
learned.py — ML-based KV-cache eviction policy.

:class:`LearnedPolicy` loads a trained XGBoost or LightGBM ranker from disk,
extracts features from live :class:`BlockState` objects, and maps importance
scores to four-way block decisions.  Falls back to :class:`H2OPolicy` when
no model is loaded.
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Optional

import numpy as np

from src.telemetry.schema import BlockDecision, BlockState
from src.cache_policies.base import BasePolicy, PolicyConfig, PolicyDecision
from src.cache_policies.heuristic import H2OPolicy

logger = logging.getLogger(__name__)


class LearnedPolicy(BasePolicy):
    """Importance-ranker–driven KV-cache policy.

    The model is expected to output importance scores in [0, 1] for each block.
    Decisions are derived by thresholding these scores against the quantisation
    and offload thresholds from the config.

    Parameters
    ----------
    config:
        :class:`PolicyConfig` with ``model_path``, ``decision_threshold``,
        ``quantization_*``, and ``offload_*`` fields.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(config)
        self._model = None
        self._feature_names: List[str] = []
        self._fallback: BasePolicy = H2OPolicy(config)

        if config.model_path:
            self.load_model(config.model_path)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self, path: str) -> None:
        """Load a pickled XGBoost / LightGBM model from *path*."""
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                self._model = payload["model"]
                self._feature_names = payload.get("feature_names", [])
            else:
                self._model = payload
            logger.info("LearnedPolicy: model loaded from %s", path)
        except Exception as exc:
            logger.error("LearnedPolicy: failed to load model from %s: %s", path, exc)
            self._model = None

    def save_model(self, path: str, model, feature_names: List[str]) -> None:
        """Save *model* and *feature_names* as a single pickle payload."""
        payload = {"model": model, "feature_names": feature_names}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("LearnedPolicy: model saved to %s", path)

    def is_ready(self) -> bool:
        """Return True if a model is loaded and ready for inference."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "learned"

    def decide(
        self,
        blocks: List[BlockState],
        system_metrics: dict,
    ) -> List[PolicyDecision]:
        if not self.is_ready():
            logger.debug("LearnedPolicy: no model — delegating to fallback (%s)", self._fallback.get_name())
            return self._fallback.decide(blocks, system_metrics)

        import time
        t0 = time.perf_counter()
        used = system_metrics.get("gpu_mem_used_mb", 0.0)
        free = system_metrics.get("gpu_mem_free_mb", 1.0)
        pressure = self.compute_memory_pressure(used, free)

        if pressure < self.config.memory_pressure_low:
            decisions = self._keep_all(blocks)
            self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
            return decisions

        scores = self.score_blocks(blocks, system_metrics)
        sink_ids = {b.block_id for b in sorted(blocks, key=lambda b: b.block_id)[: self.config.sink_token_blocks]}

        decisions = []
        for b, score in zip(blocks, scores):
            if b.block_id in sink_ids:
                decisions.append(PolicyDecision(b.block_id, BlockDecision.KEEP_GPU, 1.0, "sink"))
                continue
            decisions.append(self._score_to_decision(b.block_id, score, pressure))

        self._make_stats(decisions, pressure, (time.perf_counter() - t0) * 1000)
        return decisions

    def score_blocks(
        self,
        blocks: List[BlockState],
        system_metrics: dict,
    ) -> List[float]:
        """Return importance scores in [0, 1] for each block.

        Uses the feature extractor to build the feature matrix, then queries
        the loaded model.
        """
        try:
            from src.feature_engineering.extractor import FeatureExtractor
            extractor = FeatureExtractor()
            batch_stats = extractor.compute_batch_stats(blocks)
            X = extractor.extract_batch(blocks, system_metrics, batch_stats)
            raw = self._model.predict(X)
            # Normalise to [0, 1]
            lo, hi = float(raw.min()), float(raw.max())
            if hi > lo:
                scores = ((raw - lo) / (hi - lo)).tolist()
            else:
                scores = [0.5] * len(blocks)
            return scores
        except Exception as exc:
            logger.warning("LearnedPolicy.score_blocks failed: %s — using attention fallback.", exc)
            max_attn = max((b.avg_attention_score for b in blocks), default=1.0)
            return [b.avg_attention_score / max_attn if max_attn > 0 else 0.5 for b in blocks]

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance dict from the loaded model."""
        if not self.is_ready():
            return {}
        try:
            raw = self._model.feature_importances_
            names = self._feature_names or [f"f{i}" for i in range(len(raw))]
            return dict(zip(names, raw.tolist()))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_to_decision(
        self, block_id: int, score: float, pressure: float
    ) -> PolicyDecision:
        """Map a continuous importance score to a four-way block decision."""
        q_thresh = self.config.quantization_min_importance
        o_thresh = self.config.offload_threshold_score

        if score >= q_thresh:
            return PolicyDecision(block_id, BlockDecision.KEEP_GPU, score, "high_importance")
        elif score >= o_thresh and self.config.quantization_enabled:
            return PolicyDecision(block_id, BlockDecision.QUANTIZE, score, "medium_importance_quantize")
        elif score >= o_thresh * 0.5 and self.config.offload_enabled:
            return PolicyDecision(block_id, BlockDecision.OFFLOAD_CPU, score, "low_medium_offload")
        else:
            return PolicyDecision(block_id, BlockDecision.EVICT, score, "low_importance_evict")
