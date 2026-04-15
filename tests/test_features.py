"""
tests/test_features.py — Unit tests for feature engineering.
"""

import numpy as np
import pytest

from src.telemetry.schema import BlockState, BlockTier, WorkloadType
from src.feature_engineering.extractor import FeatureExtractor, FEATURE_NAMES
from src.feature_engineering.importance_scorer import ImportanceScorer
from src.feature_engineering.workload_classifier import WorkloadClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_block(block_id=0, avg_attn=0.5, max_attn=0.8, prefix_reuse=False):
    b = BlockState(
        block_id=block_id,
        layer_id=2,
        head_group=1,
        request_id="req1",
        block_age_steps=50,
        last_access_step=45,
        access_count=3,
        avg_attention_score=avg_attn,
        max_attention_score=max_attn,
        prefix_reuse_flag=prefix_reuse,
        current_tier=BlockTier.GPU_FP16,
        gpu_mem_used_mb=4000.0,
        gpu_mem_free_mb=4000.0,
    )
    return b


def make_system_metrics():
    return {
        "gpu_mem_used_mb": 4000.0,
        "gpu_mem_free_mb": 4000.0,
        "queue_depth": 4,
        "context_length": 8192,
        "output_length": 64,
        "workload_type": WorkloadType.SINGLE_DOC_QA.value,
        "global_step": 100,
        "attn_max": 1.0,
    }


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_extract_returns_correct_shape(self):
        extractor = FeatureExtractor()
        block = make_block()
        batch_stats = extractor.compute_batch_stats([block])
        vec = extractor.extract(block, make_system_metrics(), batch_stats)
        assert vec.shape == (len(FEATURE_NAMES),)

    def test_extract_all_finite(self):
        extractor = FeatureExtractor()
        block = make_block()
        batch_stats = extractor.compute_batch_stats([block])
        vec = extractor.extract(block, make_system_metrics(), batch_stats)
        assert np.all(np.isfinite(vec)), "Feature vector contains non-finite values"

    def test_extract_values_in_range(self):
        extractor = FeatureExtractor()
        block = make_block()
        batch_stats = extractor.compute_batch_stats([block])
        vec = extractor.extract(block, make_system_metrics(), batch_stats)
        # Most features should be in [0, 1]; allow small numerical slack
        assert vec.min() >= -0.01
        assert vec.max() <= 2.0  # raw attention scores can exceed 1 in some configs

    def test_extract_batch_shape(self):
        extractor = FeatureExtractor()
        blocks = [make_block(i) for i in range(8)]
        batch_stats = extractor.compute_batch_stats(blocks)
        X = extractor.extract_batch(blocks, make_system_metrics(), batch_stats)
        assert X.shape == (8, len(FEATURE_NAMES))

    def test_feature_names_match_vector_length(self):
        extractor = FeatureExtractor()
        assert len(extractor.get_feature_names()) == len(FEATURE_NAMES)

    def test_sink_token_flag_set_for_early_blocks(self):
        extractor = FeatureExtractor(sink_token_blocks=4)
        block_sink = make_block(block_id=0)
        block_non_sink = make_block(block_id=100)
        blocks = [block_sink, block_non_sink]
        batch_stats = extractor.compute_batch_stats(blocks)
        sm = make_system_metrics()
        vec_sink = extractor.extract(block_sink, sm, batch_stats)
        vec_non = extractor.extract(block_non_sink, sm, batch_stats)
        # Feature index 6 = sink_token_flag
        assert vec_sink[6] == pytest.approx(1.0)
        assert vec_non[6] == pytest.approx(0.0)

    def test_prefix_reuse_flag_propagated(self):
        extractor = FeatureExtractor()
        block_reuse = make_block(prefix_reuse=True)
        batch_stats = extractor.compute_batch_stats([block_reuse])
        vec = extractor.extract(block_reuse, make_system_metrics(), batch_stats)
        # Feature index 7 = prefix_reuse_flag
        assert vec[7] == pytest.approx(1.0)

    def test_batch_stats_returns_expected_keys(self):
        extractor = FeatureExtractor()
        blocks = [make_block(i) for i in range(5)]
        stats = extractor.compute_batch_stats(blocks)
        for key in ["max_age", "max_step", "max_attention", "total_blocks", "attn_sorted"]:
            assert key in stats

    def test_empty_batch_returns_empty_matrix(self):
        extractor = FeatureExtractor()
        X = extractor.extract_batch([], {}, {})
        assert X.shape[0] == 0


# ---------------------------------------------------------------------------
# ImportanceScorer
# ---------------------------------------------------------------------------

class TestImportanceScorer:
    def test_score_bounded_zero_one(self):
        scorer = ImportanceScorer()
        block = make_block(avg_attn=0.5)
        score = scorer.score_block(block)
        assert 0.0 <= score <= 1.0

    def test_high_attention_gives_high_score(self):
        scorer = ImportanceScorer()
        low = scorer.score_block(make_block(avg_attn=0.01, max_attn=0.01))
        high = scorer.score_block(make_block(avg_attn=0.99, max_attn=0.99))
        assert high > low

    def test_prefix_reuse_increases_score(self):
        scorer = ImportanceScorer()
        no_reuse = scorer.score_block(make_block(prefix_reuse=False, avg_attn=0.5))
        with_reuse = scorer.score_block(make_block(prefix_reuse=True, avg_attn=0.5))
        assert with_reuse >= no_reuse

    def test_score_batch_length(self):
        scorer = ImportanceScorer()
        blocks = [make_block(i) for i in range(6)]
        scores = scorer.score_batch(blocks)
        assert len(scores) == 6

    def test_pseudo_label_attention_mass_strategy(self):
        import pandas as pd
        scorer = ImportanceScorer()
        df = pd.DataFrame({
            "avg_attention_score": [0.1, 0.5, 0.9],
            "max_attention_score": [0.2, 0.6, 1.0],
            "prefix_reuse_flag":   [0,   1,   0],
            "access_count":        [1,   3,   5],
        })
        labeled = scorer.generate_pseudo_labels(df, strategy="attention_mass")
        assert "importance_score" in labeled.columns
        assert labeled["importance_score"].max() <= 1.0 + 1e-6

    def test_pseudo_label_combined_strategy(self):
        import pandas as pd
        scorer = ImportanceScorer()
        df = pd.DataFrame({
            "avg_attention_score": [0.1, 0.5, 0.9],
            "max_attention_score": [0.2, 0.6, 1.0],
            "prefix_reuse_flag":   [False, True, False],
            "access_count":        [1, 3, 5],
        })
        labeled = scorer.generate_pseudo_labels(df, strategy="combined")
        assert "importance_score" in labeled.columns
        assert all(labeled["importance_score"].between(0.0, 1.0 + 1e-4))


# ---------------------------------------------------------------------------
# WorkloadClassifier
# ---------------------------------------------------------------------------

class TestWorkloadClassifier:
    def test_code_detection(self):
        clf = WorkloadClassifier()
        wt = clf.classify_from_prompt("def foo(x):\n    return x + 1\n# Complete this:")
        assert wt == WorkloadType.CODE

    def test_summarization_detection(self):
        clf = WorkloadClassifier()
        wt = clf.classify_from_prompt(
            "Summarize the following article in a few sentences: ..."
        )
        assert wt == WorkloadType.SUMMARIZATION

    def test_multi_doc_detection(self):
        clf = WorkloadClassifier()
        wt = clf.classify_from_prompt(
            "Compare the findings across multiple documents and contrast their conclusions."
        )
        assert wt == WorkloadType.MULTI_DOC_QA

    def test_qa_detection(self):
        clf = WorkloadClassifier()
        wt = clf.classify_from_prompt(
            "Based on the passage, answer the question: Who founded the company?"
        )
        assert wt in (WorkloadType.SINGLE_DOC_QA, WorkloadType.MULTI_DOC_QA)

    def test_explicit_metadata_overrides_heuristic(self):
        clf = WorkloadClassifier()
        wt = clf.classify_from_prompt(
            "def foo(): pass",
            task_metadata={"workload_type": "summarization"},
        )
        assert wt == WorkloadType.SUMMARIZATION

    def test_classify_from_metrics_long_summarization(self):
        clf = WorkloadClassifier()
        wt = clf.classify_from_metrics({
            "prompt_tokens": 4000, "generated_tokens": 600, "prefix_reuse_flag": False
        })
        assert wt == WorkloadType.SUMMARIZATION
