"""
tests/test_telemetry.py — Unit tests for the telemetry layer.
"""

import time
import tempfile

import numpy as np
import pytest

from src.telemetry.schema import (
    BlockDecision,
    BlockState,
    BlockTier,
    TelemetryRecord,
    WorkloadType,
    RequestMetrics,
)
from src.telemetry.block_tracker import BlockTracker
from src.telemetry.collector import TelemetryCollector
from src.telemetry.storage import TelemetryStorage


# ---------------------------------------------------------------------------
# BlockTracker
# ---------------------------------------------------------------------------

class TestBlockTracker:
    def _make_tracker(self):
        return BlockTracker(num_layers=4, num_heads=4, block_size=16, head_dim=64)

    def test_register_and_get(self):
        tracker = self._make_tracker()
        block = tracker.register_block(block_id=1, layer_id=0, head_group=0, request_id="req1")
        assert block.block_id == 1
        assert block.current_tier == BlockTier.GPU_FP16
        assert tracker.get_block(1) is block

    def test_register_multiple_blocks(self):
        tracker = self._make_tracker()
        for i in range(10):
            tracker.register_block(i, layer_id=i % 4, head_group=0, request_id="req1")
        assert len(tracker) == 10

    def test_update_attention_basic(self):
        tracker = self._make_tracker()
        tracker.register_block(1, 0, 0, "req1")
        scores = np.array([0.1, 0.5, 0.3, 0.8], dtype=np.float32)
        tracker.update_attention(1, scores)
        block = tracker.get_block(1)
        assert block.max_attention_score == pytest.approx(0.8, abs=1e-4)
        assert 0.0 < block.avg_attention_score <= 0.8

    def test_update_attention_nonfinite_ignored(self):
        tracker = self._make_tracker()
        tracker.register_block(2, 0, 0, "req1")
        scores = np.array([np.nan, np.inf, 0.5], dtype=np.float32)
        tracker.update_attention(2, scores)
        block = tracker.get_block(2)
        assert np.isfinite(block.avg_attention_score)

    def test_mark_accessed(self):
        tracker = self._make_tracker()
        tracker.register_block(1, 0, 0, "req1")
        tracker.mark_accessed(1, step=42)
        block = tracker.get_block(1)
        assert block.last_access_step == 42
        assert block.access_count == 1

    def test_update_tier(self):
        tracker = self._make_tracker()
        tracker.register_block(1, 0, 0, "req1")
        tracker.update_tier(1, BlockTier.GPU_QUANTIZED)
        assert tracker.get_block(1).current_tier == BlockTier.GPU_QUANTIZED

    def test_evict_removes_block(self):
        tracker = self._make_tracker()
        tracker.register_block(1, 0, 0, "req1")
        evicted = tracker.evict_block(1)
        assert evicted is not None
        assert evicted.evicted_flag is True
        assert tracker.get_block(1) is None

    def test_evict_nonexistent_returns_none(self):
        tracker = self._make_tracker()
        assert tracker.evict_block(999) is None

    def test_get_blocks_by_tier(self):
        tracker = self._make_tracker()
        tracker.register_block(1, 0, 0, "req1")
        tracker.register_block(2, 0, 0, "req2")
        tracker.update_tier(2, BlockTier.CPU_OFFLOADED)
        gpu_blocks = tracker.get_blocks_by_tier(BlockTier.GPU_FP16)
        cpu_blocks = tracker.get_blocks_by_tier(BlockTier.CPU_OFFLOADED)
        assert len(gpu_blocks) == 1
        assert len(cpu_blocks) == 1

    def test_memory_stats_structure(self):
        tracker = self._make_tracker()
        tracker.register_block(1, 0, 0, "req1")
        stats = tracker.get_memory_stats()
        assert "total_blocks" in stats
        assert "blocks_per_tier" in stats
        assert "estimated_gpu_mb" in stats

    def test_get_candidate_blocks_lru(self):
        tracker = self._make_tracker()
        for i in range(5):
            tracker.register_block(i, 0, 0, "req1")
            tracker.mark_accessed(i, step=i)
        candidates = tracker.get_candidate_blocks(3, strategy="lru")
        assert len(candidates) == 3
        # Least-recently-used come first
        assert candidates[0].last_access_step <= candidates[1].last_access_step

    def test_release_request_blocks(self):
        tracker = self._make_tracker()
        tracker.register_block(1, 0, 0, "req_a")
        tracker.register_block(2, 0, 0, "req_a")
        tracker.register_block(3, 0, 0, "req_b")
        n = tracker.release_request_blocks("req_a")
        assert n == 2
        assert len(tracker) == 1


# ---------------------------------------------------------------------------
# TelemetryRecord
# ---------------------------------------------------------------------------

class TestTelemetryRecord:
    def _make_block(self):
        b = BlockState(
            block_id=10, layer_id=1, head_group=0, request_id="r1",
            avg_attention_score=0.5, max_attention_score=0.9,
        )
        b.gpu_mem_used_mb = 4000.0
        b.gpu_mem_free_mb = 2000.0
        return b

    def test_from_block_state_creates_record(self):
        block = self._make_block()
        rec = TelemetryRecord.from_block_state(
            block, BlockDecision.KEEP_GPU,
            model_name="test-model",
            workload_type=WorkloadType.SINGLE_DOC_QA,
        )
        assert rec.block_id == 10
        assert rec.decision_label == "keep_gpu"
        assert rec.model_name == "test-model"

    def test_to_dict_has_all_fields(self):
        block = self._make_block()
        rec = TelemetryRecord.from_block_state(
            block, BlockDecision.EVICT, model_name="m"
        )
        d = rec.to_dict()
        required = [
            "request_id", "timestamp", "workload_type", "model_name",
            "block_id", "layer_id", "decision_label", "avg_attention_score",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_flags_computed_from_tier(self):
        block = self._make_block()
        block.current_tier = BlockTier.GPU_QUANTIZED
        assert block.quantized_flag is True
        assert block.offloaded_flag is False
        assert block.evicted_flag is False


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------

class TestTelemetryCollector:
    def _make_collector(self):
        config = {
            "enabled": True,
            "storage_backend": "memory",
            "output_dir": "/tmp/test_telemetry",
            "flush_interval_s": 999,   # manual flush only
            "max_buffer_size": 100,
        }
        return TelemetryCollector(config, model_name="test-model")

    def _make_block(self, bid=1):
        return BlockState(
            block_id=bid, layer_id=0, head_group=0, request_id="r1",
        )

    def test_record_block_event_fills_buffer(self):
        col = self._make_collector()
        col.record_block_event(self._make_block(), BlockDecision.KEEP_GPU)
        stats = col.get_statistics()
        assert stats["buffer_size"] == 1

    def test_get_recent_records(self):
        col = self._make_collector()
        for i in range(5):
            col.record_block_event(self._make_block(i), BlockDecision.KEEP_GPU)
        recent = col.get_recent_records(3)
        assert len(recent) == 3

    def test_flush_clears_buffer(self):
        col = self._make_collector()
        for i in range(5):
            col.record_block_event(self._make_block(i), BlockDecision.EVICT)
        col.flush()
        assert col.get_statistics()["buffer_size"] == 0

    def test_disabled_collector_does_nothing(self):
        config = {"enabled": False, "storage_backend": "memory"}
        col = TelemetryCollector(config)
        col.record_block_event(self._make_block(), BlockDecision.EVICT)
        assert col.get_statistics()["buffer_size"] == 0


# ---------------------------------------------------------------------------
# TelemetryStorage
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTelemetryStorage:
    def _make_record(self):
        block = BlockState(block_id=1, layer_id=0, head_group=0, request_id="r1")
        return TelemetryRecord.from_block_state(
            block, BlockDecision.KEEP_GPU, model_name="m"
        )

    def test_memory_backend_write_read(self):
        storage = TelemetryStorage(backend="memory")
        records = [self._make_record() for _ in range(5)]
        storage.write_batch(records)
        df = storage.read_all()
        assert len(df) == 5

    def test_parquet_backend_write_read(self, tmp_path):
        storage = TelemetryStorage(backend="parquet", output_dir=str(tmp_path))
        records = [self._make_record() for _ in range(3)]
        storage.write_batch(records)
        df = storage.read_all()
        assert len(df) == 3
        assert "block_id" in df.columns

    def test_schema_matches_record(self):
        schema = TelemetryStorage.get_schema()
        field_names = {f.name for f in schema}
        assert "request_id" in field_names
        assert "decision_label" in field_names
