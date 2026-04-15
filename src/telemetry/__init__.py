"""
src/telemetry — KV-cache telemetry collection and storage.

Public API
----------
- schema:        WorkloadType, BlockTier, BlockDecision, BlockState,
                 TelemetryRecord, SystemMetrics, RequestMetrics
- BlockTracker:  thread-safe registry of live KV-cache blocks
- TelemetryCollector: buffers and flushes telemetry records
- TelemetryStorage:   parquet / in-memory backends
"""

from src.telemetry.schema import (
    BlockDecision,
    BlockState,
    BlockTier,
    RequestMetrics,
    SystemMetrics,
    TelemetryRecord,
    WorkloadType,
)
from src.telemetry.block_tracker import BlockTracker
from src.telemetry.collector import TelemetryCollector
from src.telemetry.storage import TelemetryStorage

__all__ = [
    "WorkloadType",
    "BlockTier",
    "BlockDecision",
    "BlockState",
    "TelemetryRecord",
    "SystemMetrics",
    "RequestMetrics",
    "BlockTracker",
    "TelemetryCollector",
    "TelemetryStorage",
]
