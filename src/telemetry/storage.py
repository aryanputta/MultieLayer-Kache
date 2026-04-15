"""
storage.py — Persistence backends for telemetry records.

Supported backends
------------------
- ``"parquet"``   Write Parquet files partitioned by date under *output_dir*.
- ``"memory"``    Keep everything in RAM (useful for unit tests).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.telemetry.schema import TelemetryRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyArrow schema (mirrors TelemetryRecord field order)
# ---------------------------------------------------------------------------

TELEMETRY_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string()),
        pa.field("timestamp", pa.float64()),
        pa.field("workload_type", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("prompt_tokens", pa.int32()),
        pa.field("generated_tokens", pa.int32()),
        pa.field("context_length", pa.int32()),
        pa.field("queue_depth", pa.int32()),
        pa.field("gpu_mem_used_mb", pa.float32()),
        pa.field("gpu_mem_free_mb", pa.float32()),
        pa.field("cpu_mem_used_mb", pa.float32()),
        pa.field("block_id", pa.int32()),
        pa.field("layer_id", pa.int16()),
        pa.field("head_group", pa.int16()),
        pa.field("block_age_steps", pa.int32()),
        pa.field("last_access_step", pa.int32()),
        pa.field("access_count", pa.int32()),
        pa.field("avg_attention_score", pa.float32()),
        pa.field("max_attention_score", pa.float32()),
        pa.field("prefix_reuse_flag", pa.bool_()),
        pa.field("quantized_flag", pa.bool_()),
        pa.field("offloaded_flag", pa.bool_()),
        pa.field("evicted_flag", pa.bool_()),
        pa.field("decision_label", pa.string()),
        pa.field("latency_ms", pa.float32()),
        pa.field("ttft_ms", pa.float32()),
        pa.field("tpot_ms", pa.float32()),
        pa.field("task_score", pa.float32()),
    ]
)


class TelemetryStorage:
    """Writes and reads telemetry records.

    Parameters
    ----------
    backend:
        ``"parquet"`` or ``"memory"``.
    output_dir:
        Root directory for Parquet output (ignored for memory backend).
    """

    def __init__(self, backend: str = "parquet", output_dir: str = "data/traces") -> None:
        if backend not in ("parquet", "memory"):
            raise ValueError(f"Unknown telemetry backend: '{backend}'")
        self.backend = backend
        self.output_dir = output_dir

        # In-memory store
        self._records: List[dict] = []

        if backend == "parquet":
            os.makedirs(output_dir, exist_ok=True)
            logger.debug("TelemetryStorage (parquet) → %s", output_dir)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_batch(self, records: List[TelemetryRecord]) -> None:
        """Persist a batch of records.

        For the Parquet backend records are appended to a file named by the
        current UTC date so that each day's traces live in one file.

        Parameters
        ----------
        records:
            Non-empty list of :class:`TelemetryRecord` instances.
        """
        if not records:
            return

        dicts = [r.to_dict() for r in records]

        if self.backend == "memory":
            self._records.extend(dicts)
            return

        # Parquet
        table = pa.Table.from_pylist(dicts, schema=TELEMETRY_SCHEMA)
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        path = os.path.join(self.output_dir, f"traces_{date_str}.parquet")

        if os.path.exists(path):
            existing = pq.read_table(path, schema=TELEMETRY_SCHEMA)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression="snappy")
        logger.debug("Wrote %d records to %s", len(records), path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_all(self) -> pd.DataFrame:
        """Read all stored records into a DataFrame."""
        if self.backend == "memory":
            return pd.DataFrame(self._records) if self._records else pd.DataFrame()

        parquet_files = sorted(
            f for f in os.listdir(self.output_dir) if f.endswith(".parquet")
        )
        if not parquet_files:
            return pd.DataFrame()

        tables = [
            pq.read_table(os.path.join(self.output_dir, f)) for f in parquet_files
        ]
        return pa.concat_tables(tables).to_pandas()

    def read_recent(self, hours: float = 1.0) -> pd.DataFrame:
        """Read records from the last *hours* hours."""
        df = self.read_all()
        if df.empty:
            return df
        import time
        cutoff = time.time() - hours * 3600
        return df[df["timestamp"] >= cutoff].reset_index(drop=True)

    @staticmethod
    def get_schema() -> pa.Schema:
        """Return the canonical PyArrow schema for telemetry records."""
        return TELEMETRY_SCHEMA
