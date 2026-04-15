"""
tests/test_policies.py — Unit tests for cache eviction policies.
"""

import pytest
import numpy as np

from src.telemetry.schema import BlockDecision, BlockState, BlockTier
from src.cache_policies.base import PolicyConfig, PolicyDecision
from src.cache_policies.heuristic import (
    FullCachePolicy,
    SlidingWindowPolicy,
    H2OPolicy,
    PrefixCachingPolicy,
    QuantizeOnlyPolicy,
    OffloadOnlyPolicy,
)
from src.cache_policies.hybrid import HybridAdaptivePolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_block(
    block_id: int,
    age: int = 10,
    last_access: int = 10,
    access_count: int = 1,
    avg_attn: float = 0.5,
    max_attn: float = 0.7,
    prefix_reuse: bool = False,
    tier: BlockTier = BlockTier.GPU_FP16,
) -> BlockState:
    b = BlockState(
        block_id=block_id,
        layer_id=block_id % 4,
        head_group=0,
        request_id="req1",
        block_age_steps=age,
        last_access_step=last_access,
        access_count=access_count,
        avg_attention_score=avg_attn,
        max_attention_score=max_attn,
        prefix_reuse_flag=prefix_reuse,
        current_tier=tier,
        gpu_mem_used_mb=5000.0,
        gpu_mem_free_mb=3000.0,
    )
    return b


def make_blocks(n: int = 20) -> list:
    return [
        make_block(
            block_id=i,
            age=i,
            last_access=i,
            avg_attn=float(i) / n,
            prefix_reuse=(i % 5 == 0),
        )
        for i in range(n)
    ]


def make_system_metrics(pressure: float = 0.70) -> dict:
    total = 8000.0
    used = total * pressure
    free = total - used
    return {
        "gpu_mem_used_mb": used,
        "gpu_mem_free_mb": free,
        "global_step": 100,
        "attn_max": 1.0,
    }


def make_config(**overrides) -> PolicyConfig:
    cfg = PolicyConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# FullCachePolicy
# ---------------------------------------------------------------------------

class TestFullCachePolicy:
    def test_low_pressure_keeps_all(self):
        policy = FullCachePolicy(make_config())
        blocks = make_blocks(10)
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.40))
        assert all(d.decision == BlockDecision.KEEP_GPU for d in decisions)

    def test_high_pressure_evicts_some(self):
        policy = FullCachePolicy(make_config())
        blocks = make_blocks(20)
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.90))
        evicted = [d for d in decisions if d.decision == BlockDecision.EVICT]
        assert len(evicted) > 0

    def test_high_pressure_evicts_lru_first(self):
        policy = FullCachePolicy(make_config())
        blocks = [make_block(i, last_access=i) for i in range(10)]
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.90))
        # Block 0 has the lowest last_access_step → should be among evicted
        d0 = next(d for d in decisions if d.block_id == 0)
        assert d0.decision == BlockDecision.EVICT


# ---------------------------------------------------------------------------
# SlidingWindowPolicy
# ---------------------------------------------------------------------------

class TestSlidingWindowPolicy:
    def test_low_pressure_keeps_all(self):
        policy = SlidingWindowPolicy(make_config(sliding_window_blocks=5, sink_token_blocks=2))
        blocks = make_blocks(8)
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.40))
        assert all(d.decision == BlockDecision.KEEP_GPU for d in decisions)

    def test_medium_pressure_evicts_outside_window(self):
        policy = SlidingWindowPolicy(make_config(
            sliding_window_blocks=5, sink_token_blocks=2,
            memory_pressure_low=0.60,
        ))
        blocks = [make_block(i) for i in range(20)]
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.70))
        evicted = [d for d in decisions if d.decision == BlockDecision.EVICT]
        # 20 blocks − 5 window − 2 sinks = at least 13 evicted
        assert len(evicted) >= 10

    def test_sink_tokens_always_kept(self):
        policy = SlidingWindowPolicy(make_config(
            sliding_window_blocks=3, sink_token_blocks=2,
            memory_pressure_low=0.60,
        ))
        blocks = [make_block(i) for i in range(10)]
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.75))
        # Blocks with lowest block_id (sinks) should be kept
        for bid in range(2):
            d = next(d for d in decisions if d.block_id == bid)
            assert d.decision == BlockDecision.KEEP_GPU


# ---------------------------------------------------------------------------
# H2OPolicy
# ---------------------------------------------------------------------------

class TestH2OPolicy:
    def test_heavy_hitters_kept(self):
        policy = H2OPolicy(make_config(heavy_hitter_fraction=0.20, sink_token_blocks=2))
        # Block with highest attention should be kept
        blocks = [make_block(i, avg_attn=float(i) / 10.0) for i in range(10)]
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.75))
        # Block 9 has the highest attention → should be KEEP_GPU
        d9 = next(d for d in decisions if d.block_id == 9)
        assert d9.decision == BlockDecision.KEEP_GPU

    def test_low_attention_blocks_evicted(self):
        policy = H2OPolicy(make_config(heavy_hitter_fraction=0.20, sink_token_blocks=0))
        blocks = [make_block(i, avg_attn=0.01 * i) for i in range(10)]
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.75))
        d0 = next(d for d in decisions if d.block_id == 0)
        assert d0.decision == BlockDecision.EVICT


# ---------------------------------------------------------------------------
# HybridAdaptivePolicy
# ---------------------------------------------------------------------------

class TestHybridAdaptivePolicy:
    def test_low_pressure_keeps_all_blocks(self):
        policy = HybridAdaptivePolicy(make_config(memory_pressure_low=0.65))
        blocks = make_blocks(10)
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.40))
        assert all(d.decision == BlockDecision.KEEP_GPU for d in decisions)

    def test_medium_pressure_uses_all_actions(self):
        policy = HybridAdaptivePolicy(make_config(
            memory_pressure_low=0.50,
            memory_pressure_high=0.80,
            quantization_enabled=True,
            offload_enabled=True,
        ))
        blocks = [make_block(i, avg_attn=float(i) / 20.0) for i in range(20)]
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.65))
        action_set = {d.decision for d in decisions}
        # Should see at least KEEP_GPU and one other action
        assert BlockDecision.KEEP_GPU in action_set
        assert len(action_set) > 1

    def test_high_pressure_aggressive_eviction(self):
        policy = HybridAdaptivePolicy(make_config(
            memory_pressure_high=0.80,
            heavy_hitter_fraction=0.10,
        ))
        blocks = [make_block(i, avg_attn=0.01) for i in range(20)]
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.92))
        evicted = [d for d in decisions if d.decision == BlockDecision.EVICT]
        # Under extreme pressure, most low-attention blocks should be evicted
        assert len(evicted) >= len(blocks) // 2

    def test_decision_count_equals_block_count(self):
        policy = HybridAdaptivePolicy(make_config())
        blocks = make_blocks(15)
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.70))
        assert len(decisions) == len(blocks)

    def test_memory_pressure_computation(self):
        policy = HybridAdaptivePolicy(make_config())
        p = policy.compute_memory_pressure(6000.0, 2000.0)
        assert p == pytest.approx(0.75, abs=1e-4)

    def test_policy_report_returns_dict(self):
        policy = HybridAdaptivePolicy(make_config())
        blocks = make_blocks(5)
        policy.decide(blocks, make_system_metrics(0.70))
        report = policy.get_policy_report()
        assert "policy_name" in report
        assert "eval_count" in report


# ---------------------------------------------------------------------------
# QuantizeOnlyPolicy
# ---------------------------------------------------------------------------

class TestQuantizeOnlyPolicy:
    def test_never_evicts(self):
        policy = QuantizeOnlyPolicy(make_config(
            quantization_enabled=True,
            memory_pressure_low=0.50,
        ))
        blocks = make_blocks(10)
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.75))
        assert all(d.decision != BlockDecision.EVICT for d in decisions)

    def test_low_attention_blocks_quantized(self):
        policy = QuantizeOnlyPolicy(make_config(
            quantization_enabled=True,
            quantization_min_importance=0.5,
            memory_pressure_low=0.50,
        ))
        blocks = [make_block(i, avg_attn=0.1) for i in range(5, 10)]  # no sinks
        decisions = policy.decide(blocks, make_system_metrics(pressure=0.75))
        quantized = [d for d in decisions if d.decision == BlockDecision.QUANTIZE]
        assert len(quantized) > 0


# ---------------------------------------------------------------------------
# Parametrised: all heuristic policies produce correct number of decisions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy_cls", [
    FullCachePolicy,
    SlidingWindowPolicy,
    H2OPolicy,
    PrefixCachingPolicy,
    QuantizeOnlyPolicy,
    OffloadOnlyPolicy,
    HybridAdaptivePolicy,
])
def test_decision_count_matches_block_count(policy_cls):
    policy = policy_cls(make_config())
    blocks = make_blocks(12)
    decisions = policy.decide(blocks, make_system_metrics(0.70))
    assert len(decisions) == len(blocks)


@pytest.mark.parametrize("pressure", [0.30, 0.55, 0.70, 0.88, 0.95])
def test_hybrid_pressure_levels(pressure):
    policy = HybridAdaptivePolicy(make_config(
        memory_pressure_low=0.60,
        memory_pressure_high=0.85,
    ))
    blocks = make_blocks(10)
    decisions = policy.decide(blocks, make_system_metrics(pressure))
    assert len(decisions) == 10
    for d in decisions:
        assert d.decision in list(BlockDecision)
