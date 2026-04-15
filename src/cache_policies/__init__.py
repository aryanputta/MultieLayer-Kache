"""
src/cache_policies — KV-cache eviction and tier-management policies.

All policies share the :class:`BasePolicy` interface and return
:class:`PolicyDecision` lists from their ``decide()`` method.

Available policies
------------------
- FullCachePolicy       — keep everything, evict LRU under OOM
- SlidingWindowPolicy   — StreamingLLM-inspired window + sink tokens
- H2OPolicy             — Heavy-Hitter Oracle (cumulative attention)
- PrefixCachingPolicy   — prioritise prefix-reuse blocks
- QuantizeOnlyPolicy    — never evict; quantise low-attention blocks
- OffloadOnlyPolicy     — never evict; offload medium blocks to CPU
- HybridAdaptivePolicy  — workload-aware, pressure-sensitive controller
- LearnedPolicy         — XGBoost/LightGBM ranker override
- PolicyEngine          — async scheduling loop
"""

from src.cache_policies.base import BasePolicy, PolicyConfig, PolicyDecision, PolicyStats
from src.cache_policies.heuristic import (
    FullCachePolicy,
    SlidingWindowPolicy,
    H2OPolicy,
    PrefixCachingPolicy,
    QuantizeOnlyPolicy,
    OffloadOnlyPolicy,
)
from src.cache_policies.hybrid import HybridAdaptivePolicy
from src.cache_policies.learned import LearnedPolicy
from src.cache_policies.policy_engine import PolicyEngine

__all__ = [
    "BasePolicy",
    "PolicyConfig",
    "PolicyDecision",
    "PolicyStats",
    "FullCachePolicy",
    "SlidingWindowPolicy",
    "H2OPolicy",
    "PrefixCachingPolicy",
    "QuantizeOnlyPolicy",
    "OffloadOnlyPolicy",
    "HybridAdaptivePolicy",
    "LearnedPolicy",
    "PolicyEngine",
]
