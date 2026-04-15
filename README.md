# Adaptive Multi-Tier KV Cache Orchestrator for Long-Context LLM Inference

A research-grade ML systems project that improves long-context LLM inference by building a **workload-aware KV cache controller** on top of vLLM. The system decides in real time whether KV blocks should stay on GPU, be quantized, be offloaded to CPU memory, or be evicted — adapting its policy to workload type, memory pressure, and service-level targets.

---

## Research Question

> Can a workload-aware hybrid KV cache controller outperform single-strategy KV cache methods on the memory–latency–accuracy tradeoff for long-context inference?

**Core hypothesis:** Static cache policies are suboptimal across heterogeneous workloads. A controller that combines attention statistics, recency, reuse likelihood, workload class, and memory pressure can achieve better task quality per GB of memory and better p95 latency under load than fixed policies.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Server  (src/serving/api_server.py)                    │
│  POST /v1/generate  •  GET /v1/health  •  GET /v1/metrics       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  InstrumentedVLLMEngine  (src/serving/vllm_engine.py)           │
│  Wraps AsyncLLMEngine  •  measures TTFT/TPOT  •  feeds tracker  │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
┌──────────▼────────────┐       ┌─────────────▼───────────────────┐
│  BlockTracker         │       │  TelemetryCollector              │
│  Thread-safe registry │       │  Buffers & flushes to Parquet    │
│  of live KV blocks    │       │  (src/telemetry/)                │
└──────────┬────────────┘       └─────────────────────────────────┘
           │
┌──────────▼────────────────────────────────────────────────────┐
│  PolicyEngine  (src/cache_policies/policy_engine.py)          │
│  Async loop • runs every 50 ms • applies decisions            │
│                                                               │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ HybridAdapt  │  │  H2O     │  │ Sliding  │  │ Learned  │ │
│  │ (main)       │  │ baseline │  │ Window   │  │ XGBoost  │ │
│  └──────────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                                                               │
│  Decisions:  KEEP_GPU  •  QUANTIZE  •  OFFLOAD_CPU  •  EVICT  │
└───────────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│  Feature Engineering  (src/feature_engineering/)                │
│  21-dim vector: age, recency, attention rank, reuse flag,       │
│  GPU pressure, workload type, layer depth, HH flag, …           │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Tiers

| Tier | Storage | Precision | Latency |
|------|---------|-----------|---------|
| `GPU_FP16` | HBM (on-device) | FP16 | ~0 μs |
| `GPU_QUANTIZED` | HBM (on-device) | INT8 / FP8 | ~0 μs + decode |
| `CPU_OFFLOADED` | DRAM (host) | FP16 | PCIe round-trip |
| `EVICTED` | None | — | Full recompute |

---

## Quick Start

### Prerequisites
- Python ≥ 3.10
- CUDA 12.1 + compatible GPU (optional; falls back to CPU mock for testing)
- Docker + Docker Compose (for full stack)

### Installation

```bash
git clone https://github.com/aryanputta/multielayer-kache
cd MultieLayer-Kache

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install (runtime deps)
pip install -r requirements.txt
pip install -e .

# Or with dev tools
make install-dev
```

### Run the server

```bash
# Start API server (mock engine if vLLM not installed)
make serve
# → http://localhost:8000

# With Docker (full stack: API + Prometheus + Grafana + Redis)
make docker-up
# → API:      http://localhost:8000
# → Grafana:  http://localhost:3000  (admin / admin)
# → Prometheus: http://localhost:9090
```

### Run tests

```bash
make test            # full suite
make test-fast       # skip @slow markers
make test-cov        # with HTML coverage report
```

### Run a benchmark

```bash
# LongBench evaluation (requires running API server)
make benchmark

# Or with custom options
python scripts/run_benchmark.py \
  --policy hybrid \
  --memory-budget 0.75 \
  --tasks narrativeqa hotpotqa gov_report lcc \
  --concurrency 8
```

### Train the importance model

```bash
# Collect traces first (requires running server + benchmark)
# Then train:
python scripts/train_policy.py \
  --model-type xgboost \
  --strategy combined \
  --n-trials 50
```

---

## Policies

| Policy | Description | Primary Baseline For |
|--------|-------------|----------------------|
| `full_cache` | Keep all blocks; LRU evict under OOM | System baseline |
| `sliding_window` | StreamingLLM-style window + sink tokens | Recency-only ablation |
| `h2o` | Heavy-Hitter Oracle (top-K by attention) | Attention-only ablation |
| `prefix_caching` | Prioritise reuse-flag blocks | Prefix-reuse ablation |
| `quantize_only` | Quantise low-attention blocks; no eviction | Quantisation-only ablation |
| `offload_only` | Offload medium blocks to CPU; no eviction | Offload-only ablation |
| `hybrid` | Workload-aware, pressure-sensitive 4-action controller | **Main contribution** |
| `learned` | XGBoost / LightGBM importance ranker | Learned-policy experiment |

### Hybrid Decision Logic

```
pressure < 0.65  → keep_gpu for all blocks
pressure ∈ [0.65, 0.85)
    score = α·recency + β·attention + γ·reuse   (workload-adjusted)
    score ≥ 0.30  → keep_gpu
    score ≥ 0.15  → quantize
    score ≥ 0.075 → offload_cpu
    else          → evict
pressure ≥ 0.85  → aggressive: keep only heavy hitters + sinks,
                   quantize medium-high, offload medium, evict rest
```

---

## Benchmarks

### LongBench Tasks (all 16)

| Task | Workload | Metric |
|------|----------|--------|
| narrativeqa, qasper, triviaqa | Single-doc QA | F1 / ROUGE-L |
| hotpotqa, 2wikimqa, musique, multifieldqa_en | Multi-doc QA | F1 |
| gov_report, qmsum, multi_news | Summarisation | ROUGE-L |
| samsum | Dialogue | ROUGE-L |
| trec, passage_count, passage_retrieval_en | Structured | EM |
| lcc, repobench-p | Code | Edit Sim |

### Stress Tests

- **Bursty arrivals** — Poisson arrival rate sweep
- **Repeated prefix reuse** — prefix caching effectiveness
- **Long chat sessions** — multi-turn dialogue memory pressure
- **Mixed context batch** — heterogeneous prompt lengths
- **Memory spike** — rapid concurrency ramp-up

### Memory Budget Sweeps

| Budget | GPU Utilisation Target |
|--------|----------------------|
| Generous | 0.90 |
| Medium | 0.75 |
| Constrained | 0.60 |
| Severe | 0.45 |

---

## Telemetry Schema

Every policy decision emits one `TelemetryRecord` row:

```
request_id  timestamp  workload_type  model_name  prompt_tokens  generated_tokens
context_length  queue_depth  gpu_mem_used_mb  gpu_mem_free_mb  cpu_mem_used_mb
block_id  layer_id  head_group  block_age_steps  last_access_step  access_count
avg_attention_score  max_attention_score  prefix_reuse_flag
quantized_flag  offloaded_flag  evicted_flag  decision_label
latency_ms  ttft_ms  tpot_ms  task_score
```

Stored as date-partitioned Parquet files in `data/traces/`.

---

## Feature Vector (21 dims)

```
 0  block_age_normalized          11  queue_depth_normalized
 1  recency_score                 12  context_length_normalized
 2  access_frequency              13  output_length_normalized
 3  avg_attention_score           14  workload_type_encoded
 4  max_attention_score           15  is_heavy_hitter
 5  attention_rank_normalized     16  is_recent_window
 6  sink_token_flag               17  access_frequency_log
 7  prefix_reuse_flag             18  cumulative_attention_mass_rank
 8  layer_depth_normalized        19  tier_encoded
 9  head_group_normalized         20  inter_access_interval_normalized
10  gpu_mem_pressure
```

---

## Evaluation Metrics

**Quality:** F1, EM, ROUGE-L, Edit Similarity  
**System:** throughput (TPS / RPS), p50/p95/p99 latency, TTFT, TPOT, GPU memory peak/avg, blocks per tier  
**Composite:** Pareto frontier (quality vs. memory), quality-per-GB, p95-latency vs. memory

**Statistical tests:** paired t-test, Wilcoxon signed-rank, bootstrap 95% CI, Cohen's d effect size, one-way ANOVA

---

## Project Structure

```
MultieLayer-Kache/
├── configs/                  # Hydra YAML configs
│   ├── default.yaml
│   ├── policies/             # heuristic / learned / hybrid
│   ├── benchmarks/           # longbench / stress
│   └── models/               # llama3_8b
├── data/
│   ├── traces/               # Parquet telemetry (gitignored)
│   └── benchmark_inputs/
├── docker/                   # Dockerfile, docker-compose, Prometheus, Grafana
├── scripts/
│   ├── run_benchmark.py      # LongBench CLI
│   ├── train_policy.py       # Importance model trainer CLI
│   └── run_experiment_suite.py
├── src/
│   ├── serving/              # InstrumentedVLLMEngine + FastAPI
│   ├── telemetry/            # Schema, BlockTracker, Collector, Storage
│   ├── cache_policies/       # All policies + PolicyEngine
│   ├── feature_engineering/  # FeatureExtractor, ImportanceScorer
│   ├── trainers/             # ImportanceModelTrainer, LabelGenerator
│   ├── evaluators/           # QualityEvaluator, SystemMetrics, Pareto
│   ├── benchmarks/           # LongBenchRunner, StressTestRunner, Harness
│   └── visualization/        # ResultPlotter, LiveDashboard
├── tests/                    # pytest suite (4 modules, ~80 tests)
├── results/                  # tables/, plots/, logs/ (gitignored)
├── Makefile
├── requirements.txt
└── setup.py
```

---

## Core Experiments

| # | Name | Policies | Key Question |
|---|------|----------|-------------|
| 1 | System baselines | full_cache, sliding_window, prefix_caching | Where do naive methods break? |
| 2 | Heuristic hybrid | h2o, quantize_only, offload_only, hybrid | Does combining actions help? |
| 3 | Learned ranker | hybrid, learned | Does ML beat heuristics? |
| 4 | Workload generalisation | h2o, hybrid | Does workload-awareness matter? |
| 5 | Cross-model robustness | hybrid on Llama + Mistral | Does it transfer? |
| 6 | Quantisation ablation | hybrid w/ and w/o quant | What does quantisation contribute? |
| 7 | Offload ablation | hybrid w/ and w/o offload | What does offloading contribute? |
| 8 | Stress test | full_cache, hybrid | Where does each policy fail under load? |
| 9 | Context-length degradation | all | How does quality fall off with length? |
| 10 | Feature importance | learned | Which signals matter most? |

---

## Expected Findings

- Hybrid controller preserves task quality better than static window methods under tight budgets
- Quantise-only strategy helps memory but is weaker than adaptive retain + quantize + offload
- Workload-aware policies matter most on retrieval-heavy and multi-document tasks
- Offloading preserves quality but can hurt tail latency unless triggered carefully
- Learned ranking beats hand-tuned heuristics at intermediate memory pressure

---

## Reproducing Results

```bash
# 1. Start the server
make serve

# 2. Run warmup requests
python -c "
import httpx, asyncio
async def warmup():
    async with httpx.AsyncClient() as c:
        for _ in range(10):
            await c.post('http://localhost:8000/v1/generate',
                         json={'prompt': 'Hello', 'max_tokens': 8})
asyncio.run(warmup())
"

# 3. Run full experiment suite
python scripts/run_experiment_suite.py --experiments 1 2 3 4

# 4. Train importance model on collected traces
python scripts/train_policy.py --strategy combined --n-trials 30

# 5. Re-run experiment 3 with trained model
python scripts/run_benchmark.py --policy learned
```

---

## Hardware Record

To be filled in before paper submission:
- GPU: `<model>` — `<VRAM>` GB VRAM
- CUDA version: `<version>`
- Driver version: `<version>`
- vLLM version: `<version>`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Host RAM: `<GB>`

---

## Failure Modes to Analyse

- Policy thrashing under rapid workload changes
- Offload overhead dominating gains at high PCIe saturation
- Quantisation hurting accuracy on reasoning-heavy tasks
- Weak pseudo-labels for importance model in sparse-attention regimes
- Telemetry overhead affecting measured throughput
- Poor transfer across model architectures with different attention patterns

---

## Citation / References

Key papers this project builds on:

- Kwon et al. (2023). *Efficient Memory Management for LLM Serving with PagedAttention.* (vLLM)
- Zhang et al. (2023). *H2O: Heavy-Hitter Oracle for LLM KV Cache.* NeurIPS 2023.
- Xiao et al. (2023). *StreamingLLM: Efficient Streaming Language Models with Attention Sinks.*
- Li et al. (2024). *SnapKV: LLM Knows What You are Looking for Before Generation.*
- Bai et al. (2023). *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.*
- NVIDIA (2024). *TensorRT-LLM KV Cache Reuse.*

---

## License

Apache 2.0 — see `LICENSE`.
