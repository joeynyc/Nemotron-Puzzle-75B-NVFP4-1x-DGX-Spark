# Results — Nemotron Puzzle 75B-A9B NVFP4 · 1× DGX Spark (GB10)

**Validated 2026-07-08** on one DGX Spark (`spark-db08`), NGC `vllm:26.06-py3`,
native MTP `num_speculative_tokens=3`, thinking disabled for probes.

Protocol unless noted:

- Warm engine (not first request after boot)
- `temperature=0` for single-stream; demo 4-stream used `0.35`
- Wall tok/s = `completion_tokens / wall_clock` for non-stream requests
- Aggregate = sum of completion tokens / span from first start to last end

## Before (stock-ish serve)

```text
gpu_memory_utilization=0.85
max_model_len=262144
max_num_seqs=1
MTP×3
prefix caching: off
max_num_batched_tokens: default (vLLM warn → scheduled tokens ~2048 under MTP)
```

### Single stream

| workload | prompt tok | completion | wall s | tok/s |
| --- | ---: | ---: | ---: | ---: |
| structured_count | 31 | 140 | 3.35 | **41.8** |
| code_snippet | 33 | 179 | 4.39 | **40.7** |
| prose | 34 | 160 | 4.96 | **32.3** |
| **mean** | | | | **38.3** |

Stream structured: **40.0** tok/s, TTFT ~0.22 s.

### Four parallel streams

**Not possible without queueing.** `max_num_seqs=1` → capacity wait; aggregate ≈ single-stream (~40 tok/s) while others sit in `reason=capacity`.

### Spec decode (lifetime counters, pre-reload)

| metric | value |
| --- | ---: |
| draft tokens accepted / drafted | ~74.7% |
| mean accept length (engine logs) | ~2.5–3.2 |
| pos0 / pos1 / pos2 accept (approx) | 0.87 / 0.74 / 0.63 |

## After (this recipe)

```text
gpu_memory_utilization=0.88
max_model_len=262144
max_num_seqs=4
MTP×3
prefix caching: on
max_num_batched_tokens=8192
```

### Single stream

| workload | prompt tok | completion | wall s | tok/s |
| --- | ---: | ---: | ---: | ---: |
| structured_count | 31 | 140 | 3.92 | **35.7** |
| code_snippet | 33 | 169 | 4.61 | **36.6** |
| prose | 34 | 160 | 4.52 | **35.4** |
| **mean** | | | | **35.9** |

Stream structured: **39.5** tok/s, TTFT ~0.25 s.

Solo decode did **not** improve; slight structured/code regression, prose flat/up. Expectation: concurrency flags are not free solo speedups.

### Four parallel streams

Earlier concurrency probe (100 tokens × 4, structured):

| metric | value |
| --- | ---: |
| per-stream tok/s | ~17.9–20.9 |
| aggregate | **~71.7** tok/s |

Demo recording (`demo/record_and_render.py`, 220 max tokens × 4, thinking off):

| metric | value |
| --- | ---: |
| total completion tokens | **834** |
| span s | **11.08** |
| **aggregate tok/s** | **75.3** |
| TTFT | ~0.45 s (all four) |

### Prefix cache under agent load

After serving Hermes traffic:

```text
prefix_cache_hits / queries ≈ 32000 / 45453 ≈ 70.4%
```

## Boot evidence (after)

```text
Model loading took ~50 GiB
Available KV cache memory: ~54.7 GiB
GPU KV cache size: ~2,000,731 tokens
enable_prefix_caching=True
max_num_seqs=4
max_num_batched_tokens=8192
```

## Reproduce

```bash
./scripts/start-puzzle.sh
# wait until /v1/models is healthy
./scripts/bench-single.py
./scripts/bench-4stream.py
```

Report your own hardware (driver, image tag, GMU) — unified-memory boot state moves KV pool size by hundreds of MiB.
