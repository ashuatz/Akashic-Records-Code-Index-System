# Benchmark Quick Start Guide

Get started with benchmarking in 5 minutes.

## Prerequisites

1. Indexed codebase (run `python scripts\ingest.py` first)
2. Qdrant running (`docker run -p 6333:6333 qdrant/qdrant`)
3. Embedding server running (llama.cpp on port 8081)
4. Python dependencies installed

## Quick Test (2 minutes)

Run a fast validation with 2 configurations:

```bash
cd <repo-root>

# Quick test: with/without reranker
python scripts/run_benchmark_suite.py --suite quick_test
```

Output:
```
Configuration 1/2
Running benchmark: baseline_1024
  Processing 30 queries... ████████████ 100% 0:00:45

Configuration 2/2
Running benchmark: with_reranker_1024
  Processing 30 queries... ████████████ 100% 0:01:12

╭─────────────────────────────────────────────╮
│         Benchmark Comparison                │
├──────────────────┬──────┬─────────┬─────────┤
│ Configuration    │ MRR  │ NDCG@10 │ Latency │
├──────────────────┼──────┼─────────┼─────────┤
│ baseline_1024    │ 0.68 │ 0.78    │ 95ms    │
│ with_reranker_1024│0.73 │ 0.84    │ 145ms   │
└──────────────────┴──────┴─────────┴─────────┘

Recommendations:
  Best overall quality: with_reranker_1024 (NDCG@10: 0.84)
```

## Standard Benchmark (5-10 minutes)

Compare different chunk sizes:

```bash
# Test 4 chunk sizes (512, 1024, 2048, 4096)
python scripts\run_benchmark_suite.py --suite chunk_comparison
```

This runs **8 configurations** (4 chunk sizes × 2 reranker states) and shows which chunk size works best for your codebase.

## View Results

```bash
# Visualize latest results
python scripts\visualize_benchmark.py benchmark/results/comparison_latest.json

# Show category breakdown
python scripts\visualize_benchmark.py benchmark/results/comparison_latest.json --show-categories

# Save summary to file
python scripts\visualize_benchmark.py benchmark/results/comparison_latest.json --output summary.txt
```

## Available Suites

List all available benchmark suites:

```bash
python scripts\run_benchmark_suite.py --list-suites
```

Predefined suites:
- **quick_test**: 2 configs, ~2 minutes
- **chunk_comparison**: 4 chunk sizes, ~5 minutes
- **embedding_comparison**: 4 embedding models, ~10 minutes
- **reranker_study**: Reranker ablation, ~5 minutes
- **comprehensive**: Full test (12 configs), ~30 minutes
- **speed_quality**: Trade-off analysis, ~10 minutes

## Manual Benchmark

Run custom configuration:

```bash
# Test current settings
python scripts\benchmark.py --single

# Custom chunk sizes
python scripts\benchmark.py --chunk-sizes 1024,2048,4096

# Custom query file
python scripts\benchmark.py --queries my_queries.yaml
```

## Understanding Metrics

### MRR (Mean Reciprocal Rank)
- **0.8+**: First relevant result typically in top 2 ✓
- **0.6-0.8**: First relevant result typically in top 3-4
- **<0.6**: Relevant results not appearing early ✗

### Recall@10
- **0.9+**: Almost all relevant results in top 10 ✓
- **0.7-0.9**: Most relevant results in top 10
- **<0.7**: Many relevant results missed ✗

### NDCG@10 (Ranking Quality)
- **0.9+**: Near-perfect ranking ✓✓
- **0.8-0.9**: Strong ranking quality ✓
- **0.6-0.8**: Acceptable ranking
- **<0.6**: Weak ranking quality ✗

### Latency
- **<100ms**: Excellent response time ✓✓
- **100-200ms**: Good response time ✓
- **200-500ms**: Acceptable for complex queries
- **>500ms**: May need optimization ✗

## Typical Results

For a well-configured system:
- **MRR**: 0.70-0.80
- **Recall@10**: 0.80-0.90
- **NDCG@10**: 0.80-0.90
- **Latency**: 100-200ms

If your scores are significantly lower:
1. Check if test queries match your indexed codebase
2. Verify embedding server is using correct model
3. Try enabling reranker
4. Experiment with different chunk sizes

## Common Issues

### "No results for queries"
```bash
# Solution: Ensure codebase is indexed
python scripts\init_collection.py
python scripts\ingest.py /path/to/codebase
```

### "Low MRR/NDCG scores"
```bash
# Solution 1: Enable reranker
# Edit config/settings.yaml:
reranker:
  enabled: true

# Solution 2: Try different chunk size
python scripts\benchmark.py --chunk-sizes 2048
```

### "Slow performance"
```bash
# Solution 1: Disable reranker for faster search
# Edit config/settings.yaml:
reranker:
  enabled: false

# Solution 2: Use smaller chunk sizes
# Edit config/settings.yaml:
indexing:
  chunk:
    max_tokens: 1024
```

## Next Steps

1. Run `quick_test` to validate setup
2. Run `chunk_comparison` to find optimal chunk size
3. If quality is low, try `reranker_study` to see impact
4. For production, use `speed_quality` to balance performance

## Example Workflow

```bash
# 1. Quick validation
python scripts\run_benchmark_suite.py --suite quick_test

# 2. Find optimal chunk size
python scripts\run_benchmark_suite.py --suite chunk_comparison

# 3. View results
python scripts\visualize_benchmark.py benchmark/results/comparison_latest.json

# 4. Update settings based on best configuration
# Edit config/settings.yaml with winning chunk_size

# 5. Re-index with new settings
python scripts\ingest.py /path/to/codebase

# 6. Final validation
python scripts\benchmark.py --single
```

## Advanced Usage

### Custom Queries

Create `my_queries.yaml`:

```yaml
queries:
  - id: my_test
    query: "my specific search query"
    category: function_search
    expected_files:
      - "path/to/expected/file.cpp"
    expected_symbols:
      - "ExpectedFunctionName"
```

Run benchmark:

```bash
python scripts\benchmark.py --queries my_queries.yaml --single
```

### CI/CD Integration

Add to your CI pipeline:

```bash
# Fail if quality drops below threshold
python scripts\benchmark.py --single --min-ndcg 0.75
if [ $? -ne 0 ]; then
  echo "Benchmark quality regression detected!"
  exit 1
fi
```

### Monitoring

Track metrics over time:

```bash
# Weekly benchmark
cron: 0 2 * * 0 python scripts\run_benchmark_suite.py --suite quick_test

# Compare with previous week
python scripts\compare_reports.py \
  benchmark/results/week1.json \
  benchmark/results/week2.json
```

## Support

For issues or questions:
1. Check `benchmark/README.md` for detailed documentation
2. Review example configurations in `benchmark/example_configs.yaml`
3. Run with `--help` for command options
