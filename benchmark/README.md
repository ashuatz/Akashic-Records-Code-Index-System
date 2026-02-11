# Akashic Records - Benchmark Harness

Relevance evaluation harness for comparing embedding models, chunk sizes, and reranker configurations.

## Overview

This benchmark system evaluates the quality of search results across different configurations:

- **Chunk sizes**: 512, 1024, 2048, 4096, 8000 tokens
- **Embedding models**: nomic-embed-code, BGE, others
- **Reranker**: With/without BGE-reranker-base
- **Query types**: Function search, concept search, error search, pattern search, API search

## Metrics

### Mean Reciprocal Rank (MRR)
Measures the rank of the first relevant result:
- `MRR = 1 / rank_of_first_relevant_result`
- Range: 0.0 to 1.0 (higher is better)
- Example: First relevant result at rank 3 → MRR = 0.333

### Recall@K
Measures coverage of relevant results in top K:
- `Recall@K = (relevant results in top K) / (total relevant results)`
- Range: 0.0 to 1.0 (higher is better)
- Example: 2 out of 3 expected files in top 5 → Recall@5 = 0.667

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality with position-based discounting:
- `DCG = Σ (relevance_i / log2(i + 1))`
- `NDCG = DCG / IDCG` (normalized by ideal ranking)
- Range: 0.0 to 1.0 (higher is better)
- Rewards relevant results appearing higher in the ranking

### Latency
Average query execution time in milliseconds (lower is better).

## Usage

### Run Comparison Across Multiple Configurations

Compare different chunk sizes with and without reranker:

```bash
cd <repo-root>

# Test chunk sizes: 512, 1024, 2048, 4096
python scripts/benchmark.py --chunk-sizes 512,1024,2048,4096
```

This will:
1. Create 8 configurations (4 chunk sizes × 2 reranker states)
2. Run 30 test queries for each configuration
3. Calculate metrics (MRR, Recall@5, Recall@10, NDCG@10, Latency)
4. Generate comparison report with highlighted best scores

### Run Single Configuration

Test current settings in `config/settings.yaml`:

```bash
python scripts\benchmark.py --single
```

### Custom Query Set

Use a custom test query file:

```bash
python scripts\benchmark.py --queries my_queries.yaml
```

### Output Directory

Specify where to save results:

```bash
python scripts\benchmark.py --output benchmark/my_results
```

## Test Queries

The benchmark includes 30 test queries across 5 categories:

- **Function Search** (10 queries): Finding specific functions/methods
  - Example: "collision detection function"
  - Expected: `DetectCollision`, `CheckCollision` in physics files

- **Concept Search** (6 queries): Understanding how systems work
  - Example: "how does memory allocation work"
  - Expected: Memory allocator implementation files

- **Error Search** (4 queries): Finding error handling code
  - Example: "null pointer exception handling"
  - Expected: Error handler and validation code

- **Pattern Search** (4 queries): Finding design patterns
  - Example: "observer pattern event system"
  - Expected: Event/observer implementation files

- **API Search** (6 queries): Finding API usage examples
  - Example: "JSON serialization and deserialization"
  - Expected: Serializer implementation files

## Query Format

Queries are defined in YAML:

```yaml
queries:
  - id: func_collision_detection
    query: "collision detection function"
    category: function_search
    expected_files:
      - "Source/Runtime/Physics/CollisionSystem.cpp"
      - "Engine/Collision/CollisionDetector.cs"
    expected_symbols:
      - "DetectCollision"
      - "CheckCollision"
```

## Output Format

### Single Report

```json
{
  "config": {
    "name": "chunk_2048_with_rerank",
    "chunk_size": 2048,
    "embedding_model": "nomic-embed-text-v1.5",
    "use_reranker": true,
    "reranker_model": "BAAI/bge-reranker-base"
  },
  "summary": {
    "total_queries": 30,
    "avg_mrr": 0.7234,
    "avg_recall_at_5": 0.8125,
    "avg_recall_at_10": 0.9167,
    "avg_ndcg_at_10": 0.8456,
    "avg_latency_ms": 145.32
  },
  "category_metrics": {
    "function_search": {
      "count": 10,
      "avg_mrr": 0.7800,
      "avg_recall_at_5": 0.8500,
      ...
    }
  },
  "query_results": [...]
}
```

### Comparison Report

```json
{
  "timestamp": "2026-01-30T12:00:00",
  "num_configs": 8,
  "reports": [...]
}
```

## Example Results

```
╭─────────────────────────────────────────────────────────────────────╮
│                      Benchmark Comparison                           │
├──────────────────────┬──────────┬───────────┬────────────┬─────────┤
│ Configuration        │ MRR      │ Recall@5  │ NDCG@10    │ Latency │
├──────────────────────┼──────────┼───────────┼────────────┼─────────┤
│ chunk_512_no_rerank  │ 0.6543   │ 0.7423    │ 0.7821     │ 95.23   │
│ chunk_512_with_rerank│ 0.7012   │ 0.7891    │ 0.8234     │ 132.45  │
│ chunk_1024_no_rerank │ 0.6789   │ 0.7567    │ 0.7945     │ 98.67   │
│ chunk_1024_with_rerank│0.7234   │ 0.8125    │ 0.8456     │ 145.32  │
│ chunk_2048_no_rerank │ 0.7012   │ 0.7891    │ 0.8123     │ 103.21  │
│ chunk_2048_with_rerank│0.7456   │ 0.8234    │ 0.8567     │ 156.78  │
│ chunk_4096_no_rerank │ 0.6912   │ 0.7734    │ 0.8034     │ 118.45  │
│ chunk_4096_with_rerank│0.7389   │ 0.8156    │ 0.8489     │ 178.92  │
└──────────────────────┴──────────┴───────────┴────────────┴─────────┘

Recommendations:
  Best overall quality: chunk_2048_with_rerank (NDCG@10: 0.8567)
  Fastest: chunk_512_no_rerank (95.23ms)
```

## Interpreting Results

### Good MRR Scores
- **0.8+**: Excellent - First relevant result typically in top 2
- **0.6-0.8**: Good - First relevant result typically in top 3-4
- **0.4-0.6**: Fair - First relevant result typically in top 5-7
- **<0.4**: Poor - Relevant results not appearing early

### Good Recall@5 Scores
- **0.9+**: Excellent - Almost all relevant results in top 5
- **0.7-0.9**: Good - Most relevant results in top 5
- **0.5-0.7**: Fair - Some relevant results missed
- **<0.5**: Poor - Many relevant results missed

### Good NDCG@10 Scores
- **0.9+**: Excellent - Near-perfect ranking
- **0.8-0.9**: Good - Strong ranking quality
- **0.6-0.8**: Fair - Acceptable ranking
- **<0.6**: Poor - Weak ranking quality

## Customizing Queries

Add your own test queries to `queries.yaml`:

```yaml
queries:
  - id: my_test_query
    query: "my search query"
    category: function_search  # or concept_search, error_search, etc.
    expected_files:
      - "path/to/expected/file1.cpp"
      - "path/to/expected/file2.cs"
    expected_symbols:  # Optional
      - "ExpectedFunctionName"
      - "ExpectedClassName"
```

## Best Practices

1. **Representative Queries**: Include queries that represent real usage patterns
2. **Diverse Categories**: Cover all query types (function, concept, error, etc.)
3. **Multiple Expected Results**: List all relevant files/symbols, not just one
4. **Regular Evaluation**: Run benchmarks after major changes to indexing/search
5. **A/B Testing**: Compare configurations side-by-side to find optimal settings

## Integration with CI/CD

Run benchmarks automatically:

```bash
# Fail if quality drops below threshold
python scripts\benchmark.py --single --min-mrr 0.7 --min-ndcg 0.75
```

## Troubleshooting

### No Results for Queries
- Ensure codebase is indexed: `python scripts\ingest.py`
- Check Qdrant is running: `docker ps`
- Verify collection exists: `python scripts\init_collection.py`

### Low Scores
- Review expected files in queries - are they actually in the indexed codebase?
- Check if file paths match exactly (case-sensitive on Linux)
- Ensure embedding server is running: `curl http://localhost:8081/health`

### Slow Performance
- Reduce number of queries for quick tests
- Disable reranker for faster runs
- Use smaller chunk sizes for faster embedding

## Future Enhancements

- [ ] Support for multiple embedding models in same run
- [ ] Precision@K metric
- [ ] Per-file-type breakdown
- [ ] Query difficulty classification
- [ ] Statistical significance testing
- [ ] Visual plots (matplotlib/plotly)
