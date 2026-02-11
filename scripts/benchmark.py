"""
Benchmark Harness - Evaluate embedding and retrieval performance

Compares:
- Different chunk sizes (512, 1024, 2048, 4096, 8000 tokens)
- Different embedding models (nomic-embed-code vs BGE)
- With/without reranker
- Various query types (function search, concept search, error search)

Metrics:
- MRR (Mean Reciprocal Rank)
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Latency
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Import CodeStore from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from code_store import CodeStore, SearchResult as StoreSearchResult


logger = logging.getLogger(__name__)
console = Console()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestQuery:
    """A test query with expected results."""
    id: str
    query: str
    category: str  # function_search, concept_search, error_search, etc.
    expected_files: List[str]  # Files that should be in top results
    expected_symbols: List[str] = field(default_factory=list)  # Optional specific symbols

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """Search result for benchmark."""
    file_path: str
    symbol_name: Optional[str]
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "score": self.score,
            "rank": self.rank
        }


@dataclass
class QueryResult:
    """Result for a single query execution."""
    query_id: str
    query: str
    category: str
    results: List[SearchResult]
    expected_files: List[str]
    expected_symbols: List[str]
    mrr: float  # Mean Reciprocal Rank
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "results": [r.to_dict() for r in self.results],
            "expected_files": self.expected_files,
            "expected_symbols": self.expected_symbols,
            "metrics": {
                "mrr": self.mrr,
                "recall_at_5": self.recall_at_5,
                "recall_at_10": self.recall_at_10,
                "ndcg_at_10": self.ndcg_at_10,
                "latency_ms": self.latency_ms
            }
        }


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    chunk_size: int
    embedding_model: str
    use_reranker: bool
    reranker_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    config: BenchmarkConfig
    timestamp: datetime
    total_queries: int
    avg_mrr: float
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_ndcg_at_10: float
    avg_latency_ms: float
    query_results: List[QueryResult]
    category_metrics: Dict[str, Dict[str, float]]  # Per-category breakdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_queries": self.total_queries,
                "avg_mrr": self.avg_mrr,
                "avg_recall_at_5": self.avg_recall_at_5,
                "avg_recall_at_10": self.avg_recall_at_10,
                "avg_ndcg_at_10": self.avg_ndcg_at_10,
                "avg_latency_ms": self.avg_latency_ms
            },
            "category_metrics": self.category_metrics,
            "query_results": [qr.to_dict() for qr in self.query_results]
        }


# ============================================================================
# Benchmark Runner
# ============================================================================

class RelevanceBenchmark:
    """
    Benchmark harness for evaluating search relevance.

    Usage:
        benchmark = RelevanceBenchmark(
            code_store=store,
            test_queries=load_test_queries("queries.yaml")
        )

        report = await benchmark.run(config)
        benchmark.save_report(report, "benchmark_results.json")
    """

    def __init__(
        self,
        code_store: CodeStore,
        test_queries: List[TestQuery]
    ):
        """Initialize benchmark harness.

        Args:
            code_store: CodeStore instance (must be connected)
            test_queries: List of test queries with expected results
        """
        self.code_store = code_store
        self.test_queries = test_queries

    async def run(self, config: BenchmarkConfig) -> BenchmarkReport:
        """Run benchmark with given configuration.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkReport with all metrics
        """
        console.print(f"\n[bold cyan]Running benchmark: {config.name}[/bold cyan]")
        console.print(f"  Chunk size: {config.chunk_size} tokens")
        console.print(f"  Embedding model: {config.embedding_model}")
        console.print(f"  Reranker: {'enabled' if config.use_reranker else 'disabled'}")

        query_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Processing {len(self.test_queries)} queries...",
                total=len(self.test_queries)
            )

            for test_query in self.test_queries:
                query_result = await self._run_query(test_query, config)
                query_results.append(query_result)
                progress.advance(task)

        # Calculate aggregate metrics
        timestamp = datetime.now()
        total_queries = len(query_results)

        avg_mrr = sum(qr.mrr for qr in query_results) / total_queries
        avg_recall_at_5 = sum(qr.recall_at_5 for qr in query_results) / total_queries
        avg_recall_at_10 = sum(qr.recall_at_10 for qr in query_results) / total_queries
        avg_ndcg_at_10 = sum(qr.ndcg_at_10 for qr in query_results) / total_queries
        avg_latency_ms = sum(qr.latency_ms for qr in query_results) / total_queries

        # Calculate per-category metrics
        category_metrics = self._calculate_category_metrics(query_results)

        report = BenchmarkReport(
            config=config,
            timestamp=timestamp,
            total_queries=total_queries,
            avg_mrr=avg_mrr,
            avg_recall_at_5=avg_recall_at_5,
            avg_recall_at_10=avg_recall_at_10,
            avg_ndcg_at_10=avg_ndcg_at_10,
            avg_latency_ms=avg_latency_ms,
            query_results=query_results,
            category_metrics=category_metrics
        )

        console.print(f"[green]✓ Benchmark complete: {config.name}[/green]")
        self._print_summary(report)

        return report

    async def run_comparison(
        self,
        configs: List[BenchmarkConfig]
    ) -> List[BenchmarkReport]:
        """Run multiple configurations and compare.

        Args:
            configs: List of benchmark configurations

        Returns:
            List of BenchmarkReports
        """
        console.print(f"\n[bold yellow]Running comparison across {len(configs)} configurations[/bold yellow]\n")

        reports = []
        for i, config in enumerate(configs):
            console.print(f"[cyan]Configuration {i+1}/{len(configs)}[/cyan]")
            report = await self.run(config)
            reports.append(report)
            console.print()

        # Print comparison table
        self.print_comparison(reports)

        return reports

    async def _run_query(self, test_query: TestQuery, config: BenchmarkConfig) -> QueryResult:
        """Run a single test query and calculate metrics.

        Args:
            test_query: Test query to execute
            config: Benchmark configuration

        Returns:
            QueryResult with metrics
        """
        # Execute search and measure latency
        start_time = time.perf_counter()

        try:
            # Search with current configuration
            search_results = await self.code_store.search(
                query=test_query.query,
                limit=10,  # Always get top 10 for metrics
                use_reranker=config.use_reranker
            )
        except Exception as e:
            logger.error(f"Search failed for query '{test_query.id}': {e}")
            search_results = []

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Convert to benchmark SearchResults
        results = [
            SearchResult(
                file_path=sr.chunk.file_path,
                symbol_name=sr.chunk.symbol_name,
                score=sr.rerank_score if sr.rerank_score is not None else sr.score,
                rank=i + 1
            )
            for i, sr in enumerate(search_results)
        ]

        # Calculate metrics
        mrr = self._calculate_mrr(results, test_query.expected_files, test_query.expected_symbols)
        recall_at_5 = self._calculate_recall_at_k(results, test_query.expected_files, 5)
        recall_at_10 = self._calculate_recall_at_k(results, test_query.expected_files, 10)
        ndcg_at_10 = self._calculate_ndcg(results, test_query.expected_files, 10)

        return QueryResult(
            query_id=test_query.id,
            query=test_query.query,
            category=test_query.category,
            results=results,
            expected_files=test_query.expected_files,
            expected_symbols=test_query.expected_symbols,
            mrr=mrr,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            ndcg_at_10=ndcg_at_10,
            latency_ms=latency_ms
        )

    def _calculate_mrr(
        self,
        results: List[SearchResult],
        expected_files: List[str],
        expected_symbols: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank.

        MRR = 1 / rank_of_first_relevant_result

        Args:
            results: Search results
            expected_files: Expected file paths
            expected_symbols: Expected symbol names (optional)

        Returns:
            MRR score (0.0 to 1.0)
        """
        if not expected_files and not expected_symbols:
            return 0.0

        for i, result in enumerate(results):
            # Check if file matches
            if any(expected in result.file_path for expected in expected_files):
                return 1.0 / (i + 1)

            # Check if symbol matches (if provided)
            if expected_symbols and result.symbol_name:
                if any(expected.lower() in result.symbol_name.lower() for expected in expected_symbols):
                    return 1.0 / (i + 1)

        return 0.0  # No relevant result found

    def _calculate_recall_at_k(
        self,
        results: List[SearchResult],
        expected_files: List[str],
        k: int
    ) -> float:
        """Calculate Recall@K.

        Recall@K = (number of relevant results in top K) / (total relevant results)

        Args:
            results: Search results
            expected_files: Expected file paths
            k: Number of top results to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not expected_files:
            return 0.0

        top_k_results = results[:k]

        # Count how many expected files are in top K
        found = 0
        for expected in expected_files:
            if any(expected in result.file_path for result in top_k_results):
                found += 1

        return found / len(expected_files)

    def _calculate_ndcg(
        self,
        results: List[SearchResult],
        expected_files: List[str],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.

        NDCG measures ranking quality with position-based discounting.

        DCG = Σ (rel_i / log2(i + 1))
        NDCG = DCG / IDCG

        Args:
            results: Search results
            expected_files: Expected file paths
            k: Number of top results to consider

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if not expected_files:
            return 0.0

        top_k_results = results[:k]

        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(top_k_results):
            # Relevance score: 1 if file is in expected, 0 otherwise
            relevance = 1.0 if any(expected in result.file_path for expected in expected_files) else 0.0

            # Discount by position (log2(i + 2) because i starts at 0)
            dcg += relevance / math.log2(i + 2)

        # Calculate Ideal DCG (assuming perfect ranking)
        idcg = 0.0
        num_relevant = min(len(expected_files), k)
        for i in range(num_relevant):
            idcg += 1.0 / math.log2(i + 2)

        # Normalize
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def _calculate_category_metrics(self, query_results: List[QueryResult]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per category.

        Args:
            query_results: All query results

        Returns:
            Dictionary of category -> metrics
        """
        # Group by category
        categories = {}
        for qr in query_results:
            if qr.category not in categories:
                categories[qr.category] = []
            categories[qr.category].append(qr)

        # Calculate metrics for each category
        category_metrics = {}
        for category, results in categories.items():
            n = len(results)
            category_metrics[category] = {
                "count": n,
                "avg_mrr": sum(r.mrr for r in results) / n,
                "avg_recall_at_5": sum(r.recall_at_5 for r in results) / n,
                "avg_recall_at_10": sum(r.recall_at_10 for r in results) / n,
                "avg_ndcg_at_10": sum(r.ndcg_at_10 for r in results) / n,
                "avg_latency_ms": sum(r.latency_ms for r in results) / n
            }

        return category_metrics

    def _print_summary(self, report: BenchmarkReport):
        """Print summary of a single benchmark report.

        Args:
            report: BenchmarkReport to summarize
        """
        table = Table(title=f"Summary: {report.config.name}", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="yellow")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Queries", str(report.total_queries))
        table.add_row("Avg MRR", f"{report.avg_mrr:.4f}")
        table.add_row("Avg Recall@5", f"{report.avg_recall_at_5:.4f}")
        table.add_row("Avg Recall@10", f"{report.avg_recall_at_10:.4f}")
        table.add_row("Avg NDCG@10", f"{report.avg_ndcg_at_10:.4f}")
        table.add_row("Avg Latency (ms)", f"{report.avg_latency_ms:.2f}")

        console.print(table)

        # Print category breakdown
        if report.category_metrics:
            console.print("\n[bold cyan]Per-Category Metrics:[/bold cyan]")
            for category, metrics in report.category_metrics.items():
                console.print(f"  [yellow]{category}[/yellow] ({metrics['count']} queries):")
                console.print(f"    MRR: {metrics['avg_mrr']:.4f}, "
                            f"Recall@5: {metrics['avg_recall_at_5']:.4f}, "
                            f"NDCG@10: {metrics['avg_ndcg_at_10']:.4f}")

    def print_comparison(self, reports: List[BenchmarkReport]):
        """Print comparison table across multiple reports.

        Args:
            reports: List of BenchmarkReports to compare
        """
        console.print("\n[bold yellow]Benchmark Comparison[/bold yellow]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Configuration", style="yellow")
        table.add_column("MRR", justify="right")
        table.add_column("Recall@5", justify="right")
        table.add_column("Recall@10", justify="right")
        table.add_column("NDCG@10", justify="right")
        table.add_column("Latency (ms)", justify="right")

        # Find best scores for highlighting
        best_mrr = max(r.avg_mrr for r in reports)
        best_recall5 = max(r.avg_recall_at_5 for r in reports)
        best_recall10 = max(r.avg_recall_at_10 for r in reports)
        best_ndcg = max(r.avg_ndcg_at_10 for r in reports)
        best_latency = min(r.avg_latency_ms for r in reports)

        for report in reports:
            config_name = f"{report.config.name}\n({report.config.chunk_size} tokens)"

            mrr_str = f"{report.avg_mrr:.4f}"
            if report.avg_mrr == best_mrr:
                mrr_str = f"[bold green]{mrr_str}[/bold green]"

            recall5_str = f"{report.avg_recall_at_5:.4f}"
            if report.avg_recall_at_5 == best_recall5:
                recall5_str = f"[bold green]{recall5_str}[/bold green]"

            recall10_str = f"{report.avg_recall_at_10:.4f}"
            if report.avg_recall_at_10 == best_recall10:
                recall10_str = f"[bold green]{recall10_str}[/bold green]"

            ndcg_str = f"{report.avg_ndcg_at_10:.4f}"
            if report.avg_ndcg_at_10 == best_ndcg:
                ndcg_str = f"[bold green]{ndcg_str}[/bold green]"

            latency_str = f"{report.avg_latency_ms:.2f}"
            if report.avg_latency_ms == best_latency:
                latency_str = f"[bold green]{latency_str}[/bold green]"

            table.add_row(
                config_name,
                mrr_str,
                recall5_str,
                recall10_str,
                ndcg_str,
                latency_str
            )

        console.print(table)

        # Print recommendations
        console.print("\n[bold cyan]Recommendations:[/bold cyan]")
        best_overall = max(reports, key=lambda r: r.avg_ndcg_at_10)
        console.print(f"  Best overall quality: [green]{best_overall.config.name}[/green] "
                     f"(NDCG@10: {best_overall.avg_ndcg_at_10:.4f})")

        fastest = min(reports, key=lambda r: r.avg_latency_ms)
        console.print(f"  Fastest: [green]{fastest.config.name}[/green] "
                     f"({fastest.avg_latency_ms:.2f}ms)")

    def save_report(self, report: BenchmarkReport, output_path: str):
        """Save benchmark report to JSON.

        Args:
            report: BenchmarkReport to save
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2)

        console.print(f"[green]✓ Report saved: {output_file}[/green]")

    def save_comparison(self, reports: List[BenchmarkReport], output_path: str):
        """Save comparison report to JSON.

        Args:
            reports: List of BenchmarkReports
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "num_configs": len(reports),
            "reports": [r.to_dict() for r in reports]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)

        console.print(f"[green]✓ Comparison report saved: {output_file}[/green]")


# ============================================================================
# Utilities
# ============================================================================

def load_test_queries(path: str) -> List[TestQuery]:
    """Load test queries from YAML file.

    Args:
        path: Path to YAML file with queries

    Returns:
        List of TestQuery objects
    """
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Test queries file not found: {path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    queries = []
    for q in data.get('queries', []):
        query = TestQuery(
            id=q['id'],
            query=q['query'],
            category=q['category'],
            expected_files=q.get('expected_files', []),
            expected_symbols=q.get('expected_symbols', [])
        )
        queries.append(query)

    return queries


def create_default_configs(chunk_sizes: List[int], settings_path: str) -> List[BenchmarkConfig]:
    """Create default benchmark configurations.

    Args:
        chunk_sizes: List of chunk sizes to test
        settings_path: Path to settings.yaml

    Returns:
        List of BenchmarkConfig objects
    """
    # Load current settings to get model names
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    embedding_model = settings['embedding']['model']
    reranker_model = settings['reranker']['model']

    configs = []

    for chunk_size in chunk_sizes:
        # Without reranker
        configs.append(BenchmarkConfig(
            name=f"chunk_{chunk_size}_no_rerank",
            chunk_size=chunk_size,
            embedding_model=embedding_model,
            use_reranker=False
        ))

        # With reranker
        configs.append(BenchmarkConfig(
            name=f"chunk_{chunk_size}_with_rerank",
            chunk_size=chunk_size,
            embedding_model=embedding_model,
            use_reranker=True,
            reranker_model=reranker_model
        ))

    return configs


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option('--queries', default='benchmark/queries.yaml', help='Path to test queries YAML')
@click.option('--output', default='benchmark/results', help='Output directory for reports')
@click.option('--chunk-sizes', default='512,1024,2048,4096', help='Comma-separated chunk sizes to test')
@click.option('--settings', default='config/settings.yaml', help='Path to settings.yaml')
@click.option('--single', is_flag=True, help='Run single configuration (current settings)')
def main(queries: str, output: str, chunk_sizes: str, settings: str, single: bool):
    """Run relevance benchmark for Akashic Records search system.

    Examples:
        # Run comparison across multiple chunk sizes
        python benchmark.py --chunk-sizes 512,1024,2048

        # Run single configuration with current settings
        python benchmark.py --single
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def run_benchmark():
        # Load test queries
        try:
            test_queries = load_test_queries(queries)
            console.print(f"[green]✓ Loaded {len(test_queries)} test queries from {queries}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load test queries: {e}[/red]")
            return

        # Initialize CodeStore
        console.print(f"[cyan]Connecting to CodeStore...[/cyan]")
        code_store = CodeStore(settings_path=settings)

        try:
            await code_store.connect()
            console.print(f"[green]✓ CodeStore connected[/green]")

            # Create benchmark
            benchmark = RelevanceBenchmark(
                code_store=code_store,
                test_queries=test_queries
            )

            if single:
                # Run single configuration with current settings
                with open(settings, 'r', encoding='utf-8') as f:
                    current_settings = yaml.safe_load(f)

                config = BenchmarkConfig(
                    name="current_config",
                    chunk_size=current_settings['indexing']['chunk']['max_tokens'],
                    embedding_model=current_settings['embedding']['model'],
                    use_reranker=current_settings['reranker']['enabled'],
                    reranker_model=current_settings['reranker']['model']
                )

                report = await benchmark.run(config)

                # Save report
                output_file = Path(output) / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                benchmark.save_report(report, str(output_file))

            else:
                # Run comparison across multiple configurations
                chunk_size_list = [int(x.strip()) for x in chunk_sizes.split(',')]
                configs = create_default_configs(chunk_size_list, settings)

                console.print(f"[yellow]Running {len(configs)} configurations...[/yellow]")

                reports = await benchmark.run_comparison(configs)

                # Save comparison report
                output_file = Path(output) / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                benchmark.save_comparison(reports, str(output_file))

        finally:
            await code_store.close()
            console.print("[cyan]CodeStore closed[/cyan]")

    # Run async benchmark
    asyncio.run(run_benchmark())


if __name__ == '__main__':
    main()
