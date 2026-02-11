"""
Benchmark Suite Runner - Run predefined configuration sets

Usage:
    python run_benchmark_suite.py --suite quick_test
    python run_benchmark_suite.py --suite comprehensive --output results/comprehensive
"""

import asyncio
import sys
from pathlib import Path
import logging

import click
import yaml
from rich.console import Console

# Import benchmark components
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from code_store import CodeStore
from benchmark import RelevanceBenchmark, BenchmarkConfig, load_test_queries


console = Console()
logger = logging.getLogger(__name__)


def load_config_suite(suite_file: str, suite_name: str) -> list[BenchmarkConfig]:
    """Load a configuration suite from YAML.

    Args:
        suite_file: Path to YAML file with configuration suites
        suite_name: Name of the suite to load

    Returns:
        List of BenchmarkConfig objects
    """
    suite_path = Path(suite_file)

    if not suite_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {suite_file}")

    with open(suite_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if suite_name not in data:
        available = ", ".join(data.keys())
        raise ValueError(f"Suite '{suite_name}' not found. Available: {available}")

    suite_configs = data[suite_name]
    configs = []

    for cfg in suite_configs:
        config = BenchmarkConfig(
            name=cfg['name'],
            chunk_size=cfg['chunk_size'],
            embedding_model=cfg['embedding_model'],
            use_reranker=cfg['use_reranker'],
            reranker_model=cfg.get('reranker_model')
        )
        configs.append(config)

    return configs


@click.command()
@click.option('--suite', required=True, help='Name of configuration suite to run')
@click.option('--config-file', default='benchmark/example_configs.yaml', help='Path to configuration YAML')
@click.option('--queries', default='benchmark/queries.yaml', help='Path to test queries YAML')
@click.option('--output', default='benchmark/results', help='Output directory for reports')
@click.option('--settings', default='config/settings.yaml', help='Path to settings.yaml')
@click.option('--list-suites', is_flag=True, help='List available configuration suites')
def main(suite: str, config_file: str, queries: str, output: str, settings: str, list_suites: bool):
    """Run a predefined benchmark configuration suite.

    Available suites (see benchmark/example_configs.yaml):
    - quick_test: 2 configs, fast validation
    - chunk_comparison: Compare 4 different chunk sizes
    - embedding_comparison: Compare 4 different embedding models
    - reranker_study: Ablation study for reranker impact
    - comprehensive: Full 12-config test
    - speed_quality: Trade-off analysis

    Examples:
        # Quick validation
        python run_benchmark_suite.py --suite quick_test

        # Comprehensive chunk size study
        python run_benchmark_suite.py --suite chunk_comparison --output results/chunks

        # Compare embedding models
        python run_benchmark_suite.py --suite embedding_comparison
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # List available suites if requested
    if list_suites:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            console.print("\n[bold cyan]Available Benchmark Suites:[/bold cyan]\n")
            for suite_name, configs in data.items():
                console.print(f"  [yellow]{suite_name}[/yellow] ({len(configs)} configurations)")
            console.print()
            return
        except Exception as e:
            console.print(f"[red]Error loading configuration file: {e}[/red]")
            return

    async def run_suite():
        # Load configuration suite
        try:
            configs = load_config_suite(config_file, suite)
            console.print(f"[green]✓ Loaded suite '{suite}' with {len(configs)} configurations[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load suite: {e}[/red]")
            return

        # Load test queries
        try:
            test_queries = load_test_queries(queries)
            console.print(f"[green]✓ Loaded {len(test_queries)} test queries[/green]")
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

            # Run comparison
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            reports = await benchmark.run_comparison(configs)

            # Save comparison report
            output_path = Path(output)
            output_file = output_path / f"{suite}_{timestamp}.json"
            benchmark.save_comparison(reports, str(output_file))

            # Also save individual reports
            for report in reports:
                report_file = output_path / f"{report.config.name}_{timestamp}.json"
                benchmark.save_report(report, str(report_file))

            console.print(f"\n[bold green]✓ Benchmark suite '{suite}' completed![/bold green]")
            console.print(f"[cyan]Results saved to: {output_path}[/cyan]\n")

        finally:
            await code_store.close()
            console.print("[cyan]CodeStore closed[/cyan]")

    # Run async suite
    asyncio.run(run_suite())


if __name__ == '__main__':
    main()
