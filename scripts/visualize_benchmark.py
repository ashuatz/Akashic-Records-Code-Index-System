"""
Benchmark Visualization - Generate comparison charts and reports

Usage:
    python visualize_benchmark.py benchmark/results/comparison_*.json
    python visualize_benchmark.py --report comparison_20260130.json --output charts/
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import sys

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()


def load_comparison_report(report_path: str) -> Dict[str, Any]:
    """Load a comparison report JSON file.

    Args:
        report_path: Path to comparison JSON file

    Returns:
        Parsed report data
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_comparison_table(reports: List[Dict[str, Any]]) -> Table:
    """Create a rich table comparing multiple reports.

    Args:
        reports: List of report dictionaries

    Returns:
        Rich Table object
    """
    table = Table(title="Benchmark Comparison", show_header=True, header_style="bold cyan")

    table.add_column("Configuration", style="yellow", width=30)
    table.add_column("Chunk Size", justify="right", style="blue")
    table.add_column("MRR", justify="right", width=8)
    table.add_column("Recall@5", justify="right", width=10)
    table.add_column("Recall@10", justify="right", width=11)
    table.add_column("NDCG@10", justify="right", width=10)
    table.add_column("Latency (ms)", justify="right", width=13)

    # Find best scores
    if reports:
        best_mrr = max(r['summary']['avg_mrr'] for r in reports)
        best_recall5 = max(r['summary']['avg_recall_at_5'] for r in reports)
        best_recall10 = max(r['summary']['avg_recall_at_10'] for r in reports)
        best_ndcg = max(r['summary']['avg_ndcg_at_10'] for r in reports)
        best_latency = min(r['summary']['avg_latency_ms'] for r in reports)

        for report in reports:
            config = report['config']
            summary = report['summary']

            # Format values with highlighting for best scores
            mrr = summary['avg_mrr']
            mrr_str = f"[bold green]{mrr:.4f}[/bold green]" if mrr == best_mrr else f"{mrr:.4f}"

            recall5 = summary['avg_recall_at_5']
            recall5_str = f"[bold green]{recall5:.4f}[/bold green]" if recall5 == best_recall5 else f"{recall5:.4f}"

            recall10 = summary['avg_recall_at_10']
            recall10_str = f"[bold green]{recall10:.4f}[/bold green]" if recall10 == best_recall10 else f"{recall10:.4f}"

            ndcg = summary['avg_ndcg_at_10']
            ndcg_str = f"[bold green]{ndcg:.4f}[/bold green]" if ndcg == best_ndcg else f"{ndcg:.4f}"

            latency = summary['avg_latency_ms']
            latency_str = f"[bold green]{latency:.2f}[/bold green]" if latency == best_latency else f"{latency:.2f}"

            reranker_status = "✓" if config['use_reranker'] else "✗"
            config_name = f"{config['name']}\n(Reranker: {reranker_status})"

            table.add_row(
                config_name,
                str(config['chunk_size']),
                mrr_str,
                recall5_str,
                recall10_str,
                ndcg_str,
                latency_str
            )

    return table


def create_category_breakdown(report: Dict[str, Any]) -> Table:
    """Create a table showing per-category metrics.

    Args:
        report: Report dictionary

    Returns:
        Rich Table object
    """
    table = Table(title=f"Category Breakdown: {report['config']['name']}", show_header=True, header_style="bold cyan")

    table.add_column("Category", style="yellow")
    table.add_column("Queries", justify="right", style="blue")
    table.add_column("MRR", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Latency (ms)", justify="right")

    category_metrics = report.get('category_metrics', {})

    for category, metrics in category_metrics.items():
        table.add_row(
            category.replace('_', ' ').title(),
            str(metrics['count']),
            f"{metrics['avg_mrr']:.4f}",
            f"{metrics['avg_recall_at_5']:.4f}",
            f"{metrics['avg_recall_at_10']:.4f}",
            f"{metrics['avg_ndcg_at_10']:.4f}",
            f"{metrics['avg_latency_ms']:.2f}"
        )

    return table


def print_recommendations(reports: List[Dict[str, Any]]):
    """Print recommendations based on benchmark results.

    Args:
        reports: List of report dictionaries
    """
    if not reports:
        return

    # Find best configurations for different use cases
    best_quality = max(reports, key=lambda r: r['summary']['avg_ndcg_at_10'])
    best_speed = min(reports, key=lambda r: r['summary']['avg_latency_ms'])
    best_recall = max(reports, key=lambda r: r['summary']['avg_recall_at_10'])

    # Calculate quality/speed score (normalized)
    max_ndcg = max(r['summary']['avg_ndcg_at_10'] for r in reports)
    min_latency = min(r['summary']['avg_latency_ms'] for r in reports)
    max_latency = max(r['summary']['avg_latency_ms'] for r in reports)

    for r in reports:
        # Normalize metrics (0-1 scale)
        quality_score = r['summary']['avg_ndcg_at_10'] / max_ndcg if max_ndcg > 0 else 0
        # Invert latency (lower is better)
        speed_score = 1 - ((r['summary']['avg_latency_ms'] - min_latency) / (max_latency - min_latency)) if max_latency > min_latency else 1
        r['_balance_score'] = (quality_score + speed_score) / 2

    best_balance = max(reports, key=lambda r: r['_balance_score'])

    recommendations = [
        ("Best Overall Quality", best_quality, f"NDCG@10: {best_quality['summary']['avg_ndcg_at_10']:.4f}"),
        ("Fastest", best_speed, f"Latency: {best_speed['summary']['avg_latency_ms']:.2f}ms"),
        ("Best Recall", best_recall, f"Recall@10: {best_recall['summary']['avg_recall_at_10']:.4f}"),
        ("Best Balance (Quality/Speed)", best_balance, f"Score: {best_balance['_balance_score']:.4f}")
    ]

    console.print("\n[bold cyan]Recommendations:[/bold cyan]\n")

    for title, report, metric in recommendations:
        config_name = report['config']['name']
        chunk_size = report['config']['chunk_size']
        reranker = "enabled" if report['config']['use_reranker'] else "disabled"

        panel = Panel(
            f"[green]{config_name}[/green]\n"
            f"Chunk size: {chunk_size} tokens\n"
            f"Reranker: {reranker}\n"
            f"{metric}",
            title=f"[bold yellow]{title}[/bold yellow]",
            border_style="cyan"
        )
        console.print(panel)


def print_insights(reports: List[Dict[str, Any]]):
    """Print insights and patterns from benchmark results.

    Args:
        reports: List of report dictionaries
    """
    if not reports:
        return

    console.print("\n[bold cyan]Insights:[/bold cyan]\n")

    # Analyze chunk size impact
    chunk_sizes = {}
    for r in reports:
        size = r['config']['chunk_size']
        if size not in chunk_sizes:
            chunk_sizes[size] = {'with_rerank': [], 'without_rerank': []}

        if r['config']['use_reranker']:
            chunk_sizes[size]['with_rerank'].append(r['summary']['avg_ndcg_at_10'])
        else:
            chunk_sizes[size]['without_rerank'].append(r['summary']['avg_ndcg_at_10'])

    # Find optimal chunk size
    if chunk_sizes:
        avg_scores = {}
        for size, data in chunk_sizes.items():
            all_scores = data['with_rerank'] + data['without_rerank']
            if all_scores:
                avg_scores[size] = sum(all_scores) / len(all_scores)

        if avg_scores:
            best_chunk_size = max(avg_scores, key=avg_scores.get)
            console.print(f"• [yellow]Optimal chunk size:[/yellow] {best_chunk_size} tokens "
                         f"(avg NDCG: {avg_scores[best_chunk_size]:.4f})")

    # Analyze reranker impact
    with_reranker = [r for r in reports if r['config']['use_reranker']]
    without_reranker = [r for r in reports if not r['config']['use_reranker']]

    if with_reranker and without_reranker:
        avg_with = sum(r['summary']['avg_ndcg_at_10'] for r in with_reranker) / len(with_reranker)
        avg_without = sum(r['summary']['avg_ndcg_at_10'] for r in without_reranker) / len(without_reranker)
        improvement = ((avg_with - avg_without) / avg_without * 100) if avg_without > 0 else 0

        avg_latency_with = sum(r['summary']['avg_latency_ms'] for r in with_reranker) / len(with_reranker)
        avg_latency_without = sum(r['summary']['avg_latency_ms'] for r in without_reranker) / len(without_reranker)
        latency_increase = ((avg_latency_with - avg_latency_without) / avg_latency_without * 100) if avg_latency_without > 0 else 0

        console.print(f"• [yellow]Reranker impact:[/yellow] "
                     f"+{improvement:.1f}% quality (NDCG), "
                     f"+{latency_increase:.1f}% latency")

    # Category performance analysis
    all_categories = set()
    for r in reports:
        all_categories.update(r.get('category_metrics', {}).keys())

    if all_categories:
        console.print(f"\n• [yellow]Query categories tested:[/yellow] {', '.join(sorted(all_categories))}")

        # Find hardest category
        category_scores = {}
        for category in all_categories:
            scores = []
            for r in reports:
                cat_metrics = r.get('category_metrics', {}).get(category)
                if cat_metrics:
                    scores.append(cat_metrics['avg_ndcg_at_10'])

            if scores:
                category_scores[category] = sum(scores) / len(scores)

        if category_scores:
            easiest = max(category_scores, key=category_scores.get)
            hardest = min(category_scores, key=category_scores.get)

            console.print(f"• [yellow]Easiest category:[/yellow] {easiest} (avg NDCG: {category_scores[easiest]:.4f})")
            console.print(f"• [yellow]Hardest category:[/yellow] {hardest} (avg NDCG: {category_scores[hardest]:.4f})")

    console.print()


@click.command()
@click.argument('report_file', type=click.Path(exists=True))
@click.option('--show-categories', is_flag=True, help='Show per-category breakdown for each config')
@click.option('--output', help='Save summary to text file')
def main(report_file: str, show_categories: bool, output: str):
    """Visualize benchmark comparison results.

    REPORT_FILE: Path to comparison JSON file

    Examples:
        # View latest comparison
        python visualize_benchmark.py benchmark/results/comparison_latest.json

        # Show category breakdown
        python visualize_benchmark.py report.json --show-categories

        # Save summary to file
        python visualize_benchmark.py report.json --output summary.txt
    """
    try:
        # Load report
        report_data = load_comparison_report(report_file)
        reports = report_data.get('reports', [])

        if not reports:
            console.print("[red]No reports found in file[/red]")
            return

        console.print(f"\n[bold cyan]Benchmark Report:[/bold cyan] {Path(report_file).name}")
        console.print(f"Timestamp: {report_data.get('timestamp', 'N/A')}")
        console.print(f"Configurations tested: {len(reports)}\n")

        # Create and display comparison table
        comparison_table = create_comparison_table(reports)
        console.print(comparison_table)

        # Show category breakdowns if requested
        if show_categories:
            console.print("\n")
            for report in reports:
                category_table = create_category_breakdown(report)
                console.print(category_table)
                console.print()

        # Print recommendations
        print_recommendations(reports)

        # Print insights
        print_insights(reports)

        # Save to file if requested
        if output:
            with console.capture() as capture:
                console.print(comparison_table)
                if show_categories:
                    for report in reports:
                        console.print(create_category_breakdown(report))
                print_recommendations(reports)
                print_insights(reports)

            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(capture.get())

            console.print(f"[green]✓ Summary saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()
