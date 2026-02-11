"""
Akashic Records Ingestion CLI

Index codebases into the vector database with incremental updates,
parallel processing, and comprehensive error handling.

Usage:
    python scripts/ingest.py --path "D:/Code/MyProject" --recursive
    python scripts/ingest.py status
    python scripts/ingest.py delete --path "D:/Code/OldProject"
"""

import asyncio
import gc
import hashlib
import logging
import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import click
import psutil
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime_config import load_dotenv, load_settings as load_runtime_settings, resolve_settings_path

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console(force_terminal=True, legacy_windows=False)


def load_settings(settings_path: str) -> dict:
    """Load settings from YAML configuration file."""
    try:
        return load_runtime_settings(settings_path)
    except FileNotFoundError:
        console.print(f"[red]Error: Settings file not found at {settings_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading settings file: {e}[/red]")
        sys.exit(1)


def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file content for change detection."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash {file_path}: {e}")
        return ""


def matches_exclude_pattern(file_path: Path, exclude_patterns: List[str]) -> bool:
    """Check if file path matches any exclude pattern."""
    path_str = str(file_path).replace('\\', '/')

    for pattern in exclude_patterns:
        # Simple glob-like pattern matching
        pattern = pattern.replace('\\', '/')

        # Convert glob pattern to simple matching
        if '*' in pattern:
            # Handle patterns like "node_modules/*"
            if pattern.endswith('/*'):
                dir_pattern = pattern[:-2]
                if dir_pattern in path_str:
                    return True
            # Handle patterns like "*.min.js"
            elif pattern.startswith('*.'):
                ext_pattern = pattern[1:]
                if path_str.endswith(ext_pattern):
                    return True
        else:
            # Direct substring match
            if pattern in path_str:
                return True

    return False


def scan_files(
    paths: List[str],
    settings: dict,
    recursive: bool = True
) -> List[Path]:
    """
    Scan directories for indexable files.

    Args:
        paths: List of directory paths to scan
        settings: Settings dictionary with include_extensions and exclude_patterns
        recursive: Whether to scan subdirectories

    Returns:
        List of Path objects for files to index
    """
    include_ext = settings['indexing']['include_extensions']
    exclude_patterns = settings['indexing']['exclude_patterns']

    files = []

    for path_str in paths:
        path = Path(path_str).resolve()

        if not path.exists():
            console.print(f"[yellow]Warning: Path does not exist: {path}[/yellow]")
            continue

        if path.is_file():
            # Single file
            if path.suffix in include_ext:
                if not matches_exclude_pattern(path, exclude_patterns):
                    files.append(path)
        elif path.is_dir():
            # Directory
            if recursive:
                pattern = '**/*'
            else:
                pattern = '*'

            for file_path in path.glob(pattern):
                if file_path.is_file():
                    if file_path.suffix in include_ext:
                        if not matches_exclude_pattern(file_path, exclude_patterns):
                            files.append(file_path)

    return files


def get_stored_file_hash(db_path: str, file_path: Path) -> Optional[str]:
    """Get the stored hash for a file from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT file_hash FROM files WHERE file_path = ?",
            (str(file_path),)
        )
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None
    except sqlite3.Error as e:
        logger.error(f"Database error getting file hash: {e}")
        return None


def should_index(
    file_path: Path,
    stored_hash: Optional[str],
    incremental: bool
) -> bool:
    """
    Check if file needs indexing.

    Args:
        file_path: Path to the file
        stored_hash: Previously stored hash from database
        incremental: Whether to use incremental mode

    Returns:
        True if file should be indexed
    """
    if not incremental:
        return True

    if stored_hash is None:
        # New file
        return True

    current_hash = get_file_hash(file_path)
    return current_hash != stored_hash


def batched(items: List, batch_size: int) -> Iterator[List]:
    """Yield successive batches from items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


async def ingest_file(
    file_path: Path,
    settings: dict,
    db_path: str,
    skip_errors: bool,
    chunker,
    code_store
) -> Tuple[bool, int, Optional[str]]:
    """
    Index a single file.

    Args:
        file_path: Path to file to index
        settings: Settings dictionary
        db_path: Path to SQLite database
        skip_errors: Whether to skip errors and continue
        chunker: CodeChunker instance
        code_store: CodeStore instance

    Returns:
        Tuple of (success, chunk_count, error_message)
    """
    try:
        file_hash = get_file_hash(file_path)

        # Chunk the file
        chunks = chunker.chunk_file(file_path)

        if not chunks:
            logger.warning(f"No chunks extracted from {file_path}")
            return (True, 0, None)

        # Add chunks to store
        await code_store.add_chunks(chunks)

        chunk_count = len(chunks)

        # Store file metadata
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Detect language
        language = chunker.detect_language(file_path)

        cursor.execute('''
            INSERT OR REPLACE INTO files (file_path, file_hash, indexed_at, language, chunk_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (str(file_path), file_hash, datetime.utcnow().isoformat(), language, chunk_count))

        conn.commit()
        conn.close()

        return (True, chunk_count, None)

    except Exception as e:
        error_msg = f"Failed to index {file_path}: {e}"
        logger.error(error_msg)

        if not skip_errors:
            raise

        return (False, 0, error_msg)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def check_memory_limit(max_memory_mb: int) -> bool:
    """Check if memory usage is within limit. Returns True if OK."""
    current_mb = get_memory_usage_mb()
    return current_mb < max_memory_mb


async def ingest_files(
    files: List[Path],
    settings: dict,
    batch_size: int,
    workers: int,
    skip_errors: bool,
    db_path: str,
    chunker,
    code_store
) -> dict:
    """
    Index multiple files with progress bar and parallel processing.

    Args:
        files: List of files to index
        settings: Settings dictionary
        batch_size: Number of files per batch
        workers: Number of parallel workers
        skip_errors: Whether to skip errors
        db_path: Path to SQLite database
        chunker: CodeChunker instance
        code_store: CodeStore instance

    Returns:
        Dictionary with statistics
    """
    stats = {
        'success_count': 0,
        'error_count': 0,
        'total_chunks': 0,
        'errors': []
    }

    # Memory management settings
    gc_interval = settings['indexing'].get('gc_interval', 50)
    max_memory_mb = settings['indexing'].get('max_memory_mb', 43000)  # ~42GB default
    files_processed = 0

    # Create progress bar (without SpinnerColumn to avoid unicode issues on Windows)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("-"),
        TimeElapsedColumn(),
        TextColumn("-"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task(
            "[cyan]Indexing files...",
            total=len(files)
        )

        # Process files in batches
        for batch in batched(files, batch_size):
            # Memory check before processing batch
            if not check_memory_limit(max_memory_mb):
                logger.warning(f"Memory usage exceeds {max_memory_mb}MB, forcing garbage collection...")
                gc.collect()
                await asyncio.sleep(1)  # Give system time to free memory

                # If still over limit, wait longer
                if not check_memory_limit(max_memory_mb):
                    logger.warning("Memory still high, waiting 5 seconds...")
                    await asyncio.sleep(5)
                    gc.collect()

            # Create tasks for parallel processing
            tasks = [
                ingest_file(file_path, settings, db_path, skip_errors, chunker, code_store)
                for file_path in batch
            ]

            # Execute batch with limited concurrency
            semaphore = asyncio.Semaphore(workers)

            async def bounded_task(t):
                async with semaphore:
                    return await t

            results = await asyncio.gather(
                *[bounded_task(t) for t in tasks],
                return_exceptions=True
            )

            # Process results
            for file_path, result in zip(batch, results):
                files_processed += 1

                if isinstance(result, Exception):
                    stats['error_count'] += 1
                    error_msg = f"Exception indexing {file_path}: {result}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)
                else:
                    success, chunk_count, error = result
                    if success:
                        stats['success_count'] += 1
                        stats['total_chunks'] += chunk_count
                    else:
                        stats['error_count'] += 1
                        if error:
                            stats['errors'].append(error)

                progress.update(task, advance=1)

            # Periodic garbage collection
            if files_processed % gc_interval == 0:
                gc.collect()
                mem_mb = get_memory_usage_mb()
                logger.info(f"GC at {files_processed} files, memory: {mem_mb:.0f}MB")

    return stats


async def run_ingest(
    paths: List[str],
    settings: dict,
    settings_path: str,
    recursive: bool,
    incremental: bool,
    batch_size: int,
    workers: int,
    skip_errors: bool,
    dry_run: bool
):
    """Main ingestion workflow."""

    console.print("\n[bold cyan]Akashic Records - Codebase Ingestion[/bold cyan]\n")

    # Get database path
    db_path = settings['metadata']['db_path']

    # Scan for files
    console.print("[yellow]Scanning for files...[/yellow]")
    all_files = scan_files(paths, settings, recursive)
    console.print(f"[green]Found {len(all_files)} files[/green]\n")

    if len(all_files) == 0:
        console.print("[yellow]No files to index[/yellow]")
        return

    # Filter for incremental updates
    files_to_index = []

    if incremental:
        console.print("[yellow]Checking for changed files...[/yellow]")
        for file_path in all_files:
            stored_hash = get_stored_file_hash(db_path, file_path)
            if should_index(file_path, stored_hash, incremental):
                files_to_index.append(file_path)

        skipped = len(all_files) - len(files_to_index)
        console.print(
            f"[green]{len(files_to_index)} files need indexing "
            f"({skipped} unchanged)[/green]\n"
        )
    else:
        files_to_index = all_files

    if len(files_to_index) == 0:
        console.print("[yellow]No files need indexing[/yellow]")
        return

    # Dry run mode
    if dry_run:
        console.print("[bold yellow]DRY RUN MODE - No changes will be made[/bold yellow]\n")

        table = Table(title="Files to Index")
        table.add_column("Path", style="cyan", no_wrap=False)
        table.add_column("Size", style="green", justify="right")

        for file_path in files_to_index[:20]:  # Show first 20
            size = file_path.stat().st_size
            size_str = f"{size:,} bytes"
            table.add_row(str(file_path), size_str)

        if len(files_to_index) > 20:
            table.add_row(f"... and {len(files_to_index) - 20} more files", "")

        console.print(table)
        console.print(f"\n[yellow]Total files: {len(files_to_index)}[/yellow]")
        return

    # Initialize CodeStore and CodeChunker
    console.print("[yellow]Initializing CodeStore and CodeChunker...[/yellow]")

    # Import from src
    import sys
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    from code_store import CodeStore
    from chunker import CodeChunker

    code_store = CodeStore(str(resolve_settings_path(settings_path)))
    await code_store.connect()

    chunk_settings = settings['indexing']['chunk']
    chunker = CodeChunker(
        max_tokens=chunk_settings.get('max_tokens', 4000),
        overlap=chunk_settings.get('overlap_tokens', 200)
    )

    console.print("[green]Initialized successfully[/green]\n")

    # Perform ingestion
    console.print(f"[cyan]Starting ingestion with {workers} workers...[/cyan]\n")

    start_time = datetime.now()
    try:
        stats = await ingest_files(
            files_to_index,
            settings,
            batch_size,
            workers,
            skip_errors,
            db_path,
            chunker,
            code_store
        )
    finally:
        await code_store.close()
    end_time = datetime.now()

    # Print results
    console.print("\n[bold cyan]Ingestion Complete[/bold cyan]\n")

    duration = (end_time - start_time).total_seconds()

    result_table = Table(show_header=False)
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")

    result_table.add_row("Files processed", str(stats['success_count']))
    result_table.add_row("Files failed", str(stats['error_count']))
    result_table.add_row("Total chunks", str(stats['total_chunks']))
    result_table.add_row("Duration", f"{duration:.2f}s")
    result_table.add_row("Files/sec", f"{stats['success_count']/duration:.2f}")

    console.print(result_table)

    # Show errors if any
    if stats['errors']:
        console.print(f"\n[yellow]Encountered {len(stats['errors'])} errors:[/yellow]")
        for i, error in enumerate(stats['errors'][:10], 1):
            console.print(f"  {i}. {error}")
        if len(stats['errors']) > 10:
            console.print(f"  ... and {len(stats['errors']) - 10} more errors")

    # Update metadata
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE index_metadata SET value = ?, updated_at = ? WHERE key = ?",
            (datetime.utcnow().isoformat(), datetime.utcnow(), 'last_indexed')
        )

        cursor.execute("SELECT COUNT(*) FROM files")
        total_files = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE index_metadata SET value = ?, updated_at = ? WHERE key = ?",
            (str(total_files), datetime.utcnow(), 'total_files')
        )

        cursor.execute("SELECT SUM(chunk_count) FROM files")
        total_chunks = cursor.fetchone()[0] or 0
        cursor.execute(
            "UPDATE index_metadata SET value = ?, updated_at = ? WHERE key = ?",
            (str(total_chunks), datetime.utcnow(), 'total_chunks')
        )

        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Failed to update metadata: {e}")


def show_status(settings: dict):
    """Show indexing statistics."""
    db_path = settings['metadata']['db_path']

    console.print("\n[bold cyan]Akashic Records - Index Status[/bold cyan]\n")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get metadata
        cursor.execute("SELECT key, value, updated_at FROM index_metadata")
        metadata = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        # Get file counts by language
        cursor.execute('''
            SELECT language, COUNT(*), SUM(chunk_count)
            FROM files
            GROUP BY language
            ORDER BY COUNT(*) DESC
        ''')
        by_language = cursor.fetchall()

        # Overall stats
        cursor.execute("SELECT COUNT(*) FROM files")
        total_files = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM symbols")
        total_symbols = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]

        conn.close()

        # Display overview
        overview = Table(title="Index Overview", show_header=False)
        overview.add_column("Metric", style="cyan")
        overview.add_column("Value", style="green")

        overview.add_row("Total Files", str(total_files))
        overview.add_row("Total Symbols", str(total_symbols))
        overview.add_row("Total Chunks", str(total_chunks))

        if 'last_indexed' in metadata:
            last_indexed, _ = metadata['last_indexed']
            overview.add_row("Last Indexed", last_indexed)

        console.print(overview)

        # Display by language
        if by_language:
            console.print("\n")
            lang_table = Table(title="Files by Language")
            lang_table.add_column("Language", style="cyan")
            lang_table.add_column("Files", style="green", justify="right")
            lang_table.add_column("Chunks", style="green", justify="right")

            for lang, file_count, chunk_count in by_language:
                lang_name = lang if lang else "Unknown"
                lang_table.add_row(
                    lang_name,
                    str(file_count),
                    str(chunk_count or 0)
                )

            console.print(lang_table)

        console.print()

    except sqlite3.Error as e:
        console.print(f"[red]Database error: {e}[/red]")
        sys.exit(1)


async def delete_files(path_pattern: str, settings: dict):
    """Delete indexed files matching path pattern."""
    db_path = settings['metadata']['db_path']

    console.print("\n[bold cyan]Akashic Records - Delete Files[/bold cyan]\n")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Find matching files
        cursor.execute(
            "SELECT file_path FROM files WHERE file_path LIKE ?",
            (f"%{path_pattern}%",)
        )
        files = [row[0] for row in cursor.fetchall()]

        if not files:
            console.print(f"[yellow]No files match pattern: {path_pattern}[/yellow]")
            conn.close()
            return

        console.print(f"[yellow]Found {len(files)} files to delete:[/yellow]")
        for file_path in files[:10]:
            console.print(f"  - {file_path}")
        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more files")

        # Confirm deletion
        console.print()
        confirm = click.confirm("Delete these files from the index?", default=False)

        if not confirm:
            console.print("[yellow]Deletion cancelled[/yellow]")
            conn.close()
            return

        # Delete from database
        console.print("\n[yellow]Deleting files...[/yellow]")

        for file_path in files:
            # Delete chunks
            cursor.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            # Delete symbols
            cursor.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
            # Delete file
            cursor.execute("DELETE FROM files WHERE file_path = ?", (file_path,))

        conn.commit()
        conn.close()

        console.print(f"[green]Deleted {len(files)} files from index[/green]\n")

        # TODO: Also delete from Qdrant vector database
        # This requires implementing the CodeStore interface

    except sqlite3.Error as e:
        console.print(f"[red]Database error: {e}[/red]")
        sys.exit(1)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Akashic Records - Codebase Ingestion Tool"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option(
    '--path',
    '-p',
    multiple=True,
    required=True,
    help='Directory or file to index (can be specified multiple times)'
)
@click.option(
    '--recursive/--no-recursive',
    default=True,
    help='Scan subdirectories (default: True)'
)
@click.option(
    '--incremental',
    is_flag=True,
    help='Only index changed files (check file hash)'
)
@click.option(
    '--batch-size',
    default=None,
    type=int,
    help='Batch size for processing (default from settings)'
)
@click.option(
    '--workers',
    default=None,
    type=int,
    help='Number of parallel workers (default from settings)'
)
@click.option(
    '--settings',
    default=os.environ.get('AKASHIC_SETTINGS_PATH', 'config/settings.yaml'),
    help='Path to settings YAML file'
)
@click.option(
    '--skip-errors',
    is_flag=True,
    help='Continue processing on errors'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be indexed without doing it'
)
def ingest(
    path: Tuple[str],
    recursive: bool,
    incremental: bool,
    batch_size: Optional[int],
    workers: Optional[int],
    settings: str,
    skip_errors: bool,
    dry_run: bool
):
    """Index codebase files into the vector database."""

    os.environ['AKASHIC_SETTINGS_PATH'] = settings

    # Load settings
    config = load_settings(settings)

    # Override with CLI options
    if batch_size is None:
        batch_size = config['indexing'].get('batch_size', 100)
    if workers is None:
        workers = config['indexing'].get('parallel_workers', 4)

    # Run async ingestion
    asyncio.run(
        run_ingest(
            list(path),
            config,
            settings,
            recursive,
            incremental,
            batch_size,
            workers,
            skip_errors,
            dry_run
        )
    )


@cli.command()
@click.option(
    '--settings',
    default=os.environ.get('AKASHIC_SETTINGS_PATH', 'config/settings.yaml'),
    help='Path to settings YAML file'
)
def status(settings: str):
    """Show indexing statistics."""
    os.environ['AKASHIC_SETTINGS_PATH'] = settings
    config = load_settings(settings)
    show_status(config)


@cli.command()
@click.option(
    '--path',
    '-p',
    required=True,
    help='Path pattern to match for deletion'
)
@click.option(
    '--settings',
    default=os.environ.get('AKASHIC_SETTINGS_PATH', 'config/settings.yaml'),
    help='Path to settings YAML file'
)
def delete(path: str, settings: str):
    """Remove indexed files matching path pattern."""
    os.environ['AKASHIC_SETTINGS_PATH'] = settings
    config = load_settings(settings)
    asyncio.run(delete_files(path, config))


if __name__ == '__main__':
    cli()
