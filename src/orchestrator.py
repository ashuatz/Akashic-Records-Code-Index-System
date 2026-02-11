"""
Indexing Orchestrator - Central coordinator for codebase indexing

Features:
- Async worker pool with configurable parallelism
- Smart re-indexing based on file hashes
- Retry logic with exponential backoff
- Progress tracking and event emission
- Transaction-safe DB operations
"""

import asyncio
import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Set
from enum import Enum
from datetime import datetime
import fnmatch

from .code_store import CodeStore, CodeChunk as StoreCodeChunk
from .chunker import CodeChunker, CodeChunk


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class IndexingStatus(Enum):
    """Status of an indexing task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IndexingTask:
    """Represents a single file indexing task."""
    file_path: Path
    status: IndexingStatus = IndexingStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    chunk_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class IndexingStats:
    """Overall indexing statistics."""
    total_files: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    total_chunks: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "total_chunks": self.total_chunks,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None
        }


# ============================================================================
# File Hash Manager
# ============================================================================

class FileHashManager:
    """Manages file hashes for incremental indexing."""

    def __init__(self, db_conn: sqlite3.Connection):
        """
        Initialize file hash manager.

        Args:
            db_conn: SQLite database connection
        """
        self.db_conn = db_conn
        self._init_table()

    def _init_table(self):
        """Create file_hashes table if it doesn't exist."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_count INTEGER DEFAULT 0
            )
        """)
        self.db_conn.commit()

    def compute_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash
        """
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def get_stored_hash(self, file_path: Path) -> Optional[str]:
        """
        Get stored hash for a file.

        Args:
            file_path: Path to file

        Returns:
            Stored hash or None if not found
        """
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT hash FROM file_hashes WHERE file_path = ?",
            (str(file_path),)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def should_reindex(self, file_path: Path) -> bool:
        """
        Check if file needs re-indexing based on hash comparison.

        Args:
            file_path: Path to file

        Returns:
            True if file should be re-indexed
        """
        current_hash = self.compute_hash(file_path)
        if not current_hash:
            return True  # Couldn't compute hash, try indexing anyway

        stored_hash = self.get_stored_hash(file_path)
        return stored_hash != current_hash

    def update_hash(self, file_path: Path, chunk_count: int):
        """
        Update stored hash for a file.

        Args:
            file_path: Path to file
            chunk_count: Number of chunks indexed
        """
        current_hash = self.compute_hash(file_path)
        if not current_hash:
            return

        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO file_hashes (file_path, hash, last_indexed, chunk_count)
            VALUES (?, ?, ?, ?)
        """, (str(file_path), current_hash, datetime.now(), chunk_count))
        self.db_conn.commit()

    def remove_hash(self, file_path: Path):
        """
        Remove hash record for a file.

        Args:
            file_path: Path to file
        """
        cursor = self.db_conn.cursor()
        cursor.execute("DELETE FROM file_hashes WHERE file_path = ?", (str(file_path),))
        self.db_conn.commit()


# ============================================================================
# Indexing Orchestrator
# ============================================================================

class IndexingOrchestrator:
    """
    Central coordinator for indexing operations.

    Usage:
        orchestrator = IndexingOrchestrator(
            code_store=store,
            chunker=chunker,
            max_workers=8,
            max_retries=3
        )

        stats = await orchestrator.index_directory(
            path="/path/to/codebase",
            incremental=True
        )
    """

    def __init__(
        self,
        code_store: CodeStore,
        chunker: CodeChunker,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        on_progress: Optional[Callable[[IndexingStats], None]] = None,
        on_file_complete: Optional[Callable[[IndexingTask], None]] = None,
        on_error: Optional[Callable[[IndexingTask], None]] = None
    ):
        """
        Initialize the indexing orchestrator.

        Args:
            code_store: CodeStore instance for storage operations
            chunker: CodeChunker instance for code chunking
            max_workers: Maximum number of parallel workers
            max_retries: Maximum retry attempts for failed tasks
            retry_delay: Base delay for exponential backoff (seconds)
            on_progress: Callback for progress updates
            on_file_complete: Callback when a file completes
            on_error: Callback when a file fails
        """
        self.code_store = code_store
        self.chunker = chunker
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Callbacks
        self.on_progress = on_progress
        self.on_file_complete = on_file_complete
        self.on_error = on_error

        # File hash manager
        self.hash_manager: Optional[FileHashManager] = None

        # Cancellation support
        self._cancelled = False
        self._lock = asyncio.Lock()

    def _init_hash_manager(self):
        """Initialize file hash manager using code_store's DB connection."""
        if not self.code_store.db_conn:
            raise RuntimeError("CodeStore database connection not available")

        self.hash_manager = FileHashManager(self.code_store.db_conn)

    async def index_directory(
        self,
        path: str | Path,
        incremental: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> IndexingStats:
        """
        Index a directory with parallel workers.

        Args:
            path: Directory path to index
            incremental: Skip files that haven't changed (based on hash)
            include_patterns: File patterns to include (e.g., ["*.py", "*.cs"])
            exclude_patterns: Directory/file patterns to exclude

        Returns:
            IndexingStats with results
        """
        self._cancelled = False
        path = Path(path)

        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory path: {path}")

        logger.info(f"Starting indexing of directory: {path}")
        logger.info(f"Incremental mode: {incremental}, Workers: {self.max_workers}")

        # Initialize hash manager
        if not self.hash_manager:
            self._init_hash_manager()

        # Get default patterns from config if not specified
        if include_patterns is None:
            include_patterns = self.code_store.config.get('indexing', {}).get('include_extensions', ['*'])

        if exclude_patterns is None:
            exclude_patterns = self.code_store.config.get('indexing', {}).get('exclude_patterns', [])

        # Discover files
        files = self._discover_files(path, include_patterns, exclude_patterns)
        logger.info(f"Discovered {len(files)} files to process")

        # Create tasks
        tasks = [IndexingTask(file_path=f) for f in files]

        # Filter out files that don't need re-indexing
        if incremental:
            tasks = await self._filter_incremental(tasks)
            logger.info(f"After incremental filtering: {len(tasks)} files to index")

        if not tasks:
            logger.info("No files to index")
            return IndexingStats()

        # Initialize stats
        stats = IndexingStats(
            total_files=len(tasks),
            start_time=datetime.now()
        )

        # Create task queue
        queue = asyncio.Queue()
        for task in tasks:
            await queue.put(task)

        # Start workers
        workers = [
            asyncio.create_task(self._worker(queue, stats))
            for _ in range(self.max_workers)
        ]

        # Wait for completion
        await queue.join()

        # Cancel workers
        for worker in workers:
            worker.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

        # Finalize stats
        stats.end_time = datetime.now()

        logger.info(
            f"Indexing complete: {stats.processed} processed, "
            f"{stats.skipped} skipped, {stats.failed} failed, "
            f"{stats.total_chunks} total chunks"
        )

        return stats

    def _discover_files(
        self,
        path: Path,
        include_patterns: List[str],
        exclude_patterns: List[str]
    ) -> List[Path]:
        """
        Discover files to index based on patterns.

        Args:
            path: Root directory path
            include_patterns: Patterns to include
            exclude_patterns: Patterns to exclude

        Returns:
            List of file paths to index
        """
        files = []

        for file_path in path.rglob('*'):
            if not file_path.is_file():
                continue

            # Check if path matches exclude patterns
            relative_path = file_path.relative_to(path)
            if self._matches_patterns(str(relative_path), exclude_patterns):
                continue

            # Check if file matches include patterns
            if self._matches_patterns(file_path.name, include_patterns):
                files.append(file_path)

        return files

    def _matches_patterns(self, path: str, patterns: List[str]) -> bool:
        """
        Check if path matches any pattern.

        Args:
            path: Path to check
            patterns: List of glob patterns

        Returns:
            True if path matches any pattern
        """
        path_str = str(path).replace('\\', '/')
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in patterns)

    async def _filter_incremental(self, tasks: List[IndexingTask]) -> List[IndexingTask]:
        """
        Filter tasks for incremental indexing.

        Args:
            tasks: List of indexing tasks

        Returns:
            Filtered list of tasks that need indexing
        """
        filtered = []

        for task in tasks:
            if self.hash_manager.should_reindex(task.file_path):
                filtered.append(task)
            else:
                task.status = IndexingStatus.SKIPPED
                logger.debug(f"Skipping unchanged file: {task.file_path}")

        return filtered

    async def _worker(self, queue: asyncio.Queue, stats: IndexingStats):
        """
        Worker coroutine that processes tasks from queue.

        Args:
            queue: Task queue
            stats: Shared statistics object
        """
        while not self._cancelled:
            try:
                # Get task with timeout to allow checking cancellation
                task = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # Process task with retry logic
                task = await self._process_with_retry(task)

                # Update stats
                async with self._lock:
                    if task.status == IndexingStatus.COMPLETED:
                        stats.processed += 1
                        stats.total_chunks += task.chunk_count
                    elif task.status == IndexingStatus.SKIPPED:
                        stats.skipped += 1
                    elif task.status == IndexingStatus.FAILED:
                        stats.failed += 1

                # Call callbacks
                if task.status == IndexingStatus.COMPLETED and self.on_file_complete:
                    try:
                        self.on_file_complete(task)
                    except Exception as e:
                        logger.warning(f"File complete callback error: {e}")

                if task.status == IndexingStatus.FAILED and self.on_error:
                    try:
                        self.on_error(task)
                    except Exception as e:
                        logger.warning(f"Error callback error: {e}")

                if self.on_progress:
                    try:
                        self.on_progress(stats)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

            except Exception as e:
                logger.error(f"Worker error processing {task.file_path}: {e}")
            finally:
                queue.task_done()

    async def _process_with_retry(self, task: IndexingTask) -> IndexingTask:
        """
        Process task with exponential backoff retry.

        Args:
            task: Indexing task

        Returns:
            Updated task with results
        """
        task.started_at = datetime.now()

        for attempt in range(self.max_retries):
            if self._cancelled:
                task.status = IndexingStatus.FAILED
                task.error_message = "Cancelled"
                break

            try:
                task = await self.index_file(task)

                if task.status == IndexingStatus.COMPLETED:
                    break

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for {task.file_path}: {e}"
                )
                task.error_message = str(e)
                task.retry_count = attempt + 1

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        task.completed_at = datetime.now()

        if task.status != IndexingStatus.COMPLETED:
            task.status = IndexingStatus.FAILED
            logger.error(f"Failed to index {task.file_path} after {self.max_retries} attempts")

        return task

    async def index_file(self, task: IndexingTask) -> IndexingTask:
        """
        Index a single file.

        Args:
            task: Indexing task

        Returns:
            Updated task with results
        """
        try:
            task.status = IndexingStatus.IN_PROGRESS
            logger.debug(f"Indexing file: {task.file_path}")

            # Chunk the file
            chunks = self.chunker.chunk_file(task.file_path)

            if not chunks:
                logger.warning(f"No chunks generated for {task.file_path}")
                task.status = IndexingStatus.SKIPPED
                return task

            # Convert to store format
            store_chunks = [
                StoreCodeChunk(
                    file_path=chunk.file_path,
                    code=chunk.code,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language
                )
                for chunk in chunks
            ]

            # Delete old chunks for this file first
            try:
                await self.code_store.delete_file_chunks(str(task.file_path))
            except Exception as e:
                logger.warning(f"Failed to delete old chunks for {task.file_path}: {e}")

            # Add new chunks
            await self.code_store.add_chunks(store_chunks)

            # Update file hash
            self.hash_manager.update_hash(task.file_path, len(chunks))

            task.chunk_count = len(chunks)
            task.status = IndexingStatus.COMPLETED
            logger.debug(f"Successfully indexed {task.file_path}: {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to index file {task.file_path}: {e}")
            task.error_message = str(e)
            task.status = IndexingStatus.FAILED
            raise

        return task

    async def cancel(self):
        """Cancel ongoing indexing operation."""
        logger.info("Cancelling indexing operation")
        self._cancelled = True

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress statistics.

        Returns:
            Dictionary with progress information
        """
        return {
            "cancelled": self._cancelled,
            "max_workers": self.max_workers
        }


# ============================================================================
# Convenience Function
# ============================================================================

async def index_codebase(
    directory: str | Path,
    settings_path: Optional[str] = None,
    incremental: bool = True,
    max_workers: Optional[int] = None,
    on_progress: Optional[Callable[[IndexingStats], None]] = None
) -> IndexingStats:
    """
    Convenience function to index a codebase.

    Args:
        directory: Directory to index
        settings_path: Path to settings.yaml (default: config/settings.yaml)
        incremental: Use incremental indexing
        max_workers: Number of parallel workers (default: from config)
        on_progress: Progress callback

    Returns:
        IndexingStats with results

    Example:
        stats = await index_codebase(
            directory="/path/to/unity/source",
            incremental=True,
            max_workers=8
        )
        print(f"Indexed {stats.processed} files, {stats.total_chunks} chunks")
    """
    # Initialize components
    store = CodeStore(settings_path)
    await store.connect()

    try:
        # Get worker count from config if not specified
        if max_workers is None:
            max_workers = store.config.get('indexing', {}).get('parallel_workers', 4)

        # Create chunker
        chunk_config = store.config.get('indexing', {}).get('chunk', {})
        chunker = CodeChunker(
            max_tokens=chunk_config.get('max_tokens', 4000),
            overlap=chunk_config.get('overlap_tokens', 200)
        )

        # Create orchestrator
        orchestrator = IndexingOrchestrator(
            code_store=store,
            chunker=chunker,
            max_workers=max_workers,
            on_progress=on_progress
        )

        # Index directory
        stats = await orchestrator.index_directory(
            path=directory,
            incremental=incremental
        )

        return stats

    finally:
        await store.close()


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python orchestrator.py <directory>")
            sys.exit(1)

        directory = sys.argv[1]

        def progress_callback(stats: IndexingStats):
            print(f"\rProgress: {stats.processed}/{stats.total_files} files, "
                  f"{stats.total_chunks} chunks", end='', flush=True)

        try:
            stats = await index_codebase(
                directory=directory,
                incremental=True,
                on_progress=progress_callback
            )

            print("\n\nIndexing complete!")
            print(f"  Total files: {stats.total_files}")
            print(f"  Processed: {stats.processed}")
            print(f"  Skipped: {stats.skipped}")
            print(f"  Failed: {stats.failed}")
            print(f"  Total chunks: {stats.total_chunks}")

            if stats.start_time and stats.end_time:
                duration = (stats.end_time - stats.start_time).total_seconds()
                print(f"  Duration: {duration:.2f} seconds")
                if stats.processed > 0:
                    print(f"  Rate: {stats.processed / duration:.2f} files/sec")

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            sys.exit(1)

    asyncio.run(main())
