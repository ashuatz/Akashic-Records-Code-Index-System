"""
File System Watcher - Real-time monitoring for incremental indexing

Uses watchdog library to monitor file system events and triggers
re-indexing through the orchestrator.

Features:
- Debouncing for rapid changes (IDE auto-save)
- Batch processing for efficiency
- Pattern-based filtering (include/exclude)
- Async integration with watchdog's sync API
- Rename handling as delete + create
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Set, List, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from fnmatch import fnmatch

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileSystemEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    DirCreatedEvent,
    DirModifiedEvent,
    DirDeletedEvent,
    DirMovedEvent
)

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of file system change."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileChange:
    """Represents a file system change event."""
    path: Path
    change_type: ChangeType
    timestamp: datetime
    old_path: Optional[Path] = None  # For renames

    def __hash__(self):
        return hash((str(self.path), self.change_type, str(self.old_path)))

    def __eq__(self, other):
        if not isinstance(other, FileChange):
            return False
        return (self.path == other.path and
                self.change_type == other.change_type and
                self.old_path == other.old_path)


class DebouncedFileWatcher:
    """
    File system watcher with debouncing and batch processing.

    Monitors specified directories for file changes and triggers callbacks
    with batched changes. Debounces rapid changes to the same file.

    Usage:
        watcher = DebouncedFileWatcher(
            paths=["/path/to/codebase"],
            include_patterns=["*.cpp", "*.cs", "*.py"],
            exclude_patterns=["**/node_modules/**", "**/.git/**"],
            debounce_seconds=2.0,
            on_changes=handle_changes
        )

        await watcher.start()
        # ... later
        await watcher.stop()
    """

    def __init__(
        self,
        paths: List[str | Path],
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        debounce_seconds: float = 2.0,
        batch_delay_seconds: float = 5.0,
        on_changes: Optional[Callable[[List[FileChange]], None]] = None
    ):
        """
        Initialize the file watcher.

        Args:
            paths: List of directories to watch
            include_patterns: Glob patterns for files to include (e.g., ["*.cpp", "*.py"])
            exclude_patterns: Glob patterns for files/dirs to exclude (e.g., ["**/node_modules/**"])
            debounce_seconds: Time to wait before considering a file change stable
            batch_delay_seconds: Time to accumulate changes before processing batch
            on_changes: Callback function that receives List[FileChange]
        """
        self.paths = [Path(p).resolve() for p in paths]
        self.include_patterns = include_patterns or [
            "*.cpp", "*.h", "*.hpp", "*.cc", "*.cxx",
            "*.cs", "*.py", "*.js", "*.ts", "*.java",
            "*.go", "*.rs", "*.rb", "*.php"
        ]
        self.exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/.git/**",
            "**/Build/**",
            "**/build/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/*.pyc",
            "**/.DS_Store",
            "**/Thumbs.db"
        ]
        self.debounce_seconds = debounce_seconds
        self.batch_delay_seconds = batch_delay_seconds
        self.on_changes = on_changes

        self._observer: Optional[Observer] = None
        self._pending_changes: Dict[Path, FileChange] = {}
        self._debounce_tasks: Dict[Path, asyncio.Task] = {}
        self._batch_task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()

        logger.info(f"Initialized file watcher for {len(self.paths)} paths")
        logger.debug(f"Include patterns: {self.include_patterns}")
        logger.debug(f"Exclude patterns: {self.exclude_patterns}")

    async def start(self):
        """Start watching for file changes."""
        if self._running:
            logger.warning("File watcher already running")
            return

        self._loop = asyncio.get_running_loop()
        self._running = True

        # Create observer and event handler
        self._observer = Observer()
        handler = CodebaseWatcherHandler(
            callback=self._handle_event,
            loop=self._loop
        )

        # Schedule observers for each path
        for path in self.paths:
            if not path.exists():
                logger.error(f"Watch path does not exist: {path}")
                continue

            if not path.is_dir():
                logger.error(f"Watch path is not a directory: {path}")
                continue

            self._observer.schedule(handler, str(path), recursive=True)
            logger.info(f"Started watching: {path}")

        # Start observer
        self._observer.start()

        # Start batch processing task
        self._batch_task = asyncio.create_task(self._batch_processor())

        logger.info("File watcher started")

    async def stop(self):
        """Stop watching and cleanup."""
        if not self._running:
            return

        logger.info("Stopping file watcher...")
        self._running = False

        # Stop observer
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        # Cancel pending debounce tasks
        for task in self._debounce_tasks.values():
            if not task.done():
                task.cancel()
        self._debounce_tasks.clear()

        # Cancel batch task
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Process any remaining changes
        if self._pending_changes and self.on_changes:
            changes = list(self._pending_changes.values())
            self._pending_changes.clear()
            logger.info(f"Processing {len(changes)} remaining changes before shutdown")
            try:
                if asyncio.iscoroutinefunction(self.on_changes):
                    await self.on_changes(changes)
                else:
                    self.on_changes(changes)
            except Exception as e:
                logger.error(f"Error processing final changes: {e}", exc_info=True)

        logger.info("File watcher stopped")

    def _matches_patterns(self, path: Path) -> bool:
        """
        Check if path matches include patterns and not exclude patterns.

        Args:
            path: Path to check

        Returns:
            True if path should be monitored
        """
        path_str = str(path)
        rel_path = None

        # Get relative path from one of the watch roots
        for watch_path in self.paths:
            try:
                rel_path = path.relative_to(watch_path)
                break
            except ValueError:
                continue

        if rel_path is None:
            # Path not under any watch root
            return False

        rel_path_str = str(rel_path).replace('\\', '/')

        # Check exclude patterns first (more efficient)
        for pattern in self.exclude_patterns:
            if fnmatch(rel_path_str, pattern) or fnmatch(path_str, pattern):
                return False

        # Check include patterns
        filename = path.name
        for pattern in self.include_patterns:
            if fnmatch(filename, pattern):
                return True

        return False

    async def _handle_event(self, event: FileSystemEvent):
        """
        Handle a file system event with debouncing.

        Args:
            event: Watchdog file system event
        """
        # Ignore directory events
        if event.is_directory:
            return

        path = Path(event.src_path).resolve()

        # Filter by patterns
        if not self._matches_patterns(path):
            return

        # Create FileChange
        change: Optional[FileChange] = None

        if isinstance(event, FileCreatedEvent):
            change = FileChange(
                path=path,
                change_type=ChangeType.CREATED,
                timestamp=datetime.now()
            )
            logger.debug(f"File created: {path}")

        elif isinstance(event, FileModifiedEvent):
            change = FileChange(
                path=path,
                change_type=ChangeType.MODIFIED,
                timestamp=datetime.now()
            )
            logger.debug(f"File modified: {path}")

        elif isinstance(event, FileDeletedEvent):
            change = FileChange(
                path=path,
                change_type=ChangeType.DELETED,
                timestamp=datetime.now()
            )
            logger.debug(f"File deleted: {path}")

        elif isinstance(event, FileMovedEvent):
            # Handle rename as delete old + create new
            old_path = Path(event.src_path).resolve()
            new_path = Path(event.dest_path).resolve()

            # Check if both paths match patterns
            old_matches = self._matches_patterns(old_path)
            new_matches = self._matches_patterns(new_path)

            if old_matches and new_matches:
                # Rename within watched files
                change = FileChange(
                    path=new_path,
                    change_type=ChangeType.RENAMED,
                    timestamp=datetime.now(),
                    old_path=old_path
                )
                logger.debug(f"File renamed: {old_path} -> {new_path}")
            elif old_matches:
                # Moved out of watched scope
                change = FileChange(
                    path=old_path,
                    change_type=ChangeType.DELETED,
                    timestamp=datetime.now()
                )
                logger.debug(f"File moved out: {old_path}")
            elif new_matches:
                # Moved into watched scope
                change = FileChange(
                    path=new_path,
                    change_type=ChangeType.CREATED,
                    timestamp=datetime.now()
                )
                logger.debug(f"File moved in: {new_path}")

        if change:
            await self._debounce(path, change)

    async def _debounce(self, path: Path, change: FileChange):
        """
        Debounce rapid changes to the same file.

        Cancels previous debounce task for the same file and starts a new one.

        Args:
            path: File path
            change: File change event
        """
        async with self._lock:
            # Cancel existing debounce task for this path
            if path in self._debounce_tasks:
                task = self._debounce_tasks[path]
                if not task.done():
                    task.cancel()

            # Create new debounce task
            task = asyncio.create_task(self._debounce_delay(path, change))
            self._debounce_tasks[path] = task

    async def _debounce_delay(self, path: Path, change: FileChange):
        """
        Wait for debounce period then add change to pending.

        Args:
            path: File path
            change: File change event
        """
        try:
            await asyncio.sleep(self.debounce_seconds)

            async with self._lock:
                # Merge with existing change if present
                if path in self._pending_changes:
                    existing = self._pending_changes[path]

                    # Delete always takes precedence
                    if change.change_type == ChangeType.DELETED:
                        self._pending_changes[path] = change
                    # Create + modify = create
                    elif (existing.change_type == ChangeType.CREATED and
                          change.change_type == ChangeType.MODIFIED):
                        # Keep existing create, update timestamp
                        existing.timestamp = change.timestamp
                    # Any other case, use latest change
                    else:
                        self._pending_changes[path] = change
                else:
                    self._pending_changes[path] = change

                # Clean up task reference
                if path in self._debounce_tasks:
                    del self._debounce_tasks[path]

        except asyncio.CancelledError:
            # Normal cancellation, ignore
            pass
        except Exception as e:
            logger.error(f"Error in debounce delay for {path}: {e}", exc_info=True)

    async def _batch_processor(self):
        """
        Periodically process accumulated changes as batches.

        Runs continuously while watcher is running.
        """
        while self._running:
            try:
                await asyncio.sleep(self.batch_delay_seconds)

                async with self._lock:
                    if not self._pending_changes:
                        continue

                    # Get all pending changes
                    changes = list(self._pending_changes.values())
                    self._pending_changes.clear()

                if changes:
                    logger.info(f"Processing batch of {len(changes)} file changes")
                    await self._process_batch(changes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)

    async def _process_batch(self, changes: List[FileChange]):
        """
        Process accumulated changes as a batch.

        Args:
            changes: List of file changes to process
        """
        if not self.on_changes:
            logger.warning("No on_changes callback registered")
            return

        try:
            # Call the callback
            if asyncio.iscoroutinefunction(self.on_changes):
                await self.on_changes(changes)
            else:
                # Run sync callback in executor
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    self.on_changes,
                    changes
                )

            logger.info(f"Successfully processed batch of {len(changes)} changes")

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)


class CodebaseWatcherHandler(FileSystemEventHandler):
    """
    Watchdog event handler that bridges to async processing.

    Converts synchronous watchdog events to async callbacks.
    """

    def __init__(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        """
        Initialize handler.

        Args:
            callback: Async callback to handle events
            loop: Event loop to schedule callbacks on
        """
        super().__init__()
        self.callback = callback
        self.loop = loop

    def _schedule_async(self, event: FileSystemEvent):
        """
        Schedule async callback for event.

        Args:
            event: File system event
        """
        if self.loop.is_closed():
            return

        asyncio.run_coroutine_threadsafe(
            self.callback(event),
            self.loop
        )

    def on_created(self, event: FileCreatedEvent):
        """Handle file/directory creation."""
        self._schedule_async(event)

    def on_modified(self, event: FileModifiedEvent):
        """Handle file/directory modification."""
        self._schedule_async(event)

    def on_deleted(self, event: FileDeletedEvent):
        """Handle file/directory deletion."""
        self._schedule_async(event)

    def on_moved(self, event: FileMovedEvent):
        """Handle file/directory move/rename."""
        self._schedule_async(event)


# Example usage
async def main():
    """Example usage of DebouncedFileWatcher."""
    async def handle_changes(changes: List[FileChange]):
        """Handle file changes."""
        print(f"\nReceived {len(changes)} changes:")
        for change in changes:
            if change.change_type == ChangeType.RENAMED:
                print(f"  RENAMED: {change.old_path} -> {change.path}")
            else:
                print(f"  {change.change_type.value.upper()}: {change.path}")

    watcher = DebouncedFileWatcher(
        paths=["./test_dir"],
        include_patterns=["*.py", "*.txt"],
        exclude_patterns=["**/__pycache__/**"],
        debounce_seconds=1.0,
        batch_delay_seconds=3.0,
        on_changes=handle_changes
    )

    await watcher.start()

    try:
        print("Watching for changes... Press Ctrl+C to stop")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await watcher.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
