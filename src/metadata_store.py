"""
Metadata Store - Database abstraction layer for metadata operations

Supports SQLite (development/single-user) and PostgreSQL (production/scaling).
Provides transaction-safe operations for symbols, references, and dependencies.

Design Goals:
- Abstract database operations for easy SQLite → PostgreSQL migration
- Transaction safety for batch operations
- Efficient queries for symbols, references, and dependencies
- Connection pooling support for production scaling
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from enum import Enum
from datetime import datetime

import aiosqlite

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class DatabaseBackend(Enum):
    """Supported database backends."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


@dataclass
class SymbolRecord:
    """Represents a symbol (function, class, method, field) in the codebase."""
    name: str
    kind: str  # function, class, method, field, variable, interface, etc.
    file_path: str
    line_start: int
    line_end: int
    id: Optional[int] = None
    column_start: Optional[int] = None
    signature: Optional[str] = None
    parent_id: Optional[int] = None
    language: Optional[str] = None
    docstring: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ReferenceRecord:
    """Represents a reference to a symbol (call, read, write, etc.)."""
    symbol_id: int
    file_path: str
    line: int
    column: int
    id: Optional[int] = None
    kind: Optional[str] = None  # call, read, write, definition, import

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class DependencyRecord:
    """Represents a dependency relationship between symbols."""
    from_symbol_id: int
    to_symbol_id: int
    id: Optional[int] = None
    kind: Optional[str] = None  # inherits, calls, uses, imports, implements

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class FileHashRecord:
    """Tracks file content hashes to detect changes."""
    file_path: str
    hash: str
    indexed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ============================================================================
# Abstract Base Class
# ============================================================================

class MetadataStore(ABC):
    """Abstract base class for metadata storage operations."""

    @abstractmethod
    async def connect(self):
        """Initialize database connection."""
        pass

    @abstractmethod
    async def close(self):
        """Close database connection and cleanup."""
        pass

    @abstractmethod
    @asynccontextmanager
    async def transaction(self):
        """Async context manager for database transactions.

        Usage:
            async with store.transaction():
                await store.insert_symbol(...)
                await store.insert_reference(...)
        """
        pass

    # ========================================================================
    # Symbol Operations
    # ========================================================================

    @abstractmethod
    async def insert_symbol(self, symbol: SymbolRecord) -> int:
        """Insert a symbol and return its ID."""
        pass

    @abstractmethod
    async def get_symbol_by_id(self, symbol_id: int) -> Optional[SymbolRecord]:
        """Get symbol by ID."""
        pass

    @abstractmethod
    async def get_symbol_by_name(
        self,
        name: str,
        kind: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> Optional[SymbolRecord]:
        """Get symbol by name with optional filters."""
        pass

    @abstractmethod
    async def get_symbols_in_file(self, file_path: str) -> List[SymbolRecord]:
        """Get all symbols defined in a file."""
        pass

    @abstractmethod
    async def search_symbols(
        self,
        query: str,
        kind: Optional[str] = None,
        limit: int = 50
    ) -> List[SymbolRecord]:
        """Search symbols by name pattern (case-insensitive)."""
        pass

    @abstractmethod
    async def delete_symbols_for_file(self, file_path: str) -> int:
        """Delete all symbols for a file. Returns count deleted."""
        pass

    # ========================================================================
    # Reference Operations
    # ========================================================================

    @abstractmethod
    async def insert_reference(self, ref: ReferenceRecord) -> int:
        """Insert a reference and return its ID."""
        pass

    @abstractmethod
    async def get_references_for_symbol(
        self,
        symbol_id: int,
        kind: Optional[str] = None
    ) -> List[ReferenceRecord]:
        """Get all references to a symbol."""
        pass

    @abstractmethod
    async def delete_references_for_file(self, file_path: str) -> int:
        """Delete all references in a file. Returns count deleted."""
        pass

    # ========================================================================
    # Dependency Operations
    # ========================================================================

    @abstractmethod
    async def insert_dependency(self, dep: DependencyRecord) -> int:
        """Insert a dependency and return its ID."""
        pass

    @abstractmethod
    async def get_dependencies(
        self,
        symbol_id: int,
        direction: str = "outgoing",
        kind: Optional[str] = None
    ) -> List[DependencyRecord]:
        """Get dependencies for a symbol.

        Args:
            symbol_id: Symbol ID
            direction: "outgoing" (from this symbol) or "incoming" (to this symbol)
            kind: Filter by dependency kind (optional)
        """
        pass

    @abstractmethod
    async def delete_dependencies_for_file(self, file_path: str) -> int:
        """Delete dependencies involving symbols in a file. Returns count deleted."""
        pass

    # ========================================================================
    # File Hash Operations
    # ========================================================================

    @abstractmethod
    async def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get stored hash for a file."""
        pass

    @abstractmethod
    async def set_file_hash(self, file_path: str, hash_value: str):
        """Store or update hash for a file."""
        pass

    @abstractmethod
    async def delete_file_hash(self, file_path: str):
        """Delete hash record for a file."""
        pass

    @abstractmethod
    async def get_indexed_files(self) -> List[FileHashRecord]:
        """Get all indexed file records."""
        pass

    # ========================================================================
    # Bulk Operations
    # ========================================================================

    @abstractmethod
    async def bulk_insert_symbols(self, symbols: List[SymbolRecord]) -> List[int]:
        """Bulk insert symbols. Returns list of IDs."""
        pass

    @abstractmethod
    async def bulk_insert_references(self, refs: List[ReferenceRecord]) -> List[int]:
        """Bulk insert references. Returns list of IDs."""
        pass

    @abstractmethod
    async def bulk_insert_dependencies(self, deps: List[DependencyRecord]) -> List[int]:
        """Bulk insert dependencies. Returns list of IDs."""
        pass


# ============================================================================
# SQLite Implementation
# ============================================================================

class SQLiteMetadataStore(MetadataStore):
    """SQLite implementation of MetadataStore for single-user/development use."""

    def __init__(self, db_path: str | Path):
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._connection: Optional[aiosqlite.Connection] = None
        self._in_transaction = False

    async def connect(self):
        """Initialize database connection and create schema."""
        if self._connection:
            logger.warning("SQLiteMetadataStore already connected")
            return

        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with WAL mode for better concurrency
        self._connection = await aiosqlite.connect(
            str(self.db_path),
            isolation_level=None  # Autocommit mode (we'll manage transactions explicitly)
        )

        # Enable foreign keys and WAL mode
        await self._connection.execute("PRAGMA foreign_keys = ON")
        await self._connection.execute("PRAGMA journal_mode = WAL")

        # Set row factory for named access
        self._connection.row_factory = aiosqlite.Row

        # Create schema
        await self._create_schema()

        logger.info(f"SQLite metadata store connected: {self.db_path}")

    async def _create_schema(self):
        """Create database schema."""

        # Symbols table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                column_start INTEGER,
                signature TEXT,
                parent_id INTEGER,
                language TEXT,
                docstring TEXT,
                FOREIGN KEY (parent_id) REFERENCES symbols(id) ON DELETE CASCADE
            )
        """)

        # Indexes for symbols
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_name
            ON symbols(name)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_file
            ON symbols(file_path)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_kind
            ON symbols(kind)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_parent
            ON symbols(parent_id)
        """)

        # References table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                column INTEGER NOT NULL,
                kind TEXT,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
            )
        """)

        # Indexes for references
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_references_symbol
            ON references(symbol_id)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_references_file
            ON references(file_path)
        """)

        # Dependencies table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_symbol_id INTEGER NOT NULL,
                to_symbol_id INTEGER NOT NULL,
                kind TEXT,
                FOREIGN KEY (from_symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
                FOREIGN KEY (to_symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
            )
        """)

        # Indexes for dependencies
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_dependencies_from
            ON dependencies(from_symbol_id)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_dependencies_to
            ON dependencies(to_symbol_id)
        """)

        # File hashes table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
        """)

        await self._connection.commit()
        logger.debug("SQLite schema created/verified")

    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("SQLite metadata store closed")

    @asynccontextmanager
    async def transaction(self):
        """Async context manager for database transactions."""
        if not self._connection:
            raise RuntimeError("Database not connected")

        if self._in_transaction:
            # Nested transaction - just yield (SQLite doesn't support true nested transactions)
            yield
            return

        self._in_transaction = True
        try:
            await self._connection.execute("BEGIN")
            yield
            await self._connection.commit()
        except Exception as e:
            await self._connection.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            self._in_transaction = False

    # ========================================================================
    # Symbol Operations
    # ========================================================================

    async def insert_symbol(self, symbol: SymbolRecord) -> int:
        """Insert a symbol and return its ID."""
        cursor = await self._connection.execute("""
            INSERT INTO symbols
            (name, kind, file_path, line_start, line_end, column_start, signature, parent_id, language, docstring)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol.name,
            symbol.kind,
            symbol.file_path,
            symbol.line_start,
            symbol.line_end,
            symbol.column_start,
            symbol.signature,
            symbol.parent_id,
            symbol.language,
            symbol.docstring
        ))

        if not self._in_transaction:
            await self._connection.commit()

        return cursor.lastrowid

    async def get_symbol_by_id(self, symbol_id: int) -> Optional[SymbolRecord]:
        """Get symbol by ID."""
        cursor = await self._connection.execute("""
            SELECT * FROM symbols WHERE id = ?
        """, (symbol_id,))

        row = await cursor.fetchone()
        if row:
            return self._row_to_symbol(row)
        return None

    async def get_symbol_by_name(
        self,
        name: str,
        kind: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> Optional[SymbolRecord]:
        """Get symbol by name with optional filters."""
        query = "SELECT * FROM symbols WHERE name = ?"
        params = [name]

        if kind:
            query += " AND kind = ?"
            params.append(kind)

        if file_path:
            query += " AND file_path = ?"
            params.append(file_path)

        query += " LIMIT 1"

        cursor = await self._connection.execute(query, params)
        row = await cursor.fetchone()

        if row:
            return self._row_to_symbol(row)
        return None

    async def get_symbols_in_file(self, file_path: str) -> List[SymbolRecord]:
        """Get all symbols defined in a file."""
        cursor = await self._connection.execute("""
            SELECT * FROM symbols WHERE file_path = ?
            ORDER BY line_start
        """, (file_path,))

        rows = await cursor.fetchall()
        return [self._row_to_symbol(row) for row in rows]

    async def search_symbols(
        self,
        query: str,
        kind: Optional[str] = None,
        limit: int = 50
    ) -> List[SymbolRecord]:
        """Search symbols by name pattern (case-insensitive)."""
        sql = "SELECT * FROM symbols WHERE name LIKE ? COLLATE NOCASE"
        params = [f"%{query}%"]

        if kind:
            sql += " AND kind = ?"
            params.append(kind)

        sql += f" ORDER BY name LIMIT {limit}"

        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_symbol(row) for row in rows]

    async def delete_symbols_for_file(self, file_path: str) -> int:
        """Delete all symbols for a file. Returns count deleted."""
        cursor = await self._connection.execute("""
            DELETE FROM symbols WHERE file_path = ?
        """, (file_path,))

        if not self._in_transaction:
            await self._connection.commit()

        return cursor.rowcount

    # ========================================================================
    # Reference Operations
    # ========================================================================

    async def insert_reference(self, ref: ReferenceRecord) -> int:
        """Insert a reference and return its ID."""
        cursor = await self._connection.execute("""
            INSERT INTO references (symbol_id, file_path, line, column, kind)
            VALUES (?, ?, ?, ?, ?)
        """, (ref.symbol_id, ref.file_path, ref.line, ref.column, ref.kind))

        if not self._in_transaction:
            await self._connection.commit()

        return cursor.lastrowid

    async def get_references_for_symbol(
        self,
        symbol_id: int,
        kind: Optional[str] = None
    ) -> List[ReferenceRecord]:
        """Get all references to a symbol."""
        if kind:
            cursor = await self._connection.execute("""
                SELECT * FROM references
                WHERE symbol_id = ? AND kind = ?
                ORDER BY file_path, line
            """, (symbol_id, kind))
        else:
            cursor = await self._connection.execute("""
                SELECT * FROM references
                WHERE symbol_id = ?
                ORDER BY file_path, line
            """, (symbol_id,))

        rows = await cursor.fetchall()
        return [self._row_to_reference(row) for row in rows]

    async def delete_references_for_file(self, file_path: str) -> int:
        """Delete all references in a file. Returns count deleted."""
        cursor = await self._connection.execute("""
            DELETE FROM references WHERE file_path = ?
        """, (file_path,))

        if not self._in_transaction:
            await self._connection.commit()

        return cursor.rowcount

    # ========================================================================
    # Dependency Operations
    # ========================================================================

    async def insert_dependency(self, dep: DependencyRecord) -> int:
        """Insert a dependency and return its ID."""
        cursor = await self._connection.execute("""
            INSERT INTO dependencies (from_symbol_id, to_symbol_id, kind)
            VALUES (?, ?, ?)
        """, (dep.from_symbol_id, dep.to_symbol_id, dep.kind))

        if not self._in_transaction:
            await self._connection.commit()

        return cursor.lastrowid

    async def get_dependencies(
        self,
        symbol_id: int,
        direction: str = "outgoing",
        kind: Optional[str] = None
    ) -> List[DependencyRecord]:
        """Get dependencies for a symbol."""
        if direction == "outgoing":
            query = "SELECT * FROM dependencies WHERE from_symbol_id = ?"
        elif direction == "incoming":
            query = "SELECT * FROM dependencies WHERE to_symbol_id = ?"
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'outgoing' or 'incoming'")

        params = [symbol_id]

        if kind:
            query += " AND kind = ?"
            params.append(kind)

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_dependency(row) for row in rows]

    async def delete_dependencies_for_file(self, file_path: str) -> int:
        """Delete dependencies involving symbols in a file. Returns count deleted."""
        # First get symbol IDs in this file
        cursor = await self._connection.execute("""
            SELECT id FROM symbols WHERE file_path = ?
        """, (file_path,))

        symbol_ids = [row[0] for row in await cursor.fetchall()]

        if not symbol_ids:
            return 0

        # Delete dependencies where either side is in this file
        placeholders = ','.join('?' * len(symbol_ids))
        cursor = await self._connection.execute(f"""
            DELETE FROM dependencies
            WHERE from_symbol_id IN ({placeholders})
               OR to_symbol_id IN ({placeholders})
        """, symbol_ids + symbol_ids)

        if not self._in_transaction:
            await self._connection.commit()

        return cursor.rowcount

    # ========================================================================
    # File Hash Operations
    # ========================================================================

    async def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get stored hash for a file."""
        cursor = await self._connection.execute("""
            SELECT hash FROM file_hashes WHERE file_path = ?
        """, (file_path,))

        row = await cursor.fetchone()
        if row:
            return row[0]
        return None

    async def set_file_hash(self, file_path: str, hash_value: str):
        """Store or update hash for a file."""
        indexed_at = datetime.utcnow().isoformat()

        await self._connection.execute("""
            INSERT OR REPLACE INTO file_hashes (file_path, hash, indexed_at)
            VALUES (?, ?, ?)
        """, (file_path, hash_value, indexed_at))

        if not self._in_transaction:
            await self._connection.commit()

    async def delete_file_hash(self, file_path: str):
        """Delete hash record for a file."""
        await self._connection.execute("""
            DELETE FROM file_hashes WHERE file_path = ?
        """, (file_path,))

        if not self._in_transaction:
            await self._connection.commit()

    async def get_indexed_files(self) -> List[FileHashRecord]:
        """Get all indexed file records."""
        cursor = await self._connection.execute("""
            SELECT file_path, hash, indexed_at FROM file_hashes
            ORDER BY file_path
        """)

        rows = await cursor.fetchall()
        return [
            FileHashRecord(
                file_path=row[0],
                hash=row[1],
                indexed_at=row[2]
            )
            for row in rows
        ]

    # ========================================================================
    # Bulk Operations
    # ========================================================================

    async def bulk_insert_symbols(self, symbols: List[SymbolRecord]) -> List[int]:
        """Bulk insert symbols. Returns list of IDs."""
        if not symbols:
            return []

        ids = []
        for symbol in symbols:
            cursor = await self._connection.execute("""
                INSERT INTO symbols
                (name, kind, file_path, line_start, line_end, column_start, signature, parent_id, language, docstring)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol.name,
                symbol.kind,
                symbol.file_path,
                symbol.line_start,
                symbol.line_end,
                symbol.column_start,
                symbol.signature,
                symbol.parent_id,
                symbol.language,
                symbol.docstring
            ))
            ids.append(cursor.lastrowid)

        if not self._in_transaction:
            await self._connection.commit()

        return ids

    async def bulk_insert_references(self, refs: List[ReferenceRecord]) -> List[int]:
        """Bulk insert references. Returns list of IDs."""
        if not refs:
            return []

        ids = []
        for ref in refs:
            cursor = await self._connection.execute("""
                INSERT INTO references (symbol_id, file_path, line, column, kind)
                VALUES (?, ?, ?, ?, ?)
            """, (ref.symbol_id, ref.file_path, ref.line, ref.column, ref.kind))
            ids.append(cursor.lastrowid)

        if not self._in_transaction:
            await self._connection.commit()

        return ids

    async def bulk_insert_dependencies(self, deps: List[DependencyRecord]) -> List[int]:
        """Bulk insert dependencies. Returns list of IDs."""
        if not deps:
            return []

        ids = []
        for dep in deps:
            cursor = await self._connection.execute("""
                INSERT INTO dependencies (from_symbol_id, to_symbol_id, kind)
                VALUES (?, ?, ?)
            """, (dep.from_symbol_id, dep.to_symbol_id, dep.kind))
            ids.append(cursor.lastrowid)

        if not self._in_transaction:
            await self._connection.commit()

        return ids

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _row_to_symbol(self, row: aiosqlite.Row) -> SymbolRecord:
        """Convert database row to SymbolRecord."""
        return SymbolRecord(
            id=row['id'],
            name=row['name'],
            kind=row['kind'],
            file_path=row['file_path'],
            line_start=row['line_start'],
            line_end=row['line_end'],
            column_start=row['column_start'],
            signature=row['signature'],
            parent_id=row['parent_id'],
            language=row['language'],
            docstring=row['docstring']
        )

    def _row_to_reference(self, row: aiosqlite.Row) -> ReferenceRecord:
        """Convert database row to ReferenceRecord."""
        return ReferenceRecord(
            id=row['id'],
            symbol_id=row['symbol_id'],
            file_path=row['file_path'],
            line=row['line'],
            column=row['column'],
            kind=row['kind']
        )

    def _row_to_dependency(self, row: aiosqlite.Row) -> DependencyRecord:
        """Convert database row to DependencyRecord."""
        return DependencyRecord(
            id=row['id'],
            from_symbol_id=row['from_symbol_id'],
            to_symbol_id=row['to_symbol_id'],
            kind=row['kind']
        )


# ============================================================================
# PostgreSQL Implementation (Stub for Future)
# ============================================================================

class PostgreSQLMetadataStore(MetadataStore):
    """PostgreSQL implementation of MetadataStore for production scaling.

    NOTE: This is a stub implementation. Full implementation requires asyncpg.
    When needed, install: pip install asyncpg
    """

    def __init__(self, connection_string: str, pool_size: int = 10):
        """Initialize PostgreSQL store.

        Args:
            connection_string: PostgreSQL connection string
                              (e.g., "postgresql://user:pass@host:5432/dbname")
            pool_size: Connection pool size
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self._pool = None

    async def connect(self):
        """Initialize database connection pool."""
        raise NotImplementedError(
            "PostgreSQL support not yet implemented. "
            "Install asyncpg and implement this class when scaling is needed."
        )

    async def close(self):
        """Close connection pool."""
        raise NotImplementedError("PostgreSQL support not yet implemented")

    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager."""
        raise NotImplementedError("PostgreSQL support not yet implemented")
        yield  # Unreachable, but needed for type checking

    # All other methods raise NotImplementedError
    async def insert_symbol(self, symbol: SymbolRecord) -> int:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def get_symbol_by_id(self, symbol_id: int) -> Optional[SymbolRecord]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def get_symbol_by_name(self, name: str, kind: Optional[str] = None,
                                 file_path: Optional[str] = None) -> Optional[SymbolRecord]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def get_symbols_in_file(self, file_path: str) -> List[SymbolRecord]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def search_symbols(self, query: str, kind: Optional[str] = None,
                           limit: int = 50) -> List[SymbolRecord]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def delete_symbols_for_file(self, file_path: str) -> int:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def insert_reference(self, ref: ReferenceRecord) -> int:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def get_references_for_symbol(self, symbol_id: int,
                                       kind: Optional[str] = None) -> List[ReferenceRecord]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def delete_references_for_file(self, file_path: str) -> int:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def insert_dependency(self, dep: DependencyRecord) -> int:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def get_dependencies(self, symbol_id: int, direction: str = "outgoing",
                             kind: Optional[str] = None) -> List[DependencyRecord]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def delete_dependencies_for_file(self, file_path: str) -> int:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def get_file_hash(self, file_path: str) -> Optional[str]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def set_file_hash(self, file_path: str, hash_value: str):
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def delete_file_hash(self, file_path: str):
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def get_indexed_files(self) -> List[FileHashRecord]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def bulk_insert_symbols(self, symbols: List[SymbolRecord]) -> List[int]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def bulk_insert_references(self, refs: List[ReferenceRecord]) -> List[int]:
        raise NotImplementedError("PostgreSQL support not yet implemented")

    async def bulk_insert_dependencies(self, deps: List[DependencyRecord]) -> List[int]:
        raise NotImplementedError("PostgreSQL support not yet implemented")


# ============================================================================
# Factory Function
# ============================================================================

def create_metadata_store(settings: dict) -> MetadataStore:
    """Factory function to create appropriate MetadataStore based on settings.

    Args:
        settings: Configuration dictionary with 'metadata' section

    Returns:
        MetadataStore instance (SQLite or PostgreSQL)

    Raises:
        ValueError: If unknown backend specified

    Example settings.yaml:
        metadata:
          backend: sqlite  # or postgresql
          db_path: data/metadata.db  # for sqlite
          # For postgresql:
          # connection_string: postgresql://user:pass@host:5432/akashic
          # pool_size: 10
    """
    metadata_config = settings.get('metadata', {})
    backend = metadata_config.get('backend', 'sqlite')

    if backend == 'sqlite':
        db_path = metadata_config.get('db_path', 'data/metadata.db')
        return SQLiteMetadataStore(db_path)

    elif backend == 'postgresql':
        conn_string = metadata_config.get('connection_string')
        if not conn_string:
            raise ValueError("PostgreSQL backend requires 'connection_string' in settings")

        pool_size = metadata_config.get('pool_size', 10)
        return PostgreSQLMetadataStore(conn_string, pool_size)

    else:
        raise ValueError(
            f"Unknown database backend: {backend}. "
            f"Supported backends: {[e.value for e in DatabaseBackend]}"
        )


# ============================================================================
# Context Manager Support
# ============================================================================

@asynccontextmanager
async def metadata_store_context(settings: dict):
    """Async context manager for MetadataStore.

    Usage:
        async with metadata_store_context(settings) as store:
            symbol_id = await store.insert_symbol(symbol)
            await store.insert_reference(ref)
    """
    store = create_metadata_store(settings)
    try:
        await store.connect()
        yield store
    finally:
        await store.close()


# ============================================================================
# Utility Functions
# ============================================================================

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for change detection.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal hash string
    """
    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)

    return sha256.hexdigest()


async def file_needs_reindexing(
    store: MetadataStore,
    file_path: str,
    current_hash: str
) -> bool:
    """Check if a file needs reindexing based on hash comparison.

    Args:
        store: MetadataStore instance
        file_path: Path to file
        current_hash: Current hash of file

    Returns:
        True if file needs reindexing
    """
    stored_hash = await store.get_file_hash(file_path)

    # New file or changed file
    return stored_hash is None or stored_hash != current_hash
