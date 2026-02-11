"""
Initialize Qdrant collection and SQLite database for Akashic Records.

This script sets up:
1. Qdrant collection with vector storage
2. SQLite database with metadata tables
3. FTS5 virtual table for keyword search
"""

import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path

import click
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime_config import load_dotenv, load_settings as load_runtime_settings

load_dotenv()

# Handle different qdrant-client versions for RpcError import
try:
    from qdrant_client.http.exceptions import UnexpectedResponse as RpcError
except ImportError:
    try:
        from qdrant_client.exceptions import RpcError
    except ImportError:
        # Fallback: create a dummy exception class
        class RpcError(Exception):
            pass


def load_settings(settings_path: str) -> dict:
    """Load settings from YAML configuration file."""
    try:
        return load_runtime_settings(settings_path)
    except FileNotFoundError:
        click.echo(f"Error: Settings file not found at {settings_path}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading settings file: {e}", err=True)
        sys.exit(1)


def init_qdrant(settings: dict, reset: bool = False) -> QdrantClient:
    """Initialize Qdrant collection for vector storage."""
    qdrant_config = settings.get('qdrant', {})
    embedding_config = settings.get('embedding', {})

    host = qdrant_config.get('host', 'localhost')
    port = qdrant_config.get('port', 6333)
    collection_name = qdrant_config.get('collection', 'code_index')
    vector_size = embedding_config.get('dimensions', 768)

    try:
        client = QdrantClient(host=host, port=port)
        click.echo(f"Connected to Qdrant at {host}:{port}")

        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name)
            if reset:
                click.echo(f"Dropping existing collection '{collection_name}'...")
                client.delete_collection(collection_name)
                click.echo(f"Collection '{collection_name}' dropped.")
            else:
                click.echo(
                    f"Collection '{collection_name}' already exists. "
                    f"Use --reset to recreate it."
                )
                return client
        except RpcError:
            click.echo(f"Collection '{collection_name}' does not exist.")

        # Create new collection
        click.echo(f"Creating collection '{collection_name}' with {vector_size}D vectors...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        click.echo(f"Collection '{collection_name}' created successfully.")

        return client

    except Exception as e:
        click.echo(f"Error initializing Qdrant: {e}", err=True)
        sys.exit(1)


def init_sqlite(settings: dict, reset: bool = False) -> sqlite3.Connection:
    """Initialize SQLite database with metadata tables."""
    metadata_config = settings.get('metadata', {})
    db_path = Path(metadata_config.get('db_path', './data/metadata.db'))

    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Check if database exists
        if db_path.exists() and reset:
            click.echo(f"Removing existing database at {db_path}...")
            db_path.unlink()

        connection = sqlite3.connect(str(db_path))
        cursor = connection.cursor()

        click.echo(f"Initializing SQLite database at {db_path}...")

        # Create files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT,
                indexed_at TIMESTAMP,
                language TEXT,
                chunk_count INTEGER DEFAULT 0
            )
        ''')
        click.echo("Created 'files' table")

        # Create symbols table (enhanced version)
        cursor.execute('''
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
                FOREIGN KEY (parent_id) REFERENCES symbols(id)
            )
        ''')
        click.echo("Created 'symbols' table")

        # Create indexes on symbols for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind)
        ''')

        # Create chunks table for storing vector embeddings metadata
        # Note: 'id' is TEXT (UUID) for Qdrant compatibility
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                symbol_name TEXT,
                symbol_type TEXT,
                start_line INTEGER,
                end_line INTEGER,
                language TEXT,
                code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        click.echo("Created 'chunks' table")

        # Create index on chunks.file_path for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunks_file_path
            ON chunks(file_path)
        ''')

        # Create symbol_references table (renamed from 'references' which is a SQL keyword)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                "column" INTEGER NOT NULL,
                kind TEXT,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id)
            )
        ''')
        click.echo("Created 'symbol_references' table")

        # Create indexes on symbol_references
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_references_symbol ON symbol_references(symbol_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_references_file ON symbol_references(file_path)
        ''')

        # Create dependencies table (class-level relationships)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_symbol_id INTEGER NOT NULL,
                to_symbol_id INTEGER NOT NULL,
                kind TEXT,
                FOREIGN KEY (from_symbol_id) REFERENCES symbols(id),
                FOREIGN KEY (to_symbol_id) REFERENCES symbols(id)
            )
        ''')
        click.echo("Created 'dependencies' table")

        # Create indexes on dependencies
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_deps_from ON dependencies(from_symbol_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_deps_to ON dependencies(to_symbol_id)
        ''')

        # Create file_hashes table for incremental indexing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        click.echo("Created 'file_hashes' table")

        # Create FTS5 virtual table for keyword search on chunks
        # Note: Named 'chunks_fts' to match code_store.py
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                code,
                symbol_name,
                tokenize = 'porter unicode61'
            )
        ''')
        click.echo("Created 'chunks_fts' virtual table (FTS5)")

        # Create FTS5 virtual table for symbols
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                name,
                signature,
                content='symbols',
                content_rowid='id'
            )
        ''')
        click.echo("Created 'symbols_fts' virtual table (FTS5)")

        # Create index metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        click.echo("Created 'index_metadata' table")

        # Initialize metadata
        cursor.execute('''
            INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', ('last_indexed', datetime.utcnow().isoformat(), datetime.utcnow()))

        cursor.execute('''
            INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', ('total_files', '0', datetime.utcnow()))

        cursor.execute('''
            INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', ('total_chunks', '0', datetime.utcnow()))

        connection.commit()
        click.echo(f"SQLite database initialized successfully at {db_path}")

        return connection

    except sqlite3.Error as e:
        click.echo(f"Error initializing SQLite database: {e}", err=True)
        sys.exit(1)


def verify_setup(settings: dict, qdrant_client: QdrantClient, sqlite_conn: sqlite3.Connection):
    """Verify that both Qdrant and SQLite are properly initialized."""
    qdrant_config = settings.get('qdrant', {})
    collection_name = qdrant_config.get('collection', 'code_index')

    click.echo("\n" + "="*60)
    click.echo("VERIFICATION REPORT")
    click.echo("="*60)

    # Verify Qdrant
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        click.echo(f"\n[OK] Qdrant collection '{collection_name}' exists")
        click.echo(f"  - Vector size: {collection_info.config.params.vectors.size}")
        click.echo(f"  - Distance metric: {collection_info.config.params.vectors.distance}")
        click.echo(f"  - Points count: {collection_info.points_count}")
    except Exception as e:
        click.echo(f"\n[ERROR] Qdrant verification failed: {e}", err=True)

    # Verify SQLite
    try:
        cursor = sqlite_conn.cursor()

        # Check tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('files', 'symbols', 'chunks', 'symbol_references', 'dependencies', 'file_hashes', 'index_metadata')"
        )
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]

        click.echo(f"\n[OK] SQLite tables exist: {', '.join(table_names)}")

        # Check FTS5 virtual tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks_fts', 'symbols_fts')"
        )
        fts_tables = cursor.fetchall()
        fts_names = [t[0] for t in fts_tables]
        if fts_names:
            click.echo(f"[OK] FTS5 virtual tables exist: {', '.join(fts_names)}")

        # Get row counts
        cursor.execute("SELECT COUNT(*) FROM files")
        files_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM symbols")
        symbols_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunks_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM symbol_references")
        references_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dependencies")
        dependencies_count = cursor.fetchone()[0]

        click.echo(f"\n  - Files: {files_count}")
        click.echo(f"  - Symbols: {symbols_count}")
        click.echo(f"  - Chunks: {chunks_count}")
        click.echo(f"  - References: {references_count}")
        click.echo(f"  - Dependencies: {dependencies_count}")

    except sqlite3.Error as e:
        click.echo(f"\n[ERROR] SQLite verification failed: {e}", err=True)

    click.echo("\n" + "="*60)
    click.echo("Setup complete!")
    click.echo("="*60 + "\n")


@click.command()
@click.option(
    '--settings',
    default=os.environ.get('AKASHIC_SETTINGS_PATH', 'config/settings.yaml'),
    help='Path to settings YAML file'
)
@click.option(
    '--reset',
    is_flag=True,
    help='Drop and recreate collections and databases'
)
def main(settings: str, reset: bool):
    """Initialize Qdrant collection and SQLite database for Akashic Records."""

    click.echo("="*60)
    click.echo("Akashic Records - Initialization Script")
    click.echo("="*60 + "\n")

    # Load settings
    click.echo(f"Loading settings from '{settings}'...")
    config = load_settings(settings)
    click.echo("Settings loaded successfully.\n")

    # Initialize Qdrant
    click.echo("Initializing Qdrant vector database...")
    qdrant_client = init_qdrant(config, reset=reset)
    click.echo()

    # Initialize SQLite
    click.echo("Initializing SQLite metadata database...")
    sqlite_conn = init_sqlite(config, reset=reset)
    click.echo()

    # Verify setup
    verify_setup(config, qdrant_client, sqlite_conn)

    # Close connections
    sqlite_conn.close()
    click.echo("Database connections closed.")


if __name__ == '__main__':
    main()
