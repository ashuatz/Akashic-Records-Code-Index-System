"""
Akashic Records - Core Storage and Retrieval Module

Handles vector storage (Qdrant), embeddings (llama.cpp), reranking (BGE),
and metadata management (SQLite + BM25).
"""

import asyncio
import sqlite3
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import aiohttp
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency for reranking
    CrossEncoder = None

try:
    from runtime_config import load_settings, resolve_settings_path
except ImportError:  # pragma: no cover - package import fallback
    from .runtime_config import load_settings, resolve_settings_path


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CodeChunk:
    """Represents a code chunk to be indexed."""
    file_path: str
    code: str
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None  # "function", "class", "method", etc.
    start_line: int = 0
    end_line: int = 0
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class SearchResult:
    """Search result with scoring information."""
    chunk: CodeChunk
    score: float
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rerank_score": self.rerank_score,
        }


@dataclass
class SymbolInfo:
    """Symbol metadata information."""
    name: str
    type: str
    file_path: str
    start_line: int
    code: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ============================================================================
# Embedding Client
# ============================================================================

class EmbeddingClient:
    """Async client for llama.cpp embedding server."""

    def __init__(self, base_url: str, endpoint: str, model: str, timeout: int = 30):
        """Initialize embedding client.

        Args:
            base_url: Base URL of llama.cpp server (e.g., "http://localhost:8081")
            endpoint: Endpoint path (e.g., "/v1/embeddings")
            model: Model name for embeddings
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_embedding(self, text: str, retry_count: int = 3, max_chars: int = 2000) -> List[float]:
        """Get embedding for a single text.

        For long texts exceeding max_chars, splits into multiple chunks,
        embeds each, and returns the weighted average embedding.

        Args:
            text: Text to embed
            retry_count: Number of retries on failure
            max_chars: Maximum characters per embedding request

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding generation fails after retries
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        # If text is too long, split and average embeddings
        if len(text) > max_chars:
            return await self._get_chunked_embedding(text, max_chars, retry_count)

        return await self._get_single_embedding(text, retry_count)

    async def _get_single_embedding(self, text: str, retry_count: int = 3) -> List[float]:
        """Get embedding for a single text chunk."""
        url = f"{self.base_url}{self.endpoint}"
        is_ollama = "/api/embed" in self.endpoint

        payload = {
            "model": self.model,
            "input": text
        }

        for attempt in range(retry_count):
            try:
                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if is_ollama:
                            return data["embeddings"][0]
                        else:
                            return data["data"][0]["embedding"]
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Embedding request failed (attempt {attempt + 1}/{retry_count}): "
                            f"status={response.status}, error={error_text}"
                        )
            except asyncio.TimeoutError:
                logger.warning(f"Embedding request timeout (attempt {attempt + 1}/{retry_count})")
            except Exception as e:
                logger.warning(f"Embedding request error (attempt {attempt + 1}/{retry_count}): {e}")

            if attempt < retry_count - 1:
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to get embedding after {retry_count} attempts")

    async def _get_chunked_embedding(self, text: str, max_chars: int, retry_count: int = 3) -> List[float]:
        """Split long text into chunks, embed each, and return weighted average.

        Args:
            text: Long text to embed
            max_chars: Maximum characters per chunk
            retry_count: Number of retries per chunk

        Returns:
            Weighted average embedding vector
        """
        # Split text into overlapping chunks
        overlap = int(max_chars * 0.1)  # 10% overlap
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap if end < len(text) else end

        logger.debug(f"Split text ({len(text)} chars) into {len(chunks)} chunks")

        # Get embeddings for all chunks
        embeddings = []
        weights = []

        for i, chunk in enumerate(chunks):
            try:
                emb = await self._get_single_embedding(chunk, retry_count)
                embeddings.append(emb)
                weights.append(len(chunk))  # Weight by chunk length
            except RuntimeError as e:
                logger.warning(f"Failed to embed chunk {i+1}/{len(chunks)}: {e}")
                continue

        if not embeddings:
            raise RuntimeError("Failed to embed any chunks")

        # Compute weighted average
        total_weight = sum(weights)
        dim = len(embeddings[0])
        avg_embedding = [0.0] * dim

        for emb, weight in zip(embeddings, weights):
            w = weight / total_weight
            for j in range(dim):
                avg_embedding[j] += emb[j] * w

        logger.debug(f"Averaged {len(embeddings)} chunk embeddings")
        return avg_embedding

    async def get_embeddings(self, texts: List[str], retry_count: int = 3) -> List[List[float]]:
        """Get embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed
            retry_count: Number of retries on failure

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self.session:
            self.session = aiohttp.ClientSession()

        # Detect API format (Ollama vs OpenAI-compatible)
        is_ollama = "/api/embed" in self.endpoint
        url = f"{self.base_url}{self.endpoint}"

        # Ollama doesn't support batch well, process individually
        if is_ollama:
            embeddings = []
            for text in texts:
                embedding = await self.get_embedding(text, retry_count)
                embeddings.append(embedding)
            return embeddings

        # OpenAI-compatible batch processing
        payload = {
            "model": self.model,
            "input": texts
        }

        for attempt in range(retry_count):
            try:
                async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout * 2)  # Longer timeout for batch
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Return embeddings in order
                        return [item["embedding"] for item in data["data"]]
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Batch embedding request failed (attempt {attempt + 1}/{retry_count}): "
                            f"status={response.status}, error={error_text}"
                        )
            except asyncio.TimeoutError:
                logger.warning(f"Batch embedding timeout (attempt {attempt + 1}/{retry_count})")
            except Exception as e:
                logger.warning(f"Batch embedding error (attempt {attempt + 1}/{retry_count}): {e}")

            if attempt < retry_count - 1:
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to get batch embeddings after {retry_count} attempts")


# ============================================================================
# Reranker
# ============================================================================

class Reranker:
    """BGE-based reranker for search results."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """Initialize reranker.

        Args:
            model_name: HuggingFace model name for CrossEncoder
        """
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self._initialized = False

    def _ensure_loaded(self):
        """Lazy load the model."""
        if not self._initialized:
            try:
                if CrossEncoder is None:
                    raise RuntimeError(
                        "sentence-transformers is not installed. "
                        "Install it or disable reranker in settings."
                    )
                logger.info(f"Loading reranker model: {self.model_name}")
                self.model = CrossEncoder(self.model_name, max_length=512)
                self._initialized = True
                logger.info("Reranker model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Return top K results (None = all)

        Returns:
            List of (index, score) tuples sorted by score descending
        """
        if not documents:
            return []

        self._ensure_loaded()

        if not self.model:
            logger.warning("Reranker model not available, returning empty results")
            return []

        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Get reranking scores
            scores = self.model.predict(pairs)

            # Create (index, score) tuples and sort by score descending
            results = [(idx, float(score)) for idx, score in enumerate(scores)]
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top K if specified
            if top_k is not None:
                results = results[:top_k]

            return results
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return []


# ============================================================================
# Code Store
# ============================================================================

class CodeStore:
    """Main storage and retrieval system for code chunks."""

    def __init__(self, settings_path: Optional[str] = None):
        """Initialize CodeStore.

        Args:
            settings_path: Path to settings.yaml file (default: config/settings.yaml)
        """
        self.settings_path = resolve_settings_path(settings_path)
        self.config = self._load_config()

        # Components (initialized in connect())
        self.qdrant: Optional[QdrantClient] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        self.reranker: Optional[Reranker] = None
        self.db_conn: Optional[sqlite3.Connection] = None

        self._connected = False

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config = load_settings(str(self.settings_path))
            logger.info(f"Configuration loaded from {self.settings_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    async def connect(self):
        """Initialize all connections and components."""
        if self._connected:
            logger.warning("CodeStore already connected")
            return

        try:
            # Initialize Qdrant client
            qdrant_config = self.config['qdrant']
            self.qdrant = QdrantClient(
                host=qdrant_config['host'],
                port=qdrant_config['port']
            )
            logger.info(f"Connected to Qdrant at {qdrant_config['host']}:{qdrant_config['port']}")

            # Create collection if it doesn't exist
            await self._ensure_collection()

            # Initialize embedding client
            emb_config = self.config['embedding']
            self.embedding_client = EmbeddingClient(
                base_url=emb_config['url'],
                endpoint=emb_config['endpoint'],
                model=emb_config['model'],
                timeout=emb_config['timeout']
            )
            await self.embedding_client.__aenter__()
            logger.info(f"Embedding client initialized: {emb_config['url']}")

            # Initialize reranker if enabled
            reranker_config = self.config['reranker']
            if reranker_config['enabled']:
                try:
                    self.reranker = Reranker(model_name=reranker_config['model'])
                    logger.info("Reranker initialized")
                except Exception as e:
                    logger.warning(f"Reranker initialization failed (will continue without it): {e}")
                    self.reranker = None

            # Initialize SQLite metadata DB
            await self._init_metadata_db()

            self._connected = True
            logger.info("CodeStore connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect CodeStore: {e}")
            await self.close()
            raise

    async def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        collection_name = self.config['qdrant']['collection']
        dimensions = self.config['embedding']['dimensions']

        try:
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                logger.info(f"Creating Qdrant collection: {collection_name}")
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimensions,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{collection_name}' created")
            else:
                logger.info(f"Collection '{collection_name}' already exists")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    async def _init_metadata_db(self):
        """Initialize SQLite metadata database."""
        metadata_config = self.config['metadata']
        db_path = Path(metadata_config['db_path'])

        # Create parent directory if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to SQLite
        self.db_conn = sqlite3.connect(str(db_path))
        self.db_conn.row_factory = sqlite3.Row

        # Create tables
        cursor = self.db_conn.cursor()

        # Chunks metadata table
        cursor.execute("""
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
        """)

        # Full-text search index (FTS5 for BM25)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                code,
                symbol_name,
                tokenize = 'porter unicode61'
            )
        """)

        # Symbols index for quick lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_name
            ON chunks(symbol_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path
            ON chunks(file_path)
        """)

        self.db_conn.commit()
        logger.info(f"SQLite metadata DB initialized: {db_path}")

    async def close(self):
        """Close all connections and cleanup."""
        if self.embedding_client:
            try:
                await self.embedding_client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing embedding client: {e}")

        if self.db_conn:
            try:
                self.db_conn.close()
            except Exception as e:
                logger.warning(f"Error closing SQLite connection: {e}")

        # Qdrant client doesn't need explicit closing

        self._connected = False
        logger.info("CodeStore closed")

    async def add_chunk(self, chunk: CodeChunk) -> str:
        """Add a single code chunk to the store.

        Args:
            chunk: CodeChunk to add

        Returns:
            Chunk ID
        """
        if not self._connected:
            raise RuntimeError("CodeStore not connected. Call connect() first.")

        # Generate unique ID (UUID for Qdrant compatibility)
        import uuid
        import hashlib
        chunk_id_str = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}"
        chunk_id = str(uuid.UUID(hashlib.md5(chunk_id_str.encode()).hexdigest()))

        try:
            # Get embedding
            embedding = await self.embedding_client.get_embedding(chunk.code)

            # Store in Qdrant
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "file_path": chunk.file_path,
                    "symbol_name": chunk.symbol_name,
                    "symbol_type": chunk.symbol_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "code": chunk.code,
                }
            )

            self.qdrant.upsert(
                collection_name=self.config['qdrant']['collection'],
                points=[point]
            )

            # Store metadata in SQLite
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO chunks
                (id, file_path, symbol_name, symbol_type, start_line, end_line, language, code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                chunk.file_path,
                chunk.symbol_name,
                chunk.symbol_type,
                chunk.start_line,
                chunk.end_line,
                chunk.language,
                chunk.code
            ))

            # Add to FTS index
            cursor.execute("""
                INSERT OR REPLACE INTO chunks_fts (id, code, symbol_name)
                VALUES (?, ?, ?)
            """, (chunk_id, chunk.code, chunk.symbol_name or ""))

            self.db_conn.commit()

            logger.debug(f"Added chunk: {chunk_id}")
            return chunk_id

        except Exception as e:
            logger.error(f"Failed to add chunk {chunk_id}: {e}")
            raise

    async def add_chunks(self, chunks: List[CodeChunk]) -> List[str]:
        """Add multiple code chunks in batch.

        Args:
            chunks: List of CodeChunks to add

        Returns:
            List of chunk IDs
        """
        if not self._connected:
            raise RuntimeError("CodeStore not connected. Call connect() first.")

        if not chunks:
            return []

        try:
            # Generate IDs (UUID for Qdrant compatibility)
            import uuid
            import hashlib
            chunk_ids = [
                str(uuid.UUID(hashlib.md5(f"{c.file_path}:{c.start_line}:{c.end_line}".encode()).hexdigest()))
                for c in chunks
            ]
            # Keep original string IDs for SQLite
            chunk_id_strs = [
                f"{c.file_path}:{c.start_line}:{c.end_line}"
                for c in chunks
            ]

            # Get embeddings in batch
            texts = [c.code for c in chunks]
            embeddings = await self.embedding_client.get_embeddings(texts)

            # Create Qdrant points
            points = [
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "file_path": chunk.file_path,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "language": chunk.language,
                        "code": chunk.code,
                    }
                )
                for chunk_id, chunk, embedding in zip(chunk_ids, chunks, embeddings)
            ]

            # Batch upsert to Qdrant
            self.qdrant.upsert(
                collection_name=self.config['qdrant']['collection'],
                points=points
            )

            # Batch insert to SQLite
            cursor = self.db_conn.cursor()

            # Metadata
            cursor.executemany("""
                INSERT OR REPLACE INTO chunks
                (id, file_path, symbol_name, symbol_type, start_line, end_line, language, code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    chunk_id,
                    chunk.file_path,
                    chunk.symbol_name,
                    chunk.symbol_type,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.language,
                    chunk.code
                )
                for chunk_id, chunk in zip(chunk_ids, chunks)
            ])

            # FTS index
            cursor.executemany("""
                INSERT OR REPLACE INTO chunks_fts (id, code, symbol_name)
                VALUES (?, ?, ?)
            """, [
                (chunk_id, chunk.code, chunk.symbol_name or "")
                for chunk_id, chunk in zip(chunk_ids, chunks)
            ])

            self.db_conn.commit()

            logger.info(f"Added {len(chunks)} chunks in batch")
            return chunk_ids

        except Exception as e:
            logger.error(f"Failed to add chunks in batch: {e}")
            raise

    async def search(
        self,
        query: str,
        limit: int = 10,
        language: Optional[str] = None,
        use_reranker: bool = True
    ) -> List[SearchResult]:
        """Search for code chunks using hybrid vector + keyword search.

        Args:
            query: Search query
            limit: Maximum number of results
            language: Filter by programming language (optional)
            use_reranker: Use reranker for result refinement

        Returns:
            List of SearchResults sorted by relevance
        """
        if not self._connected:
            raise RuntimeError("CodeStore not connected. Call connect() first.")

        try:
            # Step 1: Vector search in Qdrant
            query_embedding = await self.embedding_client.get_embedding(query)

            # Build filter if language specified
            search_filter = None
            if language:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="language",
                            match=MatchValue(value=language)
                        )
                    ]
                )

            # Retrieve more candidates for reranking
            candidate_limit = limit
            if use_reranker and self.reranker:
                reranker_config = self.config['reranker']
                candidate_limit = min(
                    reranker_config.get('top_k_candidates', 100),
                    self.config['search'].get('max_limit', 100)
                )

            # Perform vector search (using query_points for newer qdrant-client)
            try:
                # Try newer API first
                from qdrant_client.models import QueryRequest
                search_response = self.qdrant.query_points(
                    collection_name=self.config['qdrant']['collection'],
                    query=query_embedding,
                    query_filter=search_filter,
                    limit=candidate_limit,
                    with_payload=True
                )
                search_results = search_response.points
            except (AttributeError, ImportError):
                # Fallback to older API
                search_results = self.qdrant.search(
                    collection_name=self.config['qdrant']['collection'],
                    query_vector=query_embedding,
                    query_filter=search_filter,
                    limit=candidate_limit
                )

            # Step 2: Convert to SearchResult objects
            results = []
            for hit in search_results:
                chunk = CodeChunk(
                    file_path=hit.payload['file_path'],
                    code=hit.payload['code'],
                    symbol_name=hit.payload.get('symbol_name'),
                    symbol_type=hit.payload.get('symbol_type'),
                    start_line=hit.payload.get('start_line', 0),
                    end_line=hit.payload.get('end_line', 0),
                    language=hit.payload.get('language')
                )
                results.append(SearchResult(
                    chunk=chunk,
                    score=hit.score
                ))

            # Step 3: Rerank if enabled
            if use_reranker and self.reranker and results:
                try:
                    documents = [r.chunk.code for r in results]
                    rerank_results = self.reranker.rerank(
                        query=query,
                        documents=documents,
                        top_k=limit
                    )

                    # Update with rerank scores and reorder
                    if rerank_results:
                        reranked = []
                        for idx, rerank_score in rerank_results:
                            results[idx].rerank_score = rerank_score
                            reranked.append(results[idx])
                        results = reranked
                        logger.debug(f"Reranked {len(results)} results")
                    else:
                        # Reranker returned empty, use vector results
                        logger.warning("Reranker returned empty results, using vector search results")
                        results = results[:limit]

                except Exception as e:
                    logger.warning(f"Reranking failed, using vector search results: {e}")
                    # Fallback to vector search results on reranking failure
                    results = results[:limit]
            else:
                # Just return top K vector results
                results = results[:limit]

            logger.info(f"Search returned {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def get_symbol(
        self,
        name: str,
        symbol_type: Optional[str] = None
    ) -> Optional[SymbolInfo]:
        """Get symbol information by name.

        Args:
            name: Symbol name
            symbol_type: Symbol type filter (optional)

        Returns:
            SymbolInfo if found, None otherwise
        """
        if not self._connected:
            raise RuntimeError("CodeStore not connected. Call connect() first.")

        try:
            cursor = self.db_conn.cursor()

            if symbol_type:
                cursor.execute("""
                    SELECT file_path, symbol_type, start_line, code
                    FROM chunks
                    WHERE symbol_name = ? AND symbol_type = ?
                    LIMIT 1
                """, (name, symbol_type))
            else:
                cursor.execute("""
                    SELECT file_path, symbol_type, start_line, code
                    FROM chunks
                    WHERE symbol_name = ?
                    LIMIT 1
                """, (name,))

            row = cursor.fetchone()

            if row:
                return SymbolInfo(
                    name=name,
                    type=row['symbol_type'],
                    file_path=row['file_path'],
                    start_line=row['start_line'],
                    code=row['code']
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get symbol '{name}': {e}")
            raise

    async def list_files(
        self,
        path_pattern: Optional[str] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """List all indexed files with optional filters.

        Args:
            path_pattern: Glob pattern to filter file paths (e.g., "*/Camera/*")
            language: Language filter (e.g., "cpp", "csharp")

        Returns:
            List of file paths
        """
        if not self._connected:
            raise RuntimeError("CodeStore not connected. Call connect() first.")

        try:
            cursor = self.db_conn.cursor()

            query = "SELECT DISTINCT file_path FROM chunks WHERE 1=1"
            params = []

            if language:
                query += " AND language = ?"
                params.append(language)

            if path_pattern:
                # Convert glob pattern to SQL LIKE pattern
                like_pattern = path_pattern.replace("*", "%").replace("?", "_")
                query += " AND file_path LIKE ?"
                params.append(f"%{like_pattern}%")

            query += " ORDER BY file_path LIMIT 1000"

            cursor.execute(query, params)
            return [row['file_path'] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise

    async def delete_file_chunks(self, file_path: str) -> int:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path of file whose chunks should be deleted

        Returns:
            Number of chunks deleted
        """
        if not self._connected:
            raise RuntimeError("CodeStore not connected. Call connect() first.")

        try:
            # Get chunk IDs to delete
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,))
            chunk_ids = [row['id'] for row in cursor.fetchall()]

            if not chunk_ids:
                logger.debug(f"No chunks found for file: {file_path}")
                return 0

            # Delete from Qdrant
            self.qdrant.delete(
                collection_name=self.config['qdrant']['collection'],
                points_selector=chunk_ids
            )

            # Delete from SQLite
            cursor.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            cursor.execute("DELETE FROM chunks_fts WHERE id IN ({})".format(
                ','.join('?' * len(chunk_ids))
            ), chunk_ids)

            self.db_conn.commit()

            logger.info(f"Deleted {len(chunk_ids)} chunks for file: {file_path}")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Failed to delete chunks for file '{file_path}': {e}")
            raise


# ============================================================================
# Context Manager Support
# ============================================================================

@asynccontextmanager
async def code_store_context(settings_path: Optional[str] = None):
    """Async context manager for CodeStore.

    Usage:
        async with code_store_context() as store:
            await store.add_chunk(chunk)
    """
    store = CodeStore(settings_path)
    try:
        await store.connect()
        yield store
    finally:
        await store.close()
