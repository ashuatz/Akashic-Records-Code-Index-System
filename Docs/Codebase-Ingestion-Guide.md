# Codebase Ingestion Guide - 코드베이스 인덱싱 가이드

## 핵심 질문: "통 파일을 학습해서 넣으려면?"

**결론부터**: 이 시스템에서 "학습"은 LLM 학습이 아니라 **"인덱싱"**입니다.

---

## "학습" vs "인덱싱" 개념 정리

| 개념 | 설명 | 이 시스템에서 |
|------|------|---------------|
| **LLM 학습 (Fine-tuning)** | 모델 가중치 변경 | ❌ 사용 안함 |
| **인덱싱 (Indexing)** | 코드를 검색 가능한 형태로 저장 | ✅ 이것을 함 |

### 왜 LLM 학습이 아닌가?

1. **MCP 전용**: 외부 AI(Claude, ChatGPT)가 직접 코드를 해석
2. **실시간 업데이트**: 인덱싱은 즉시 반영, 학습은 수 시간
3. **소스 추적**: 인덱싱은 "어디서 왔는지" 추적 가능

---

## 인덱싱 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ STEP 1: 파일 스캔                                     │  │
│  │                                                        │  │
│  │   Input: D:\Engines\UnrealEngine\Source\**\*.cpp     │  │
│  │          D:\Engines\Unity\Runtime\**\*.cs            │  │
│  │                                                        │  │
│  │   Filter:                                              │  │
│  │   - 제외: *.uasset, *.dll, ThirdParty/*, Build/*     │  │
│  │   - 포함: *.cs, *.cpp, *.h, *.hpp                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ STEP 2: 청킹 (Chunking)                               │  │
│  │                                                        │  │
│  │   방법: Tree-sitter (빠른 구조 파싱)                  │  │
│  │                                                        │  │
│  │   청크 단위:                                          │  │
│  │   - 클래스/구조체 전체                                │  │
│  │   - 함수/메서드 단위                                  │  │
│  │   - 최대 8000 토큰 (nomic 컨텍스트 한계)             │  │
│  │                                                        │  │
│  │   예시:                                               │  │
│  │   ┌────────────────────────────────────────────────┐ │  │
│  │   │ // CollisionSystem.cs                          │ │  │
│  │   │ public class CollisionSystem {                 │ │  │
│  │   │     public bool DetectCollision(...) { ... }  │ │  │ → Chunk 1
│  │   │     public void ResolveCollision(...) { ... } │ │  │ → Chunk 2
│  │   │ }                                              │ │  │
│  │   └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ STEP 3: 임베딩 생성                                   │  │
│  │                                                        │  │
│  │   Model: nomic-embed-code                             │  │
│  │   Output: 768차원 벡터                                │  │
│  │                                                        │  │
│  │   코드:                                               │  │
│  │   embedding = model.encode(chunk.code)               │  │
│  │   # [0.12, -0.34, 0.56, ...] (768 floats)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ STEP 4: 저장                                          │  │
│  │                                                        │  │
│  │   Qdrant (Vector DB):                                 │  │
│  │   {                                                   │  │
│  │     "id": "uuid-xxx",                                │  │
│  │     "vector": [0.12, -0.34, ...],                   │  │
│  │     "payload": {                                     │  │
│  │       "file_path": "Source/Runtime/Collision.cpp",  │  │
│  │       "symbol_name": "DetectCollision",             │  │
│  │       "symbol_type": "function",                    │  │
│  │       "code": "bool DetectCollision(...) { ... }",  │  │
│  │       "start_line": 42,                             │  │
│  │       "end_line": 87,                               │  │
│  │       "language": "cpp"                             │  │
│  │     }                                                │  │
│  │   }                                                   │  │
│  │                                                        │  │
│  │   SQLite (Metadata):                                  │  │
│  │   - 심볼 테이블 (이름 → 위치)                        │  │
│  │   - BM25 인덱스 (키워드 검색)                        │  │
│  │   - 파일 해시 (변경 감지)                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 실제 구현 코드

### 1. 파일 스캔

```python
from pathlib import Path
import fnmatch

INCLUDE_PATTERNS = ["*.cs", "*.cpp", "*.h", "*.hpp", "*.c"]
EXCLUDE_PATTERNS = [
    "ThirdParty/*", "Plugins/*", "Build/*", "Intermediate/*",
    "*.generated.cs", "*.g.cs", "*.uasset", "*.dll"
]

def scan_codebase(root_path: str) -> list[Path]:
    """코드베이스에서 인덱싱할 파일 목록 반환"""
    root = Path(root_path)
    files = []

    for pattern in INCLUDE_PATTERNS:
        for file in root.rglob(pattern):
            # 제외 패턴 체크
            relative = file.relative_to(root)
            if not any(fnmatch.fnmatch(str(relative), ex) for ex in EXCLUDE_PATTERNS):
                files.append(file)

    return files


# 사용 예시
files = scan_codebase("D:/Engines/UnrealEngine/Source")
print(f"Found {len(files)} files to index")
```

### 2. 청킹 (Tree-sitter)

```python
import tree_sitter_cpp as ts_cpp
import tree_sitter_c_sharp as ts_csharp
from tree_sitter import Parser

def get_parser(language: str) -> Parser:
    parser = Parser()
    if language == "cpp":
        parser.set_language(ts_cpp.language())
    elif language == "csharp":
        parser.set_language(ts_csharp.language())
    return parser


def chunk_file(file_path: Path) -> list[dict]:
    """파일을 시맨틱 청크로 분할"""
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    language = "csharp" if file_path.suffix == ".cs" else "cpp"
    parser = get_parser(language)

    tree = parser.parse(bytes(content, "utf8"))
    chunks = []

    # 함수/메서드/클래스 단위로 추출
    for node in traverse_tree(tree.root_node):
        if node.type in ["function_definition", "method_declaration", "class_declaration"]:
            chunk_text = content[node.start_byte:node.end_byte]

            # 너무 긴 경우 분할
            if len(chunk_text) > 8000:
                # 하위 노드로 분할
                for child in node.children:
                    if child.type in ["function_definition", "method_declaration"]:
                        chunks.append({
                            "code": content[child.start_byte:child.end_byte],
                            "type": child.type,
                            "start_line": child.start_point[0],
                            "end_line": child.end_point[0],
                            "file": str(file_path)
                        })
            else:
                chunks.append({
                    "code": chunk_text,
                    "type": node.type,
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "file": str(file_path)
                })

    return chunks


def traverse_tree(node):
    """AST 트리 순회"""
    yield node
    for child in node.children:
        yield from traverse_tree(child)
```

### 3. 임베딩 생성

```python
import requests

EMBEDDING_SERVER = "http://localhost:8081"

def embed_code(text: str) -> list[float]:
    """코드 텍스트를 768차원 벡터로 변환"""
    response = requests.post(
        f"{EMBEDDING_SERVER}/v1/embeddings",
        json={"input": text, "model": "nomic-embed-code"}
    )
    return response.json()["data"][0]["embedding"]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """청크 리스트에 임베딩 추가"""
    for chunk in chunks:
        chunk["embedding"] = embed_code(chunk["code"])
    return chunks
```

### 4. Qdrant 저장

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

client = QdrantClient("localhost", port=6333)

# Collection 생성 (최초 1회)
def create_collection():
    client.create_collection(
        collection_name="code_index",
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )


def store_chunks(chunks: list[dict]):
    """청크를 Qdrant에 저장"""
    points = []

    for chunk in chunks:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["embedding"],
            payload={
                "file_path": chunk["file"],
                "code": chunk["code"],
                "symbol_type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"]
            }
        )
        points.append(point)

    # 배치 저장
    client.upsert(
        collection_name="code_index",
        points=points
    )
```

### 5. 전체 파이프라인

```python
def ingest_codebase(root_path: str):
    """전체 코드베이스 인덱싱"""

    print(f"Scanning {root_path}...")
    files = scan_codebase(root_path)
    print(f"Found {len(files)} files")

    total_chunks = 0

    for i, file in enumerate(files):
        try:
            # 청킹
            chunks = chunk_file(file)

            # 임베딩
            chunks = embed_chunks(chunks)

            # 저장
            store_chunks(chunks)

            total_chunks += len(chunks)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(files)} files, {total_chunks} chunks")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    print(f"Done! Indexed {total_chunks} chunks from {len(files)} files")


# 실행
if __name__ == "__main__":
    # Unreal Engine 소스
    ingest_codebase("D:/Engines/UnrealEngine/Engine/Source")

    # Unity Runtime
    ingest_codebase("D:/Engines/Unity/Runtime")

    # 커스텀 엔진
    ingest_codebase("D:/Projects/MyEngine/Source")
```

---

## 인덱싱 시간 추정

| 코드베이스 크기 | 예상 파일 수 | 예상 청크 수 | 인덱싱 시간 |
|-----------------|--------------|--------------|-------------|
| 1GB | ~10,000 | ~100,000 | ~30분 |
| 10GB | ~100,000 | ~1,000,000 | ~5시간 |
| 30GB | ~300,000 | ~3,000,000 | ~15시간 |
| 150GB | ~1,500,000 | ~15,000,000 | ~3일 |

**병목**: 임베딩 생성 (GPU 사용 시 5-10배 빨라짐)

---

## 증분 업데이트 (Incremental Update)

전체 재인덱싱 대신 변경된 파일만 업데이트:

```python
import hashlib

def get_file_hash(file_path: Path) -> str:
    """파일 해시 계산"""
    content = file_path.read_bytes()
    return hashlib.md5(content).hexdigest()


def incremental_update(root_path: str):
    """변경된 파일만 업데이트"""

    files = scan_codebase(root_path)
    updated = 0

    for file in files:
        current_hash = get_file_hash(file)
        stored_hash = get_stored_hash(file)  # SQLite에서 조회

        if current_hash != stored_hash:
            # 기존 청크 삭제
            delete_chunks_for_file(file)

            # 새로 인덱싱
            chunks = chunk_file(file)
            chunks = embed_chunks(chunks)
            store_chunks(chunks)

            # 해시 업데이트
            update_stored_hash(file, current_hash)
            updated += 1

    print(f"Updated {updated} files")
```

---

## FAQ

### Q: "학습"이 아니라 "인덱싱"이면, AI가 코드를 정말 이해하는 건가요?

**A**: 네, 하지만 이해하는 주체가 다릅니다.

```
기존 방식 (Fine-tuning):
코드 → LLM 학습 → LLM이 코드 "기억"

이 시스템 (RAG/Indexing):
코드 → 인덱싱 → 검색 → Claude/ChatGPT가 "읽고 이해"
```

Claude나 ChatGPT는 이미 코드를 이해하는 능력이 있습니다.
우리는 그들에게 "필요한 코드를 찾아주는 것"만 하면 됩니다.

### Q: 150GB 코드를 한번에 인덱싱해야 하나요?

**A**: 아니요, 모듈별로 나눠서 가능합니다.

```python
# 우선순위별 인덱싱
ingest_codebase("D:/UE5/Source/Runtime/Engine")      # 핵심 엔진
ingest_codebase("D:/UE5/Source/Runtime/Renderer")    # 렌더러
ingest_codebase("D:/UE5/Source/Runtime/Physics")     # 물리
# ... 나머지는 필요할 때
```

### Q: 새 파일이 추가되면 어떻게 하나요?

**A**: 증분 업데이트 또는 File Watcher 사용:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(('.cs', '.cpp', '.h')):
            reindex_file(event.src_path)

# File watcher 시작
observer = Observer()
observer.schedule(CodeChangeHandler(), path="D:/UE5/Source", recursive=True)
observer.start()
```

---

## 다음 단계

1. **Phase 0 PoC**: 1GB 서브셋으로 파이프라인 검증
2. **서버 구축**: nomic-embed-code + Qdrant 설치
3. **파일럿 인덱싱**: 자주 사용하는 모듈부터 시작
4. **MCP 서버 연결**: Claude Code에서 테스트

---

*문서 버전: 1.0*
*작성일: 2026-01-30*
