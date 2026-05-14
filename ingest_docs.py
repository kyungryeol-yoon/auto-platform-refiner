"""organize_docs.py 로 정리된 README.md 를 RAG 벡터 DB 에 적재.

호출량 폭주 방지가 목표.
- 폴더당 1 청크 (Front-matter + 본문 발췌, MAX_EMBED_CHARS 상한)
- Front-matter 는 메타데이터로 분리 → 벡터 DB metadata 로 저장 (본문 노이즈 제거)
- sha256 캐시: 변경 없는 README 는 임베딩 호출 자체를 스킵
- Semaphore + 지수 백오프 재시도로 게이트웨이 rate-limit 회피

사내망 규칙 준수:
- 임베딩:  embed = AsyncOpenAI(base_url, api_key, default_headers)
           embed.embeddings.create(input=[text], model=EMBEDDING_MODEL)
- 벡터DB:  VECTOR_DB 스위치로 chroma / qdrant / pgvector 선택. 모두 host:port
           로 접속하는 사내 k8s 인스턴스 가정.
"""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI

# ================= CONFIGURATION =================
SOURCE_DIR = r"./test_docs"

# 사내 임베딩 API (OpenAI 호환 게이트웨이).
#   embed = AsyncOpenAI(base_url, api_key, default_headers)
#   embed.embeddings.create(input=[text], model=EMBEDDING_MODEL)
EMBEDDING_API_BASE = "your-api-endpoint"
EMBEDDING_API_KEY = "your-api-key"
EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_DIM = 1024             # 모델별: mxbai-embed-large=1024, nomic-embed-text=768
EMBEDDING_DEFAULT_HEADERS: dict[str, str] = {
    # 사내 게이트웨이가 요구하는 헤더 (인증/테넌트 등)
    # 예) "X-Tenant-Id": "platform-team",
}

# ---------- 벡터 DB 선택 ----------
VECTOR_DB = "chroma"   # "chroma" | "qdrant" | "pgvector"

# Chroma (사내 k8s 의 chromadb 서비스)
CHROMA_HOST = "chroma.intra"
CHROMA_PORT = 8000
CHROMA_SSL = False
CHROMA_HEADERS: dict[str, str] = {}
CHROMA_COLLECTION = "platform_docs"

# Qdrant (사내 k8s 의 qdrant 서비스)
QDRANT_HOST = "qdrant.intra"
QDRANT_PORT = 6333
QDRANT_HTTPS = False
QDRANT_API_KEY: str | None = None
QDRANT_COLLECTION = "platform_docs"
QDRANT_DISTANCE = "Cosine"       # "Cosine" | "Dot" | "Euclid"

# pgvector (사내 k8s 의 postgres + vector extension)
PGVECTOR_DSN = "postgresql://rag:rag@pgvector.intra:5432/rag"
PGVECTOR_TABLE = "platform_docs"
# 참고 — 테이블 스키마는 미리 만들어 두어야 합니다:
#   CREATE EXTENSION IF NOT EXISTS vector;
#   CREATE TABLE platform_docs (
#       id        TEXT PRIMARY KEY,
#       embedding vector(1024),     -- EMBEDDING_DIM 과 일치
#       document  TEXT,
#       metadata  JSONB
#   );

# 호출량 제어
MAX_EMBED_CHARS = 800
EMBED_CONCURRENCY = 2
EMBED_RETRY = 3
CACHE_FILE = "./.embedding_cache.json"
# =================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _validate_headers(headers: dict | None, var_name: str) -> dict[str, str] | None:
    """헤더 dict 값이 모두 문자열인지 검증. 잘못된 값 발견 시 즉시 fail-fast."""
    if not headers:
        return None
    for k, v in headers.items():
        if not isinstance(v, (str, bytes)):
            raise TypeError(
                f"{var_name}['{k}'] 값은 문자열이어야 합니다. 받은 값: {type(v).__name__} = {v!r}\n"
                f"올바른 예: {var_name} = {{'X-Tenant-Id': 'platform-team', "
                f"'Authorization': 'Bearer xxx'}}"
            )
    return dict(headers)


# ================= Vector store backends =================
class VectorStore(Protocol):
    """공통 인터페이스. 새 백엔드 추가 시 upsert 만 구현하면 됨."""
    name: str
    location: str

    def upsert(
        self,
        doc_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None: ...


class ChromaStore:
    name = "chroma"

    def __init__(self) -> None:
        import chromadb  # 선택적 의존성
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=CHROMA_SSL,
            headers=_validate_headers(CHROMA_HEADERS, "CHROMA_HEADERS"),
        )
        self.collection = client.get_or_create_collection(CHROMA_COLLECTION)
        scheme = "https" if CHROMA_SSL else "http"
        self.location = f"{scheme}://{CHROMA_HOST}:{CHROMA_PORT}/{CHROMA_COLLECTION}"

    def upsert(self, doc_id, embedding, document, metadata):
        # Chroma 메타데이터는 scalar(str/int/float/bool) 만 허용
        flat = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in metadata.items() if v is not None}
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[flat],
        )


class QdrantStore:
    name = "qdrant"

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            https=QDRANT_HTTPS,
            api_key=QDRANT_API_KEY,
        )
        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
        }
        if not self.client.collection_exists(QDRANT_COLLECTION):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=distance_map.get(QDRANT_DISTANCE, Distance.COSINE),
                ),
            )
        scheme = "https" if QDRANT_HTTPS else "http"
        self.location = f"{scheme}://{QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION}"

    def upsert(self, doc_id, embedding, document, metadata):
        import uuid
        from qdrant_client.http.models import PointStruct
        # Qdrant 의 point id 는 UUID 또는 unsigned int 만 허용 → doc_id 로 UUID5 생성
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))
        payload = {"doc_id": doc_id, "document": document, **metadata}
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )


class PgVectorStore:
    name = "pgvector"

    def __init__(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(PGVECTOR_DSN, autocommit=False)
        register_vector(self.conn)
        # 테이블 이름은 SQL 식별자 — 화이트리스트 검증
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", PGVECTOR_TABLE):
            raise ValueError(f"Unsafe PGVECTOR_TABLE: {PGVECTOR_TABLE}")
        self.table = PGVECTOR_TABLE
        self.location = f"{PGVECTOR_DSN}#{self.table}"

    def upsert(self, doc_id, embedding, document, metadata):
        with self.conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table} (id, embedding, document, metadata) "
                f"VALUES (%s, %s, %s, %s) "
                f"ON CONFLICT (id) DO UPDATE SET "
                f"  embedding = EXCLUDED.embedding, "
                f"  document  = EXCLUDED.document, "
                f"  metadata  = EXCLUDED.metadata",
                (doc_id, embedding, document, json.dumps(metadata, ensure_ascii=False)),
            )
        self.conn.commit()


def make_vector_store(kind: str) -> VectorStore:
    if kind == "chroma":
        return ChromaStore()
    if kind == "qdrant":
        return QdrantStore()
    if kind == "pgvector":
        return PgVectorStore()
    raise ValueError(f"Unknown VECTOR_DB: {kind!r} (expected chroma|qdrant|pgvector)")
# =========================================================


class DocIngestor:
    def __init__(self) -> None:
        # 임베딩 클라이언트 (사내망 규칙: AsyncOpenAI + input=[text])
        self.embed = AsyncOpenAI(
            base_url=EMBEDDING_API_BASE,
            api_key=EMBEDDING_API_KEY,
            default_headers=_validate_headers(EMBEDDING_DEFAULT_HEADERS, "EMBEDDING_DEFAULT_HEADERS"),
        )
        self.sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        self.cache_path = Path(CACHE_FILE)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), starting empty.")

        self.store: VectorStore = make_vector_store(VECTOR_DB)
        self.source_root = Path(SOURCE_DIR).resolve()

    # ---------- parsing ----------
    @staticmethod
    def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
        """YAML Front-matter 와 본문 분리. 미존재 시 ({}, text)."""
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
        if not m:
            return {}, text
        front_raw, body = m.group(1), m.group(2)
        meta: dict[str, str] = {}
        for line in front_raw.splitlines():
            mm = re.match(r"^\s*([\w\-]+)\s*:\s*(.+?)\s*$", line)
            if mm:
                key = mm.group(1).strip()
                val = mm.group(2).strip().strip("\"'")
                meta[key] = val
        return meta, body

    @staticmethod
    def _build_embed_text(body: str, max_chars: int = MAX_EMBED_CHARS) -> str:
        """본문에서 임베딩에 보낼 발췌 - 코드블럭 제거 + 첫 부분 발췌."""
        stripped = re.sub(r"```[\s\S]*?```", "", body)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
        return stripped[:max_chars]

    # ---------- collection ----------
    def _readme_files(self) -> list[Path]:
        return [p for p in self.source_root.rglob("README.md") if p.is_file()]

    def _doc_id(self, readme_path: Path) -> str:
        return readme_path.relative_to(self.source_root).as_posix()

    def _derive_category(self, readme_path: Path, fm_category: str | None) -> str:
        """category 우선순위: Front-matter > 부모 폴더 (target/<category>/<project>/README.md)."""
        if fm_category:
            return fm_category
        rel_parts = readme_path.relative_to(self.source_root).parts
        if len(rel_parts) >= 3:
            return rel_parts[0]
        return "etc"

    # ---------- embedding ----------
    async def _embed_text(self, text: str) -> list[float] | None:
        async with self.sem:
            for attempt in range(1, EMBED_RETRY + 1):
                try:
                    resp = await self.embed.embeddings.create(
                        input=[text],
                        model=EMBEDDING_MODEL,
                    )
                    return resp.data[0].embedding
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(
                        f"   embedding fail [{attempt}/{EMBED_RETRY}] {e} — retry in {wait}s"
                    )
                    await asyncio.sleep(wait)
            return None

    async def ingest_one(self, readme_path: Path) -> str:
        try:
            text = readme_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"read fail {readme_path}: {e}")
            return "fail:read"

        meta, body = self._parse_front_matter(text)
        embed_text = self._build_embed_text(body)
        if not embed_text.strip():
            logger.info(f"   skip (empty) {readme_path}")
            return "skip:empty"

        # Hash cache — embed 대상 텍스트 자체가 동일하면 재호출 안 함
        digest = hashlib.sha256(embed_text.encode("utf-8")).hexdigest()
        doc_id = self._doc_id(readme_path)
        if self.cache.get(doc_id) == digest:
            return "skip:cached"

        vec = await self._embed_text(embed_text)
        if vec is None:
            return "fail:embed"

        category = self._derive_category(readme_path, meta.get("category"))
        rel = readme_path.relative_to(self.source_root).as_posix()
        metadata: dict[str, Any] = {
            "source_path": rel,
            "folder": readme_path.parent.relative_to(self.source_root).as_posix(),
            "project_name": readme_path.parent.name,
            "category": category,
        }
        for k, v in meta.items():
            if k == "category" or v is None:
                continue
            metadata[k] = str(v)

        try:
            self.store.upsert(doc_id, vec, embed_text, metadata)
        except Exception as e:
            logger.error(f"   vector-store upsert failed for {doc_id}: {e}")
            return "fail:upsert"

        self.cache[doc_id] = digest
        return "ok"

    # ---------- run ----------
    async def run(self) -> None:
        if not self.source_root.exists():
            logger.error(f"SOURCE_DIR not found: {self.source_root}")
            return

        readmes = self._readme_files()
        if not readmes:
            logger.error("No README.md found. Run organize_docs.py first.")
            return

        logger.info(
            f"Ingest start - {len(readmes)} README files, "
            f"store={self.store.name} ({self.store.location}), "
            f"concurrency={EMBED_CONCURRENCY}, max_chars={MAX_EMBED_CHARS}"
        )
        results = await asyncio.gather(*(self.ingest_one(p) for p in readmes))

        counts: dict[str, int] = {}
        for r in results:
            counts[r] = counts.get(r, 0) + 1
        logger.info(f"Ingest done: {counts}")

        try:
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(DocIngestor().run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
