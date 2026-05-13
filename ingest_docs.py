"""organize_docs.py 로 정리된 README.md 를 RAG 벡터 DB(Chroma) 에 적재.

호출량 폭주 방지가 목표.
- 폴더당 1 청크 (Front-matter + 본문 발췌, MAX_EMBED_CHARS 상한)
- Front-matter 는 메타데이터로 분리 → Chroma metadata 로 저장 (본문 노이즈 제거)
- sha256 캐시: 변경 없는 README 는 임베딩 호출 자체를 스킵
- AsyncOpenAI(default_headers=...) 로 사내 임베딩 API 호출
- Semaphore + 지수 백오프 재시도로 게이트웨이 rate-limit 회피
"""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import chromadb
from openai import AsyncOpenAI

# ================= CONFIGURATION =================
SOURCE_DIR = r"./test_docs"

# 사내 임베딩 API (OpenAI 호환 게이트웨이 가정).
# Ollama 를 그대로 쓰려면 BASE 를 "http://localhost:11434/v1" 로 지정.
EMBEDDING_API_KEY = "your-api-key"
EMBEDDING_API_BASE = "your-api-endpoint"
EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_DEFAULT_HEADERS: dict[str, str] = {
    # 사내 게이트웨이가 요구하는 헤더가 있다면 여기에 추가
}

# Chroma
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "platform_docs"

# 호출량 제어
MAX_EMBED_CHARS = 800            # 임베딩에 보낼 텍스트 최대 글자수
EMBED_CONCURRENCY = 2            # 동시 호출 수 (사내 게이트웨이가 약하면 1 로)
EMBED_RETRY = 3                  # 실패 시 재시도 횟수 (백오프: 2,4,8s)
CACHE_FILE = "./.embedding_cache.json"
# =================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DocIngestor:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_API_BASE,
            default_headers=EMBEDDING_DEFAULT_HEADERS or None,
        )
        self.sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        self.cache_path = Path(CACHE_FILE)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), starting empty.")

        self.chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma.get_or_create_collection(CHROMA_COLLECTION)

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
        if len(rel_parts) >= 3:  # <category>/<project>/README.md
            return rel_parts[0]
        return "etc"

    # ---------- embedding ----------
    async def _embed(self, text: str) -> list[float] | None:
        async with self.sem:
            for attempt in range(1, EMBED_RETRY + 1):
                try:
                    resp = await self.client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=text,
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

        vec = await self._embed(embed_text)
        if vec is None:
            return "fail:embed"

        category = self._derive_category(readme_path, meta.get("category"))
        rel = readme_path.relative_to(self.source_root).as_posix()
        chroma_meta: dict[str, str | int | float | bool] = {
            "source_path": rel,
            "folder": readme_path.parent.relative_to(self.source_root).as_posix(),
            "project_name": readme_path.parent.name,
            "category": category,
        }
        for k, v in meta.items():
            if k == "category":
                continue
            if v is None:
                continue
            chroma_meta[k] = str(v)

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[vec],
            documents=[embed_text],
            metadatas=[chroma_meta],
        )
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
