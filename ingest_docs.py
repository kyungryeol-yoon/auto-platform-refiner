import os
from pathlib import Path
import logging
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ================= CONFIGURATION =================
# 1. 소스 경로 (organize_docs.py에 의해 정리된 폴더)
SOURCE_DIR = r"./test_docs" 

# 2. Ollama 임베딩 설정
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large" # 또는 "nomic-embed-text"

# 3. 벡터 DB 저장 경로 (ChromaDB 기준)
CHROMA_PERSIST_DIR = "./chroma_db"
# =================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocIngestor:
    def __init__(self):
        # Ollama를 이용한 임베딩 모델 설정
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL
        )
        # 텍스트 분할 설정 (Chunking)
        # RAG 성능을 위해 의미 있는 단위로 쪼갭니다.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True
        )

    def load_documents(self):
        """정리된 폴더 구조에서 README.md 파일만 수집합니다."""
        logger.info(f"📄 {SOURCE_DIR} 내의 README.md 파일을 수집 중입니다...")
        
        if not Path(SOURCE_DIR).exists():
            logger.error(f"경로가 존재하지 않습니다: {SOURCE_DIR}")
            return []

        # DirectoryLoader는 폴더 구조를 유지하며 파일을 읽고, 
        # 메타데이터에 파일 경로(source)를 자동으로 넣어줍니다.
        loader = DirectoryLoader(
            SOURCE_DIR, 
            glob="**/README.md", 
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True
        )
        return loader.load()

    def run(self):
        # 1. 문서 로드
        raw_docs = self.load_documents()
        if not raw_docs:
            logger.error("❌ 수집된 문서가 없습니다. 먼저 organize_docs.py를 실행했는지 확인하세요.")
            return

        # 2. 문서 분할 (Chunking)
        # 너무 긴 문서는 AI가 처리하기 힘들므로 적절한 크기로 자릅니다.
        final_docs = self.text_splitter.split_documents(raw_docs)
        logger.info(f"✂️ 총 {len(raw_docs)}개의 파일을 {len(final_docs)}개의 조각(Chunk)으로 분할했습니다.")

        # 3. 벡터 DB 저장 (ChromaDB)
        # PGVector나 Qdrant 사용 시 이 부분을 해당 클래스로 교체하면 됩니다.
        logger.info(f"📦 벡터 DB 생성 및 저장 중 (저장소: {CHROMA_PERSIST_DIR})...")
        vector_db = Chroma.from_documents(
            documents=final_docs,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        # 저장 확정 (Chroma v0.4+ 에서는 자동으로 되지만 명시적 확인)
        logger.info("✅ 벡터 DB 저장 완료! 이제 Dify 등에서 이 DB를 참조할 수 있습니다.")

if __name__ == "__main__":
    # 필요한 라이브러리 체크 및 안내
    try:
        import langchain
        import chromadb
    except ImportError:
        print("\n[알림] 실행을 위해 아래 라이브러리 설치가 필요합니다:")
        print("pip install langchain langchain-community chromadb unstructured markdown\n")
        exit(1)

    ingestor = DocIngestor()
    ingestor.run()
