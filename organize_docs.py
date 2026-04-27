import os
import asyncio
import shutil
import json
import logging
from pathlib import Path
from openai import AsyncOpenAI
import aiohttp

# ================= CONFIGURATION =================
# 1. 경로 설정 (Windows 경로 입력 시 r"" 사용 필수)
# 예: r"C:\Users\Name\Documents\platform-doc"
TARGET_DIR = r"./test_docs" 
BACKUP_DIR = r"./test_docs_backup" 

# 2. Ollama 설정 (분류용)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"  # 사용 중인 모델명으로 변경 (예: eeve-korean:10.8b)

# 3. 사내 API 설정 (README 생성용)
COMPANY_API_KEY = "your-api-key"
COMPANY_API_BASE = "your-api-endpoint" 
COMPANY_API_MODEL = "your-model-name"

# 4. 실행 옵션
DRY_RUN = False  # True일 경우 실제 이동/파일생성을 하지 않고 로그만 출력
# =================================================

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocOrganizer:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=COMPANY_API_KEY,
            base_url=COMPANY_API_BASE
        )
        # 기본 카테고리 후보 (AI가 이 중에서 고르거나 새로 제안함)
        self.categories = ["infra", "database", "tool", "monitoring", "api-gateway", "ci-cd", "security", "k8s-cluster"]

    async def call_ollama(self, prompt):
        """Ollama API를 호출하여 분류 결과를 가져옵니다."""
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_URL, json=payload, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"Ollama API Error: {resp.status}")
                        return None
                    result = await resp.json()
                    return json.loads(result['message']['content'])
        except Exception as e:
            logger.error(f"Ollama Connection Error: {e}")
            return None

    async def classify_folder(self, folder_name, files):
        """폴더명과 파일 목록을 기반으로 카테고리를 결정합니다."""
        prompt = f"""
        당신은 IT 인프라 분류 전문가입니다. 다음 폴더명과 파일 목록을 보고 가장 적절한 카테고리를 하나만 골라주세요.
        카테고리 후보: {", ".join(self.categories)}
        만약 후보에 적당한 것이 없다면 가장 직관적인 새 카테고리명을 제안해주세요.
        응답은 반드시 아래 JSON 형식으로만 해주세요:
        {{"category": "카테고리명"}}

        폴더명: {folder_name}
        파일 목록: {files[:20]}
        """
        res = await self.call_ollama(prompt)
        if res and "category" in res:
            return res["category"].lower().replace(" ", "-")
        return "etc"

    async def generate_readme(self, folder_path, category):
        """폴더 내 주요 파일을 분석하여 README.md 내용을 생성합니다."""
        context_files = ["Chart.yaml", "values.yaml", "deployment.yaml", "README.md", "README.txt"]
        content_snippet = ""
        
        # 파일 내용 수집
        for f_name in context_files:
            f_path = folder_path / f_name
            if f_path.exists() and f_path.is_file():
                try:
                    with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # 파일당 최대 2000자까지만 읽어 context 구성
                        content_snippet += f"\n--- File: {f_name} ---\n{f.read(2000)}\n"
                except Exception as e:
                    logger.warning(f"Could not read {f_name} in {folder_path}: {e}")

        if not content_snippet:
            content_snippet = "설정 파일이 없거나 내용을 읽을 수 없습니다. 폴더 이름과 구조를 기반으로 작성해주세요."

        prompt = f"""
        당신은 사내 인프라 문서를 정리하는 전문가입니다. 제공된 내용을 바탕으로 이 프로젝트의 'README.md'를 작성해주세요.
        이 문서는 나중에 RAG(검색 기반 생성) 시스템의 지식 베이스로 사용될 것이므로, 검색에 유리하도록 핵심 키워드와 기술적 맥락을 상세히 포함해야 합니다.

        [작성 가이드]
        1. 프로젝트 명칭 및 목적 (무엇을 위한 것인가?)
        2. 주요 기술 스택 (예: Kong, Helm, Kubernetes v1.x 등)
        3. 핵심 설정 요약 (중요한 엔드포인트나 옵션)
        4. RAG용 메타데이터 (문서 최상단에 YAML Front-matter 형식으로 작성)
           예:
           ---
           category: {category}
           tags: [tag1, tag2]
           version: v1.0.0
           ---

        [데이터 정보]
        - 카테고리: {category}
        - 현재 폴더명: {folder_path.name}
        - 수집된 내용:
        {content_snippet}
        
        응답은 Markdown 형식으로만 작성해주세요.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=COMPANY_API_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Company API Error: {e}")
            return f"# README.md\n\n자동 생성 실패 (오류: {str(e)})\n카테고리: {category}"

    async def process_folder(self, folder, target_path):
        """개별 폴더를 분류, 이동 및 문서화합니다."""
        try:
            files = [f.name for f in folder.iterdir()]
            category = await self.classify_folder(folder.name, files)
            
            new_parent = target_path / category
            dest_path = new_parent / folder.name
            
            logger.info(f"Processing: [{folder.name}] -> Category: [{category}]")

            if not DRY_RUN:
                new_parent.mkdir(exist_ok=True, parents=True)
                
                # 폴더 이동
                if not dest_path.exists():
                    shutil.move(str(folder), str(dest_path))
                else:
                    logger.warning(f"Destination already exists: {dest_path}. Skipping move.")
                    dest_path = folder # 이동 실패 시 현재 위치에서 README 생성 시도

                # README 생성
                readme_content = await self.generate_readme(dest_path, category)
                with open(dest_path / "README.md", "w", encoding="utf-8") as f:
                    f.write(readme_content)
                logger.info(f"   Successfully created README.md for {folder.name}")
        except Exception as e:
            logger.error(f"Error processing folder {folder.name}: {e}")

    async def run(self):
        target = Path(TARGET_DIR)
        if not target.exists():
            logger.error(f"Target directory does not exist: {TARGET_DIR}")
            return

        # 최상위 폴더 목록 (이미 카테고리화된 폴더 제외 로직 필요 시 추가)
        folders = [f for f in target.iterdir() if f.is_dir() and not f.name.startswith(('.', '_'))]
        
        if not folders:
            logger.info("No folders found to process.")
            return

        logger.info(f"Starting to process {len(folders)} folders...")
        
        # 병렬 처리를 원할 경우 asyncio.gather 사용 가능하나, 
        # API 레이트 리밋과 파일 I/O 안정성을 위해 순차 처리를 기본으로 합니다.
        for folder in folders:
            await self.process_folder(folder, target)
        
        logger.info("All tasks completed.")

if __name__ == "__main__":
    organizer = DocOrganizer()
    try:
        asyncio.run(organizer.run())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
