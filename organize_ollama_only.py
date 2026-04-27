import os
import asyncio
import shutil
import json
import logging
import re
from pathlib import Path
import aiohttp

# ================= CONFIGURATION =================
TARGET_DIR = r"./test_docs" 
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3" # 또는 "eeve-korean:10.8b"
DRY_RUN = False
# =================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaOrganizer:
    async def call_ollama(self, prompt, use_json=True):
        payload = {"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
        if use_json: payload["format"] = "json"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_URL, json=payload, timeout=60) as resp:
                    result = await resp.json()
                    content = result['message']['content']
                    return json.loads(content) if use_json else content
        except Exception as e:
            logger.error(f"Ollama Error: {e}")
            return None

    async def run(self):
        target = Path(TARGET_DIR)
        folders = [f for f in target.iterdir() if f.is_dir() and not f.name.startswith(('.', '_'))]
        
        for folder in folders:
            # 1. 분류
            version_match = re.search(r'-v?(\d+\.\d+\.\d+.*)', folder.name)
            v_hint = version_match.group(1) if version_match else "unknown"
            
            res = await self.call_ollama(f"분류 카테고리(infra, tool, database 등)를 JSON으로 답해줘. 폴더명: {folder.name}", True)
            category = res.get("category", "etc").lower() if res else "etc"
            
            # 2. README 생성용 컨텍스트 수집
            context = ""
            for f_path in folder.glob("*"):
                if f_path.suffix in ['.yaml', '.yml', '.md'] and f_path.name.lower() != "readme.md":
                    with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                        context += f"\n-- {f_path.name} --\n{f.read(1000)}\n"

            # 3. README 생성
            readme = await self.call_ollama(f"다음 인프라 설정을 요약해서 README.md 마크다운을 작성해줘. 내용: {context[:3000]}", False)
            
            # 4. 실행
            if not DRY_RUN:
                dest = target / category / folder.name
                (target / category).mkdir(exist_ok=True, parents=True)
                if (folder / "README.md").exists(): (folder / "README.md").rename(folder / "README_original.md")
                with open(folder / "README.md", "w", encoding="utf-8") as f: f.write(readme)
                shutil.move(str(folder), str(dest))
                logger.info(f"Done: {dest}")

if __name__ == "__main__":
    asyncio.run(OllamaOrganizer().run())
