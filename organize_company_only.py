import os
import asyncio
import shutil
import json
import logging
from pathlib import Path
from openai import AsyncOpenAI
import aiohttp

# ================= CONFIGURATION =================
TARGET_DIR = r"./test_docs" 
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"
COMPANY_API_KEY = "your-api-key"
COMPANY_API_BASE = "your-api-endpoint" 
COMPANY_API_MODEL = "your-model-name"
# =================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyOrganizer:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=COMPANY_API_KEY, base_url=COMPANY_API_BASE)

    async def classify(self, name):
        payload = {"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": f"Category for {name} (json: {{'category': '...'}})"}], "stream": False, "format": "json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_URL, json=payload) as resp:
                res = await resp.json()
                return json.loads(res['message']['content']).get("category", "etc").lower()

    async def generate_readme(self, context):
        resp = await self.client.chat.completions.create(
            model=COMPANY_API_MODEL,
            messages=[{"role": "user", "content": f"Create a detailed RAG-friendly README.md for this k8s project:\n{context}"}]
        )
        return resp.choices[0].message.content

    async def run(self):
        target = Path(TARGET_DIR)
        for folder in [f for f in target.iterdir() if f.is_dir() and not f.name.startswith(('.', '_'))]:
            category = await self.classify(folder.name)
            context = ""
            for f_path in list(folder.glob("*.yaml")) + list(folder.glob("*.md")):
                if f_path.name.lower() != "readme.md":
                    context += f"\nFile {f_path.name}:\n{f_path.read_text(encoding='utf-8', errors='ignore')[:1000]}"
            
            readme = await self.generate_readme(context)
            
            # Execute
            dest = target / category / folder.name
            (target / category).mkdir(exist_ok=True, parents=True)
            with open(folder / "README.md", "w", encoding="utf-8") as f: f.write(readme)
            shutil.move(str(folder), str(dest))
            logger.info(f"Organized: {dest}")

if __name__ == "__main__":
    asyncio.run(CompanyOrganizer().run())
