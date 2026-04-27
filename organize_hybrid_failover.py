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

class HybridOrganizer:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=COMPANY_API_KEY, base_url=COMPANY_API_BASE)

    async def call_ollama(self, prompt, use_json=True):
        payload = {"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
        if use_json: payload["format"] = "json"
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_URL, json=payload) as resp:
                res = await resp.json()
                content = res['message']['content']
                return json.loads(content) if use_json else content

    async def generate_readme(self, context):
        try:
            logger.info("   Trying Company API...")
            resp = await self.client.chat.completions.create(
                model=COMPANY_API_MODEL,
                messages=[{"role": "user", "content": f"Write README.md for:\n{context}"}],
                timeout=15
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"   Company API failed ({e}), falling back to Ollama...")
            return await self.call_ollama(f"Write README.md for this config:\n{context}", False)

    async def run(self):
        target = Path(TARGET_DIR)
        for folder in [f for f in target.iterdir() if f.is_dir() and not f.name.startswith(('.', '_'))]:
            # 1. Classify (Local is faster for this)
            res = await self.call_ollama(f"Category for {folder.name}", True)
            category = res.get("category", "etc").lower()
            
            # 2. Gather Info
            context = f"Project: {folder.name}\n"
            for f_path in list(folder.glob("*.yaml")) + list(folder.glob("*.md")):
                if f_path.name.lower() != "readme.md":
                    context += f"\n--- {f_path.name} ---\n{f_path.read_text(encoding='utf-8', errors='ignore')[:1000]}\n"

            # 3. Generate & Move
            readme = await self.generate_readme(context)
            dest = target / category / folder.name
            (target / category).mkdir(exist_ok=True, parents=True)
            with open(folder / "README.md", "w", encoding="utf-8") as f: f.write(readme)
            shutil.move(str(folder), str(dest))
            logger.info(f"Done: {dest}")

if __name__ == "__main__":
    asyncio.run(HybridOrganizer().run())
