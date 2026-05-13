"""사내 platform-doc 자동 분류 및 README 생성 통합 스크립트.

- MODE = "hybrid" | "company" | "ollama" 로 동작 방식 선택
- 사내 API는 AsyncOpenAI(api_key, base_url, default_headers) 로 호출
- 분류는 (1) 룰 기반 화이트리스트 → (2) LLM + few-shot + JSON enum 검증 순서
"""

import asyncio
import json
import logging
import re
import shutil
from pathlib import Path

import aiohttp
from openai import AsyncOpenAI

# ================= CONFIGURATION =================
TARGET_DIR = r"./test_docs"

# 동작 모드
#   "hybrid"  : 사내 API 우선, 실패 시 Ollama
#   "company" : 사내 API 만 사용
#   "ollama"  : 로컬 Ollama 만 사용
MODE = "hybrid"

# 사내 API (OpenAI 호환 게이트웨이 가정)
COMPANY_API_KEY = "your-api-key"
COMPANY_API_BASE = "your-api-endpoint"          # 예: "https://llm.intra.company.com/v1"
COMPANY_API_MODEL = "gpt-oss"
COMPANY_DEFAULT_HEADERS = {
    # 사내 게이트웨이가 요구하는 헤더가 있다면 여기에 추가
    # 예) "X-Tenant-Id": "platform-team",
    #     "X-Project":   "doc-refiner",
}

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"

# 옵션
DRY_RUN = False
LLM_TEMPERATURE = 0.1
MAX_README_TOKENS = 600          # LLM 측 토큰 상한 (대략 한글 1200~1500자)
MAX_README_CHARS = 2000          # 디스크 저장 직전 강제 truncate (LLM 이 무시할 때 대비)
CLASSIFY_TIMEOUT = 30
README_TIMEOUT = 60
# =================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------- 카테고리 enum ----------
# LLM 이 이 집합 밖의 값을 내면 "etc" 로 강등됨.
ALLOWED_CATEGORIES = {
    "infrastructure",   # OpenStack, VMware, bare-metal 같은 IaaS 레이어
    "k8s-cluster",      # 클러스터 자체 (kubeadm, etcd, CNI, CRI, upgrade)
    "api-gateway",
    "service-mesh",
    "ci-cd",
    "devops-tools",     # n8n, ansible, terraform 등 자동화 도구
    "monitoring",
    "logging",
    "database",
    "messaging",
    "storage",
    "networking",
    "security",
    "registry",         # Harbor, Nexus 등 컨테이너/아티팩트 레지스트리
    "etc",
}

# ---------- 룰 기반 화이트리스트 (CNCF Landscape + 사내 자주 쓰는 도구) ----------
# 키: 도구명 (소문자), 값: 카테고리. 폴더명/Chart.yaml name 에서 word-boundary 매칭.
KNOWN_TOOLS: dict[str, str] = {
    # api-gateway
    "kong": "api-gateway",
    "ambassador": "api-gateway",
    "traefik": "api-gateway",
    "apisix": "api-gateway",
    "nginx-ingress": "api-gateway",
    "ingress-nginx": "api-gateway",
    "haproxy": "api-gateway",
    # service-mesh
    "istio": "service-mesh",
    "linkerd": "service-mesh",
    "consul": "service-mesh",
    # ci-cd
    "argocd": "ci-cd",
    "argo-cd": "ci-cd",
    "argo-workflows": "ci-cd",
    "argo-rollouts": "ci-cd",
    "jenkins": "ci-cd",
    "tekton": "ci-cd",
    "flux": "ci-cd",
    "fluxcd": "ci-cd",
    "spinnaker": "ci-cd",
    "gitlab-runner": "ci-cd",
    "gitea": "ci-cd",
    # devops-tools
    "n8n": "devops-tools",
    "airflow": "devops-tools",
    "rundeck": "devops-tools",
    "ansible": "devops-tools",
    "terraform": "devops-tools",
    "packer": "devops-tools",
    "nifi": "devops-tools",
    # monitoring
    "prometheus": "monitoring",
    "grafana": "monitoring",
    "thanos": "monitoring",
    "victoriametrics": "monitoring",
    "alertmanager": "monitoring",
    "jaeger": "monitoring",
    "zipkin": "monitoring",
    "opentelemetry": "monitoring",
    "otel": "monitoring",
    "datadog": "monitoring",
    # logging
    "elasticsearch": "logging",
    "kibana": "logging",
    "logstash": "logging",
    "fluentd": "logging",
    "fluent-bit": "logging",
    "fluentbit": "logging",
    "loki": "logging",
    "promtail": "logging",
    "opensearch": "logging",
    # database
    "postgres": "database",
    "postgresql": "database",
    "mysql": "database",
    "mariadb": "database",
    "mongodb": "database",
    "mongo": "database",
    "redis": "database",
    "cassandra": "database",
    "cockroachdb": "database",
    "tidb": "database",
    "clickhouse": "database",
    "influxdb": "database",
    # messaging
    "kafka": "messaging",
    "rabbitmq": "messaging",
    "nats": "messaging",
    "pulsar": "messaging",
    "activemq": "messaging",
    # storage
    "minio": "storage",
    "ceph": "storage",
    "rook": "storage",
    "longhorn": "storage",
    "openebs": "storage",
    "velero": "storage",
    # networking
    "cilium": "networking",
    "calico": "networking",
    "flannel": "networking",
    "metallb": "networking",
    "coredns": "networking",
    # security
    "cert-manager": "security",
    "falco": "security",
    "kyverno": "security",
    "opa": "security",
    "gatekeeper": "security",
    "trivy": "security",
    "vault": "security",
    # registry
    "harbor": "registry",
    "nexus": "registry",
    "artifactory": "registry",
    # k8s-cluster (클러스터 운영 자체에 한정)
    "kubeadm": "k8s-cluster",
    "kubespray": "k8s-cluster",
    "rancher": "k8s-cluster",
    "k3s": "k8s-cluster",
    "rke": "k8s-cluster",
    "etcd": "k8s-cluster",
    "kubelet": "k8s-cluster",
    # infrastructure
    "openstack": "infrastructure",
    "vmware": "infrastructure",
}


FEW_SHOT_EXAMPLES = """[분류 예시]
- 폴더명 "n8n-v1.0", 파일 ["deployment.yaml", "service.yaml"]
  → {"category": "devops-tools", "reason": "n8n은 워크플로 자동화 도구 (deployment.yaml은 단지 k8s 배포 방식)"}
- 폴더명 "kong-v2.0.1", 파일 ["Chart.yaml", "values.yaml"]
  → {"category": "api-gateway", "reason": "Kong은 대표적인 API Gateway"}
- 폴더명 "k8s-upgrade-v1.24-to-v1.26", 파일 ["upgrade.md"]
  → {"category": "k8s-cluster", "reason": "클러스터 자체 업그레이드 가이드"}
- 폴더명 "prometheus-stack", 파일 ["values.yaml"]
  → {"category": "monitoring", "reason": "Prometheus 모니터링 스택"}
- 폴더명 "harbor-registry", 파일 ["values.yaml"]
  → {"category": "registry", "reason": "Harbor는 컨테이너 이미지 레지스트리"}
"""


def _build_classify_prompt(folder_name: str, files: list[str], context: str) -> str:
    return f"""당신은 CNCF Landscape와 IT 인프라 분류 전문가입니다.

[분류 규칙]
1. 반드시 다음 카테고리 중 하나만 선택하세요 (그 외 값은 거부됩니다):
   {", ".join(sorted(ALLOWED_CATEGORIES))}
2. "k8s-cluster" 카테고리는 클러스터 자체 운영(kubeadm, etcd, CNI, CRI, upgrade)에만 사용합니다.
   k8s 위에 deployment.yaml 로 배포되는 워크로드(n8n, kong, grafana 등)는
   해당 도구의 본질적 기능 카테고리로 분류하세요.
3. 분류 근거 우선순위: ① 폴더명에 포함된 도구명 → ② Chart.yaml 의 name/description
   → ③ README 첫 단락 → ④ 파일 구성.

{FEW_SHOT_EXAMPLES}

응답은 반드시 아래 JSON 형식 한 줄로만 답하세요:
{{"category": "<카테고리>", "reason": "<한 줄 근거>"}}

[데이터]
- 폴더명: {folder_name}
- 파일 목록(최대 30개): {files[:30]}
- 설정 파일 발췌:
{context}
"""


class DocOrganizer:
    def __init__(self, mode: str = MODE):
        if mode not in {"hybrid", "company", "ollama"}:
            raise ValueError(f"Unknown MODE: {mode}")
        self.mode = mode
        self.client = AsyncOpenAI(
            api_key=COMPANY_API_KEY,
            base_url=COMPANY_API_BASE,
            default_headers=COMPANY_DEFAULT_HEADERS or None,
        )
        # word-boundary 매칭용으로 키워드를 길이 내림차순 정렬해 둠
        self._known_tool_keys = sorted(KNOWN_TOOLS.keys(), key=len, reverse=True)

    # ---------- LLM callers ----------
    async def _call_company(
        self,
        prompt: str,
        use_json: bool,
        max_tokens: int | None,
        timeout: int,
    ) -> str | None:
        kwargs: dict = {
            "model": COMPANY_API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "timeout": timeout,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except TypeError:
            # 사내 게이트웨이가 response_format 을 지원하지 않을 수 있음 → 재시도
            kwargs.pop("response_format", None)
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                logger.warning(f"Company API failed (retry): {e}")
                return None
        except Exception as e:
            logger.warning(f"Company API failed: {e}")
            return None

    async def _call_ollama(self, prompt: str, use_json: bool, timeout: int) -> str | None:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if use_json:
            payload["format"] = "json"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_URL, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.error(f"Ollama API HTTP {resp.status}")
                        return None
                    result = await resp.json()
                    return result["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None

    async def call_llm(
        self,
        prompt: str,
        use_json: bool = False,
        max_tokens: int | None = None,
        timeout: int = 60,
    ) -> str | None:
        if self.mode == "ollama":
            return await self._call_ollama(prompt, use_json, timeout)
        result = await self._call_company(prompt, use_json, max_tokens, timeout)
        if result is not None:
            return result
        if self.mode == "hybrid":
            logger.info("   Falling back to Ollama...")
            return await self._call_ollama(prompt, use_json, timeout)
        return None

    # ---------- Classification ----------
    def _match_known_tool(self, text: str) -> str | None:
        text_lc = text.lower()
        for kw in self._known_tool_keys:
            # word-boundary: 영문/숫자/밑줄 외 문자로 둘러싸이거나 문자열 끝
            pattern = rf"(?:^|[^a-z0-9]){re.escape(kw)}(?:[^a-z0-9]|$)"
            if re.search(pattern, text_lc):
                return KNOWN_TOOLS[kw]
        return None

    def _read_chart_name(self, folder: Path) -> str | None:
        for cname in ("Chart.yaml", "Chart.yml"):
            cpath = folder / cname
            if not cpath.exists():
                continue
            try:
                for line in cpath.read_text(encoding="utf-8", errors="ignore").splitlines():
                    m = re.match(r"^\s*name\s*:\s*['\"]?([\w\-\.]+)", line)
                    if m:
                        return m.group(1)
            except Exception:
                continue
        return None

    def rule_based_classify(self, folder: Path) -> tuple[str, str] | None:
        """폴더명 또는 Chart.yaml 의 name 으로 화이트리스트 매칭."""
        hit = self._match_known_tool(folder.name)
        if hit:
            return hit, f"rule:folder-name~{folder.name}"
        chart_name = self._read_chart_name(folder)
        if chart_name:
            hit = self._match_known_tool(chart_name)
            if hit:
                return hit, f"rule:Chart.yaml.name~{chart_name}"
        return None

    def _gather_classification_context(self, folder: Path) -> str:
        """분류용 컨텍스트: Chart.yaml/values.yaml/README 첫 부분만 발췌."""
        parts: list[str] = []
        for fname in ("Chart.yaml", "Chart.yml", "values.yaml", "README.md", "README.txt"):
            fpath = folder / fname
            if fpath.exists() and fpath.is_file():
                try:
                    snippet = fpath.read_text(encoding="utf-8", errors="ignore")[:600]
                    parts.append(f"--- {fname} ---\n{snippet}")
                except Exception:
                    continue
        return "\n".join(parts) if parts else "(no config files)"

    def _parse_category(self, raw: str | None) -> tuple[str, str]:
        if not raw:
            return "etc", "llm-empty-response"
        # JSON 객체만 추출 (모델이 코드펜스/잡담을 붙이는 경우 대비)
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = m.group(0) if m else raw
        try:
            obj = json.loads(payload)
            cat = str(obj.get("category", "etc")).lower().strip().replace(" ", "-")
            reason = str(obj.get("reason", "")).strip()
            if cat not in ALLOWED_CATEGORIES:
                logger.warning(f"   LLM returned unknown category '{cat}', falling back to etc")
                return "etc", f"llm-unknown:{cat}"
            return cat, f"llm:{reason}" if reason else "llm"
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"   Failed to parse classification JSON ({e}): {raw[:200]!r}")
            return "etc", "llm-parse-error"

    async def classify_folder(self, folder: Path) -> tuple[str, str]:
        # 1) Rule-based 우선
        rule = self.rule_based_classify(folder)
        if rule:
            return rule

        # 2) LLM (풍부한 컨텍스트 + few-shot + JSON enum 검증)
        try:
            files = [f.name for f in folder.iterdir()]
        except Exception as e:
            logger.warning(f"   Cannot list {folder}: {e}")
            files = []
        context = self._gather_classification_context(folder)
        prompt = _build_classify_prompt(folder.name, files, context)
        raw = await self.call_llm(prompt, use_json=True, timeout=CLASSIFY_TIMEOUT)
        return self._parse_category(raw)

    # ---------- README generation ----------
    @staticmethod
    def _truncate_readme(text: str, max_chars: int = MAX_README_CHARS) -> str:
        """LLM 이 길이 제약을 무시할 때를 대비한 안전망 (단어 경계로 자르기)."""
        if len(text) <= max_chars:
            return text
        head = text[:max_chars]
        # 줄 단위 경계에서 자르기
        nl = head.rfind("\n")
        if nl > max_chars * 0.7:
            head = head[:nl]
        return head.rstrip() + "\n\n<!-- truncated -->\n"

    async def generate_readme(self, folder: Path, category: str) -> str:
        snippets: list[str] = []
        for fname in ("Chart.yaml", "values.yaml", "deployment.yaml", "README.md", "README.txt"):
            fp = folder / fname
            if fp.exists() and fp.is_file():
                try:
                    snippets.append(
                        f"--- {fname} ---\n"
                        f"{fp.read_text(encoding='utf-8', errors='ignore')[:1200]}"
                    )
                except Exception:
                    continue
        content_snippet = "\n".join(snippets) or "(설정 파일 없음)"

        # RAG 친화: Front-matter 는 ingest_docs.py 가 파싱할 수 있는 정확한 형식으로 강제.
        # 본문은 임베딩 대상이 되므로 짧고 키워드 밀도가 높아야 함.
        prompt = f"""당신은 사내 인프라 문서화 전문가입니다.
아래 프로젝트의 README.md 를 RAG 검색용으로 작성하세요.

[엄수 사항]
- 전체 1200자 이내 (한글 기준). 절대 초과 금지.
- YAML 원문 복사 금지. 핵심 값(포트, 이미지, 엔드포인트)만 한 줄씩 요약.
- 출력은 아래 템플릿을 그대로 따르세요. 섹션을 추가/생략하지 마세요.

[출력 템플릿]
---
category: {category}
project: <도구명 소문자>
version: <감지된 버전 또는 unknown>
location: {folder.as_posix()}
tags: [<쉼표로 3~5개>]
---

# <도구명> <버전>

**Summary**: <한 문장 한국어 요약>

**Keywords**: <쉼표로 5개의 검색 키워드>

## Stack
- <핵심 기술 1>
- <핵심 기술 2>

## Config
- <중요 설정값 1>
- <중요 설정값 2>

[입력 데이터]
- 폴더명: {folder.name}
- 카테고리: {category}
- 설정 발췌:
{content_snippet}
"""
        result = await self.call_llm(
            prompt,
            use_json=False,
            max_tokens=MAX_README_TOKENS,
            timeout=README_TIMEOUT,
        )
        if result:
            return self._truncate_readme(result)
        return (
            f"---\ncategory: {category}\nproject: {folder.name}\n"
            f"version: unknown\nlocation: {folder.as_posix()}\ntags: []\n---\n"
            f"# {folder.name}\n\n**Summary**: 자동 생성 실패 — 수동 작성 필요.\n"
        )

    # ---------- Processing ----------
    async def process_folder(self, folder: Path, target_path: Path) -> None:
        try:
            category, source = await self.classify_folder(folder)
            new_parent = target_path / category
            dest_path = new_parent / folder.name
            logger.info(f"[{folder.name}] -> [{category}]  ({source})")

            if DRY_RUN:
                return

            new_parent.mkdir(exist_ok=True, parents=True)

            if dest_path.exists():
                logger.warning(f"   Destination exists: {dest_path} (skipping move)")
                working_path = folder
            else:
                shutil.move(str(folder), str(dest_path))
                working_path = dest_path

            # 기존 README 백업 (모든 모드에서 일관되게)
            existing = working_path / "README.md"
            if existing.exists():
                backup = working_path / "README_original.md"
                if not backup.exists():
                    existing.rename(backup)

            readme = await self.generate_readme(working_path, category)
            (working_path / "README.md").write_text(readme, encoding="utf-8")
            logger.info(f"   README.md created at {working_path}")
        except Exception as e:
            logger.error(f"Error processing {folder.name}: {e}")

    async def run(self) -> None:
        target = Path(TARGET_DIR)
        if not target.exists():
            logger.error(f"Target directory does not exist: {TARGET_DIR}")
            return

        # 이미 카테고리 폴더로 이동된 항목은 건너뜀
        folders = [
            f for f in target.iterdir()
            if f.is_dir()
            and not f.name.startswith((".", "_"))
            and f.name.lower() not in ALLOWED_CATEGORIES
        ]
        if not folders:
            logger.info("No folders to process.")
            return

        logger.info(f"Starting (mode={self.mode}) — {len(folders)} folders")
        for folder in folders:
            await self.process_folder(folder, target)
        logger.info("All tasks completed.")


if __name__ == "__main__":
    organizer = DocOrganizer(mode=MODE)
    try:
        asyncio.run(organizer.run())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
