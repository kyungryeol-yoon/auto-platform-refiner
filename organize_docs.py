"""사내 platform-doc 자동 분류 및 README 생성 통합 스크립트.

- MODE = "hybrid" | "company" | "ollama" 로 동작 방식 선택
- 사내 API는 AsyncOpenAI(api_key, base_url, default_headers) 로 호출
- 분류는 (1) 룰 기반 화이트리스트 → (2) LLM + few-shot + JSON enum 검증 순서
"""

import asyncio
import csv
import datetime as _dt
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
MODE = "company"

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

# 재실행 옵션
WIPE_EXISTING_README = True      # process_folder 시작 시 README.md + README_original.md 둘 다 삭제.
                                  # → 완전 클린 상태에서 AI README 새로 생성. 평탄화 후 재실행에 권장.

# 산출물
CLASSIFICATION_LOG = "./classification_log.csv"   # 분류 결과 기록 (사후 검증용)
INDEX_FILE_NAME = "INDEX.md"                       # TARGET_DIR 최상단에 생성
DISCOVERED_CATEGORIES_FILE = "./.discovered_categories.json"
# → LLM 이 ALLOWED_CATEGORIES 밖에서 제안한 새 카테고리를 누적 저장.
#   다음 실행 때 prompt 에 주입돼 같은 종류 도구가 같은 카테고리로 수렴.

# 동작 옵션
ALLOW_NEW_CATEGORIES = True       # False → 미매칭 시 무조건 etc (엄격 모드)
# =================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _validate_headers(headers: dict | None, var_name: str) -> dict[str, str] | None:
    """헤더 dict 의 값이 모두 문자열인지 검증. 잘못된 값 발견 시 즉시 fail-fast."""
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
    "security",         # 인증/인가(SSO, LDAP, OIDC), 시크릿, 스캐너 포함
    "registry",         # Harbor, Nexus 등 컨테이너/아티팩트 레지스트리
    "ml-ai",            # Kubeflow, MLflow, Jupyter, Ray 등 ML/AI 플랫폼
    "documentation",    # Confluence, Bookstack, MediaWiki, Docusaurus 등
    "etc",
}

# ---------- 룰 기반 화이트리스트 (CNCF Landscape + 사내 자주 쓰는 도구) ----------
# 키: 도구명 (소문자), 값: 카테고리. 폴더명/Chart.yaml name 에서 word-boundary 매칭.
# 키워드는 다른 키워드의 부분문자열이 되지 않게 주의 (예: "nginx" 단독 키 X)
KNOWN_TOOLS: dict[str, str] = {
    # ---- api-gateway ----
    "kong": "api-gateway",
    "ambassador": "api-gateway",
    "traefik": "api-gateway",
    "apisix": "api-gateway",
    "nginx-ingress": "api-gateway",
    "ingress-nginx": "api-gateway",
    "haproxy": "api-gateway",
    "gloo": "api-gateway",
    "krakend": "api-gateway",
    "tyk": "api-gateway",
    "zuul": "api-gateway",
    "emissary": "api-gateway",
    "contour": "api-gateway",
    # ---- service-mesh ----
    "istio": "service-mesh",
    "linkerd": "service-mesh",
    "consul": "service-mesh",
    "envoy": "service-mesh",
    "kuma": "service-mesh",
    "kiali": "service-mesh",
    "open-service-mesh": "service-mesh",
    "osm": "service-mesh",
    # ---- ci-cd ----
    "argocd": "ci-cd",
    "argo-cd": "ci-cd",
    "argo-workflows": "ci-cd",
    "argo-rollouts": "ci-cd",
    "argo-events": "ci-cd",
    "argo": "ci-cd",                 # 위 변형들에 매칭 안 되는 "argo-*" 폴더 fallback
    "knative": "ci-cd",
    "knative-serving": "ci-cd",
    "kubevela": "ci-cd",
    "vela": "ci-cd",
    "jenkins": "ci-cd",
    "jenkins-x": "ci-cd",
    "tekton": "ci-cd",
    "flux": "ci-cd",
    "fluxcd": "ci-cd",
    "flux2": "ci-cd",
    "spinnaker": "ci-cd",
    "gitlab-runner": "ci-cd",
    "gitea": "ci-cd",
    "drone": "ci-cd",
    "woodpecker": "ci-cd",
    "concourse": "ci-cd",
    "harness": "ci-cd",
    "codefresh": "ci-cd",
    "circleci": "ci-cd",
    "buildkite": "ci-cd",
    "teamcity": "ci-cd",
    "bamboo": "ci-cd",
    "keptn": "ci-cd",
    "dagger": "ci-cd",
    "earthly": "ci-cd",
    "github-actions-runner": "ci-cd",
    "actions-runner-controller": "ci-cd",
    # ---- devops-tools (자동화/오케스트레이션) ----
    "n8n": "devops-tools",
    "airflow": "devops-tools",
    "rundeck": "devops-tools",
    "ansible": "devops-tools",
    "ansible-awx": "devops-tools",
    "awx": "devops-tools",
    "terraform": "devops-tools",
    "terragrunt": "devops-tools",
    "packer": "devops-tools",
    "nifi": "devops-tools",
    "temporal": "devops-tools",
    "prefect": "devops-tools",
    "dagster": "devops-tools",
    "mage": "devops-tools",
    "kestra": "devops-tools",
    "pulumi": "devops-tools",
    "crossplane": "devops-tools",
    "kustomize": "devops-tools",
    "helm": "devops-tools",
    "helmfile": "devops-tools",
    "skaffold": "devops-tools",
    "tilt": "devops-tools",
    "garden": "devops-tools",
    "okteto": "devops-tools",
    "devspace": "devops-tools",
    # ---- monitoring (메트릭/APM/트레이싱) ----
    "prometheus": "monitoring",
    "grafana": "monitoring",
    "thanos": "monitoring",
    "victoriametrics": "monitoring",
    "alertmanager": "monitoring",
    "jaeger": "monitoring",
    "zipkin": "monitoring",
    "opentelemetry": "monitoring",
    "otel-collector": "monitoring",
    "otel": "monitoring",
    "datadog": "monitoring",
    "newrelic": "monitoring",
    "dynatrace": "monitoring",
    "appdynamics": "monitoring",
    "instana": "monitoring",
    "lightstep": "monitoring",
    "honeycomb": "monitoring",
    "signoz": "monitoring",
    "sentry": "monitoring",
    "pixie": "monitoring",
    "kube-state-metrics": "monitoring",
    "node-exporter": "monitoring",
    "blackbox-exporter": "monitoring",
    "cadvisor": "monitoring",
    "robusta": "monitoring",
    "kubernetes-dashboard": "monitoring",
    "k9s": "monitoring",
    "lens": "monitoring",
    "pyroscope": "monitoring",
    "alloy": "monitoring",            # Grafana Alloy (OTel-based collector)
    "grafana-alloy": "monitoring",
    "kore-board": "monitoring",       # k8s 운영 보드 (사내 자주 쓰는 OSS)
    # 전통적인 인프라 모니터링
    "zabbix": "monitoring",
    "zabbix-agent": "monitoring",
    "zabbix-server": "monitoring",
    "zabbix-proxy": "monitoring",
    "nagios": "monitoring",
    "icinga": "monitoring",
    "icinga2": "monitoring",
    "centreon": "monitoring",
    "checkmk": "monitoring",
    "check-mk": "monitoring",
    "sensu": "monitoring",
    "cacti": "monitoring",
    "munin": "monitoring",
    "observium": "monitoring",
    "librenms": "monitoring",
    "netdata": "monitoring",
    # APM / 트레이싱
    "pinpoint": "monitoring",
    "skywalking": "monitoring",
    "elastic-apm": "monitoring",
    "scouter": "monitoring",
    "glowroot": "monitoring",
    "tempo": "monitoring",
    "mimir": "monitoring",
    "uptrace": "monitoring",
    "uptime-kuma": "monitoring",
    # ---- logging ----
    "elasticsearch": "logging",
    "kibana": "logging",
    "logstash": "logging",
    "fluentd": "logging",
    "fluent-bit": "logging",
    "fluentbit": "logging",
    "loki": "logging",
    "promtail": "logging",
    "opensearch": "logging",
    "opendistro": "logging",          # OpenDistro for Elasticsearch (AWS fork, OpenSearch 전신)
    "graylog": "logging",
    "splunk": "logging",
    "filebeat": "logging",
    "journalbeat": "logging",
    "metricbeat": "logging",
    "vector-log": "logging",
    # ---- database ----
    "postgres": "database",
    "postgresql": "database",
    "patroni": "database",
    "mysql": "database",
    "mariadb": "database",
    "mongodb": "database",
    "mongo": "database",
    "redis": "database",
    "valkey": "database",
    "keydb": "database",
    "cassandra": "database",
    "scylladb": "database",
    "cockroachdb": "database",
    "tidb": "database",
    "clickhouse": "database",
    "influxdb": "database",
    "timescaledb": "database",
    "questdb": "database",
    "oracle": "database",
    "mssql": "database",
    "sqlserver": "database",
    "couchdb": "database",
    "couchbase": "database",
    "neo4j": "database",
    "arangodb": "database",
    "dgraph": "database",
    "hbase": "database",
    "hive": "database",
    "druid": "database",
    "doris": "database",
    "starrocks": "database",
    "elasticsearch-db": "database",
    "milvus": "database",
    "weaviate": "database",
    "qdrant": "database",
    "chromadb": "database",
    "pgvector": "database",
    # ---- messaging ----
    "kafka": "messaging",
    "kafka-connect": "messaging",
    "strimzi": "messaging",
    "rabbitmq": "messaging",
    "nats": "messaging",
    "pulsar": "messaging",
    "activemq": "messaging",
    "redpanda": "messaging",
    "mqtt": "messaging",
    "mosquitto": "messaging",
    "emqx": "messaging",
    "vernemq": "messaging",
    "knative-eventing": "messaging",
    # ---- storage ----
    "minio": "storage",
    "ceph": "storage",
    "rook": "storage",
    "longhorn": "storage",
    "openebs": "storage",
    "velero": "storage",
    "kasten": "storage",
    "kopia": "storage",
    "restic": "storage",
    "stash": "storage",
    "nfs-provisioner": "storage",
    "glusterfs": "storage",
    "juicefs": "storage",
    "alluxio": "storage",
    "portworx": "storage",
    # CSI 드라이버 패밀리 (Container Storage Interface)
    "csi": "storage",
    "csi-driver": "storage",
    "csi-snapshotter": "storage",
    "csi-resizer": "storage",
    "csi-attacher": "storage",
    "csi-provisioner": "storage",
    "nfs-csi": "storage",
    "smb-csi": "storage",
    "aws-ebs-csi": "storage",
    "aws-efs-csi": "storage",
    "gcp-pd-csi": "storage",
    "azure-disk-csi": "storage",
    "vsphere-csi": "storage",
    "hostpath-csi": "storage",
    "storageclass": "storage",        # k8s 네이티브 리소스 정의 모음
    "storage-class": "storage",
    "persistent-volume": "storage",
    "pv-provisioner": "storage",
    # ---- networking ----
    "cilium": "networking",
    "calico": "networking",
    "flannel": "networking",
    "weave": "networking",
    "metallb": "networking",
    "coredns": "networking",
    "kube-router": "networking",
    "kube-proxy": "networking",
    "antrea": "networking",
    "multus": "networking",
    "submariner": "networking",
    "ovn": "networking",
    "kube-vip": "networking",
    "nginx-ingress-controller": "networking",
    "external-dns": "networking",
    "externaldns": "networking",
    "caddy": "networking",
    "aws-load-balancer-controller": "networking",
    "aws-lb-controller": "networking",
    "ingress-controller": "networking",
    # ---- security (인증/인가/시크릿/스캐너) ----
    "cert-manager": "security",
    "falco": "security",
    "kyverno": "security",
    "opa": "security",
    "gatekeeper": "security",
    "trivy": "security",
    "vault": "security",
    "oauth2-proxy": "security",
    "oauth2": "security",
    "oauth": "security",
    "oidc": "security",
    "oidc-proxy": "security",
    "saml": "security",
    "sso": "security",
    "ldap": "security",
    "openldap": "security",
    "freeipa": "security",
    "dex": "security",
    "keycloak": "security",
    "authentik": "security",
    "authelia": "security",
    "okta": "security",
    "auth0": "security",
    "pingfederate": "security",
    "sealed-secrets": "security",
    "external-secrets": "security",
    "sops": "security",
    "snyk": "security",
    "aqua": "security",
    "sysdig": "security",
    "twistlock": "security",
    "clair": "security",
    "anchore": "security",
    "kube-bench": "security",
    "kube-hunter": "security",
    "tetragon": "security",
    "starboard": "security",
    "spire": "security",
    "cert-bot": "security",
    "certbot": "security",
    "sonarqube": "security",
    "checkmarx": "security",
    # ---- registry ----
    "harbor": "registry",
    "nexus": "registry",
    "artifactory": "registry",
    "jfrog": "registry",
    "quay": "registry",
    "docker-registry": "registry",
    "distribution": "registry",
    # ---- k8s-cluster (클러스터 자체 운영 + 관리 플랫폼 + 컨테이너 런타임) ----
    "kubeadm": "k8s-cluster",
    "kubespray": "k8s-cluster",
    "rancher": "k8s-cluster",
    "rke": "k8s-cluster",
    "rke2": "k8s-cluster",
    "k3s": "k8s-cluster",
    "k0s": "k8s-cluster",
    "kops": "k8s-cluster",
    "kind": "k8s-cluster",
    "minikube": "k8s-cluster",
    "microk8s": "k8s-cluster",
    "talos": "k8s-cluster",
    "openshift": "k8s-cluster",
    "vcluster": "k8s-cluster",
    "kubefed": "k8s-cluster",
    "cluster-api": "k8s-cluster",
    "capi": "k8s-cluster",
    "etcd": "k8s-cluster",
    "kubelet": "k8s-cluster",
    "containerd": "k8s-cluster",
    "cri-o": "k8s-cluster",
    "crio": "k8s-cluster",
    "docker": "k8s-cluster",          # 컨테이너 런타임/엔진
    "dockerd": "k8s-cluster",
    "docker-compose": "k8s-cluster",
    "podman": "k8s-cluster",
    "kubesphere": "k8s-cluster",      # k8s 관리 플랫폼 UI
    "portainer": "k8s-cluster",
    "lens-desktop": "k8s-cluster",
    # ---- infrastructure (IaaS 레이어) ----
    "openstack": "infrastructure",
    "vmware": "infrastructure",
    "vsphere": "infrastructure",
    "ovirt": "infrastructure",
    "proxmox": "infrastructure",
    "hyperv": "infrastructure",
    "kvm": "infrastructure",
    "libvirt": "infrastructure",
    # ---- ml-ai (ML/AI 플랫폼) ----
    "kubeflow": "ml-ai",
    "kubeflow-pipelines": "ml-ai",
    "kfp": "ml-ai",
    "mlflow": "ml-ai",
    "ray": "ml-ai",
    "ray-cluster": "ml-ai",
    "kserve": "ml-ai",
    "kfserving": "ml-ai",
    "seldon": "ml-ai",
    "seldon-core": "ml-ai",
    "jupyter": "ml-ai",
    "jupyterhub": "ml-ai",
    "jupyter-notebook": "ml-ai",
    "jupyterlab": "ml-ai",
    "bentoml": "ml-ai",
    "triton": "ml-ai",
    "nvidia-triton": "ml-ai",
    "tritonserver": "ml-ai",
    "mlrun": "ml-ai",
    "feast": "ml-ai",
    "determined": "ml-ai",
    "polyaxon": "ml-ai",
    "nemo": "ml-ai",
    "dvc": "ml-ai",
    "huggingface": "ml-ai",
    "vllm": "ml-ai",
    "ollama-server": "ml-ai",
    "litellm": "ml-ai",
    "langfuse": "ml-ai",
    "tensorflow-serving": "ml-ai",
    "torchserve": "ml-ai",
    "ollama": "ml-ai",                # LLM 서빙 (※ 우리 코드의 fallback 모드 'ollama' 와는 무관)
    # ---- documentation (사내 위키/문서 시스템 + 이슈 트래커 + 협업) ----
    "confluence": "documentation",
    "bookstack": "documentation",
    "mediawiki": "documentation",
    "docusaurus": "documentation",
    "mkdocs": "documentation",
    "outline": "documentation",
    "wikijs": "documentation",
    "dokuwiki": "documentation",
    "gitbook": "documentation",
    "backstage": "documentation",   # 사내 service catalog
    # 이슈 트래커 / 협업
    "jira": "documentation",
    "redmine": "documentation",
    "mantis": "documentation",
    "mattermost": "documentation",
    "rocket-chat": "documentation",
    "rocketchat": "documentation",
    "discourse": "documentation",
    # 형상관리/SCM (CI/CD 인접이지만 도구 자체는 문서/협업으로)
    "gitlab": "ci-cd",
    "bitbucket": "ci-cd",
    "gerrit": "ci-cd",
    "github-enterprise": "ci-cd",
}


FEW_SHOT_EXAMPLES = """[분류 예시]
- 폴더명 "n8n-v1.0", 파일 ["deployment.yaml", "service.yaml"]
  → {"category": "devops-tools", "is_new": false, "reason": "n8n은 워크플로 자동화 도구"}
- 폴더명 "kong-v2.0.1", 파일 ["Chart.yaml", "values.yaml"]
  → {"category": "api-gateway", "is_new": false, "reason": "Kong은 대표적인 API Gateway"}
- 폴더명 "k8s-upgrade-v1.24-to-v1.26", 파일 ["upgrade.md"]
  → {"category": "k8s-cluster", "is_new": false, "reason": "클러스터 자체 업그레이드 가이드"}
- 폴더명 "chip-stress-test", 파일 ["scripts/", "results/"]
  → {"category": "load-testing", "is_new": true, "reason": "표준 카테고리에 부하/스트레스 테스트 없음 - 새 카테고리 제안"}
- 폴더명 "kore-board", 파일 ["values.yaml"]
  → {"category": "k8s-dashboard", "is_new": true, "reason": "k8s 운영 대시보드 UI - 기존 monitoring 보단 별도 분류가 검색에 유리"}
"""

# 새 카테고리 이름 검증: 영문 소문자로 시작, 소문자/숫자/하이픈만, 3~30자
_NEW_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9\-]{2,29}$")


def _build_classify_prompt(
    folder_name: str,
    files: list[str],
    context: str,
    discovered: set[str] | None = None,
) -> str:
    discovered = discovered or set()
    discovered_str = ", ".join(sorted(discovered)) if discovered else "(아직 없음)"
    allow_new_section = (
        "3. 위에 적합한 것이 정말 없으면, 새 카테고리를 제안할 수 있습니다.\n"
        "   - 형식: 영문 소문자 시작, 소문자/숫자/하이픈만, 3~30자 (예: load-testing, data-pipeline)\n"
        "   - 도구명 자체보다 의미 있는 분류명 (chip-stress → load-testing)\n"
        "   - 너무 좁지도 너무 넓지도 않게.\n"
        f"   - 이전에 자동 생성된 카테고리(있으면 우선 사용): {discovered_str}\n"
        if ALLOW_NEW_CATEGORIES else
        "3. 표준 카테고리 외의 값은 모두 etc 로 강등됩니다.\n"
    )
    return f"""당신은 CNCF Landscape와 IT 인프라 분류 전문가입니다.

[분류 규칙]
1. 우선 다음 표준 카테고리 중에서 가장 적합한 것을 선택하세요:
   {", ".join(sorted(ALLOWED_CATEGORIES))}
2. "k8s-cluster" 는 클러스터 자체 운영(kubeadm, etcd, CNI, CRI, upgrade)에만 사용.
   k8s 위에 deployment.yaml 로 배포되는 워크로드는 도구의 본질적 기능 카테고리로 분류.
{allow_new_section}4. 카테고리 매핑 힌트:
   - 인증/인가(SSO, LDAP, OAuth, OIDC, Keycloak)         → security
   - ML/AI 플랫폼(Kubeflow, MLflow, Jupyter, Ray, Ollama) → ml-ai
   - 사내 위키/이슈 트래커(Confluence, Jira, Bookstack)  → documentation
   - 이미지/아티팩트 저장소(Harbor, Nexus, Artifactory)  → registry
5. "etc" 는 진짜 인프라/플랫폼이 아닐 때만 (회의록, 개인 메모 등).
   도구/시스템은 항상 가장 가까운 카테고리를 선택하거나 새 카테고리를 제안하세요.
6. 분류 근거 우선순위: ① 폴더명 도구명 → ② Chart.yaml name/description
   → ③ README 첫 단락 → ④ 파일 구성.

{FEW_SHOT_EXAMPLES}

응답은 반드시 아래 JSON 형식 한 줄로만 답하세요:
{{"category": "<카테고리>", "is_new": <true|false>, "reason": "<한 줄 근거>"}}

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
            default_headers=_validate_headers(COMPANY_DEFAULT_HEADERS, "COMPANY_DEFAULT_HEADERS"),
        )
        # word-boundary 매칭용으로 키워드를 길이 내림차순 정렬해 둠
        self._known_tool_keys = sorted(KNOWN_TOOLS.keys(), key=len, reverse=True)
        # 분류 로그 (run 종료 시 CSV 로 flush)
        self._log_rows: list[dict[str, str]] = []
        # LLM 이 동적 제안한 카테고리 (영구 저장 → 다음 run 에 prompt 로 주입)
        self.discovered: set[str] = self._load_discovered_categories()

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
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = m.group(0) if m else raw
        try:
            obj = json.loads(payload)
            cat = str(obj.get("category", "etc")).lower().strip().replace(" ", "-")
            reason = str(obj.get("reason", "")).strip()
            is_new_flag = bool(obj.get("is_new", False))

            # 1) 표준 카테고리 또는 이미 발견된 새 카테고리는 그대로 수용
            if cat in ALLOWED_CATEGORIES or cat in self.discovered:
                return cat, f"llm:{reason}" if reason else "llm"

            # 2) ALLOW_NEW_CATEGORIES + 이름 검증 통과 시 새 카테고리로 채택
            if (
                ALLOW_NEW_CATEGORIES
                and cat != "etc"
                and _NEW_CATEGORY_RE.match(cat)
            ):
                self.discovered.add(cat)
                logger.info(f"   [new category] {cat} ← {reason or '(no reason)'}")
                return cat, f"llm-new:{reason}" if reason else "llm-new"

            # 3) 이름 검증 실패 → etc
            logger.warning(
                f"   LLM returned invalid category '{cat}' (is_new={is_new_flag}); falling back to etc"
            )
            return "etc", f"llm-unknown:{cat}"
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"   Failed to parse classification JSON ({e}): {raw[:200]!r}")
            return "etc", "llm-parse-error"

    def _load_discovered_categories(self) -> set[str]:
        p = Path(DISCOVERED_CATEGORIES_FILE)
        if not p.exists():
            return set()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return {str(c).lower() for c in data if _NEW_CATEGORY_RE.match(str(c).lower())}
        except Exception as e:
            logger.warning(f"Failed to load discovered categories: {e}")
            return set()

    def _save_discovered_categories(self) -> None:
        if not self.discovered:
            return
        try:
            Path(DISCOVERED_CATEGORIES_FILE).write_text(
                json.dumps(sorted(self.discovered), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                f"Discovered categories saved: {len(self.discovered)} entries → "
                f"{DISCOVERED_CATEGORIES_FILE}"
            )
        except Exception as e:
            logger.warning(f"Failed to save discovered categories: {e}")

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
        prompt = _build_classify_prompt(folder.name, files, context, self.discovered)
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
- **모든 설명문은 반드시 한국어로 작성**. 입력 자료가 영어/중국어/일본어여도
  의미를 한국어로 번역/의역해서 작성하세요.
  예외: 도구명·버전번호·포트·이미지명·설정키 같은 기술 식별자는 원문 유지.
- 전체 1200자 이내 (한글 기준). 절대 초과 금지.
- YAML 원문 복사 금지. 핵심 값(포트, 이미지, 엔드포인트)만 한국어로 한 줄씩 요약.
- 출력은 아래 템플릿을 그대로 따르세요. 섹션을 추가/생략하지 마세요.

[출력 템플릿]
---
category: {category}
project: <도구명 소문자>
version: <감지된 버전 또는 unknown>
location: {folder.as_posix()}
tags: [<쉼표로 3~5개, 영문 식별자 우선>]
---

# <도구명> <버전>

**요약**: <한 문장 한국어 요약. 이 프로젝트가 무엇이고 왜 쓰는지>

**검색 키워드**: <쉼표로 5개. 한국어 + 영문 식별자 혼합 가능>

## 기술 스택
- <기술명>: <한국어로 한 줄 설명>
- <기술명>: <한국어로 한 줄 설명>

## 핵심 설정
- <설정 항목>: <한국어로 의미와 값 설명>
- <설정 항목>: <한국어로 의미와 값 설명>

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
        category, source, dest_str, status = "etc", "init", "", "error"
        try:
            category, source = await self.classify_folder(folder)
            new_parent = target_path / category
            dest_path = new_parent / folder.name
            dest_str = dest_path.as_posix()
            logger.info(f"[{folder.name}] -> [{category}]  ({source})")

            if DRY_RUN:
                status = "dry-run"
                return

            new_parent.mkdir(exist_ok=True, parents=True)

            if dest_path.exists():
                logger.warning(f"   Destination exists: {dest_path} (skipping move)")
                working_path = folder
                status = "skip-existing"
            else:
                shutil.move(str(folder), str(dest_path))
                working_path = dest_path
                status = "ok"

            # README 정리 정책:
            #   WIPE_EXISTING_README=True  → 둘 다 삭제하고 새로 생성 (재실행 / 클린 슬레이트용)
            #   WIPE_EXISTING_README=False → 첫 실행 시 사용자 원본을 README_original.md 로 백업
            if WIPE_EXISTING_README:
                for fname in ("README.md", "README_original.md"):
                    p = working_path / fname
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception as e:
                            logger.warning(f"   Could not delete {p}: {e}")
            else:
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
            status = f"error:{type(e).__name__}"
        finally:
            self._log_rows.append({
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "folder": folder.name,
                "category": category,
                "source": source,
                "destination": dest_str,
                "status": status,
            })

    # ---------- Reporting ----------
    def _write_classification_log(self) -> None:
        if not self._log_rows:
            return
        path = Path(CLASSIFICATION_LOG)
        # 기존 파일에 append, 헤더는 새 파일일 때만
        write_header = not path.exists()
        try:
            with path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "folder", "category", "source", "destination", "status"],
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(self._log_rows)
            logger.info(f"Classification log written: {path} (+{len(self._log_rows)} rows)")
        except Exception as e:
            logger.warning(f"Failed to write classification log: {e}")

    @staticmethod
    def _parse_front_matter(text: str) -> dict[str, str]:
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
        if not m:
            return {}
        meta: dict[str, str] = {}
        for line in m.group(1).splitlines():
            mm = re.match(r"^\s*([\w\-]+)\s*:\s*(.+?)\s*$", line)
            if mm:
                meta[mm.group(1).strip()] = mm.group(2).strip().strip("\"'")
        return meta

    def _write_index_md(self, target: Path) -> None:
        """TARGET_DIR/INDEX.md 생성 — 전체 프로젝트 카탈로그."""
        valid_cats = ALLOWED_CATEGORIES | self.discovered
        entries: list[tuple[str, str, str, str]] = []  # (category, project, version, location)
        for cat_dir in sorted(target.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.lower() not in valid_cats:
                continue
            for proj_dir in sorted(cat_dir.iterdir()):
                if not proj_dir.is_dir():
                    continue
                meta = {}
                readme = proj_dir / "README.md"
                if readme.exists():
                    try:
                        meta = self._parse_front_matter(readme.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        pass
                entries.append((
                    cat_dir.name,
                    meta.get("project", proj_dir.name),
                    meta.get("version", "unknown"),
                    f"{cat_dir.name}/{proj_dir.name}",
                ))

        if not entries:
            logger.info("No categorized entries found, skipping INDEX.md")
            return

        # 카테고리별 카운트
        by_cat: dict[str, int] = {}
        for cat, _, _, _ in entries:
            by_cat[cat] = by_cat.get(cat, 0) + 1

        lines: list[str] = []
        lines.append("# Platform Doc Catalog")
        lines.append("")
        lines.append(f"_Generated: {_dt.date.today().isoformat()}  ·  "
                     f"{len(entries)} projects across {len(by_cat)} categories_")
        lines.append("")
        lines.append("## Categories")
        for cat in sorted(by_cat):
            lines.append(f"- **{cat}** ({by_cat[cat]})")
        lines.append("")
        lines.append("## Projects")
        lines.append("")
        lines.append("| Category | Project | Version | Location |")
        lines.append("|---|---|---|---|")
        for cat, proj, ver, loc in entries:
            lines.append(f"| {cat} | {proj} | {ver} | `{loc}` |")
        lines.append("")

        index_path = target / INDEX_FILE_NAME
        try:
            index_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info(f"INDEX written: {index_path} ({len(entries)} entries)")
        except Exception as e:
            logger.warning(f"Failed to write INDEX.md: {e}")

    async def run(self) -> None:
        target = Path(TARGET_DIR)
        if not target.exists():
            logger.error(f"Target directory does not exist: {TARGET_DIR}")
            return

        # 이미 카테고리 폴더로 이동된 항목은 건너뜀 (표준 + 이전에 발견한 카테고리 모두)
        valid_cats = ALLOWED_CATEGORIES | self.discovered
        folders = [
            f for f in target.iterdir()
            if f.is_dir()
            and not f.name.startswith((".", "_"))
            and f.name.lower() not in valid_cats
        ]
        if not folders:
            logger.info("No folders to process.")
            return

        logger.info(f"Starting (mode={self.mode}) — {len(folders)} folders")
        for folder in folders:
            await self.process_folder(folder, target)

        # 산출물: 분류 로그(CSV) + 최상위 카탈로그(INDEX.md) + 발견 카테고리 영속화
        self._write_classification_log()
        self._save_discovered_categories()
        self._write_index_md(target)
        logger.info("All tasks completed.")


if __name__ == "__main__":
    organizer = DocOrganizer(mode=MODE)
    try:
        asyncio.run(organizer.run())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
