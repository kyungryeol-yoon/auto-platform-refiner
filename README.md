# Platform-Doc 자동 분류 및 문서화 가이드

본 프로젝트는 사내망(`platform-doc`)에 5년간 축적된 방대한 인프라 및 도구 관련 설정 파일(Helm, YAML, Markdown)을 AI(Ollama & 사내 LLM)를 활용해 체계적으로 분류하고 정리하는 자동화 솔루션을 제공합니다.

## 📋 개요
- **목표:** 수백 개의 폴더를 카테고리별로 자동 이동하고, 각 프로젝트의 기술 요약을 담은 `README.md`를 생성하여 RAG(검색) 시스템의 기반 마련.
- **환경:** Windows 11, Python 3.10+, Ollama(로컬), 사내 전용 LLM API.

## 📂 제공되는 스크립트 (시나리오별 선택)

상황에 맞게 가장 적절한 파일을 선택하여 실행하세요.

| 파일명 | 권장 상황 | 주요 기능 |
| :--- | :--- | :--- |
| **`organize_hybrid_failover.py`** | **실무 메인 권장** | 사내 API 우선 시도, 실패 시 로컬 Ollama로 자동 전환 (중단 없는 작업 가능) |
| **`organize_ollama_only.py`** | 폐쇄망/보안 강조 | 100% 로컬(Ollama) 자원만 사용하여 외부망/사내망 API 없이 작업 |
| **`organize_company_only.py`** | 고성능 문서화 | 사내 API(GPT-4급 등)의 성능을 극대화하여 가장 정교한 README 생성 |
| **`organize_docs.py`** | 초기 테스트 | 기본적인 분류 및 이동 기능 확인용 스크립트 |

## 🛠️ 설치 및 준비 사항

1. **파이썬 라이브러리 설치:**
   ```powershell
   pip install openai aiohttp
   ```

2. **Ollama 실행:**
   - 로컬에 Ollama가 구동 중이어야 하며, `llama3` 또는 `eeve-korean:10.8b` 모델이 설치되어 있어야 합니다.
   - 명령어: `ollama run llama3`

3. **스크립트 설정 수정:**
   각 스크립트 상단의 `CONFIGURATION` 섹션을 본인의 환경에 맞게 수정하세요.
   - `TARGET_DIR`: 정리 대상 폴더 경로 (예: `r"C:\workspace\platform-doc"`)
   - `COMPANY_API_KEY`: 사내 API 키
   - `COMPANY_API_BASE`: 사내 API 엔드포인트 URL

## 🚀 실행 단계

### 1단계: 폴더명 규칙 정비 (권장)
AI의 정확도를 높이기 위해 정리할 폴더들을 아래와 같은 형식으로 미리 변경해 두는 것이 좋습니다.
- 예: `kong-v2.0.1`, `grafana-v9.5.2`, `k8s-upgrade-v1.24`

### 2단계: 자동화 스크립트 실행
```powershell
# 하이브리드 모드 실행 예시
python organize_hybrid_failover.py
```

### 3단계: 결과 확인 및 백업
- 각 프로젝트 폴더 안에 `README.md`가 생성되었는지 확인합니다.
- 기존에 있던 수동 작성 `README.md`는 `README_original.md`로 안전하게 백업됩니다.

## 💡 RAG(검색) 최적화 정보
생성된 `README.md`는 다음과 같은 RAG 최적화 요소를 포함합니다.
- **Metadata Front-matter:** 문서 최상단에 카테고리, 태그, 버전 정보를 YAML 형식으로 포함하여 검색 엔진의 인덱싱 정확도 향상.
- **Contextual Summary:** YAML 설정 내용(포트, 엔드포인트, 이미지 정보 등)을 인간이 읽기 쉬운 텍스트로 변환하여 검색 시 매칭률 향상.

## ⚠️ 주의 사항
- **Windows 경로:** 경로 입력 시 역슬래시(`\`) 문제가 발생하지 않도록 반드시 `r"C:\경로"` 처럼 앞에 `r`을 붙여주세요.
- **백업 필수:** 작업 전 원본 데이터를 별도의 장소에 백업해 두는 것을 강력히 권장합니다.

---
**작성일:** 2026-04-27
**담당자:** Gemini CLI (Yoon)
