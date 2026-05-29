# AI Agent Lab

금융 도메인을 소재로 **AI 에이전트(Agent)** 개념을 단계별로 실습하는 학습용 저장소입니다.
LLM의 **Tool Use(Function Calling)**, **RAG**, **단일 에이전트(ReAct)**, **보안·모니터링**, **멀티에이전트 협업 패턴**까지 모듈별로 다룹니다.

> ⚠️ 모든 외부 데이터(뉴스·환율·신용점수·시장 데이터 등)는 **교육용 모의(Mock) 데이터**입니다.
> 계산 결과나 모의 데이터는 실제 금융 상담·투자·여신 판단의 근거로 사용할 수 없습니다.

---

## 📁 프로젝트 구조

```
ai-agent-lab/
├── common/
│   ├── utils.py                       # 공통 유틸 (클라이언트, PII 마스킹, 인젝션 탐지, 재시도, 모델 상수)
│   └── openai_compat.py               # OpenAI 호출 호환 셔임 (system 병합, tools 변환, 토큰 추출)
├── m02_tool_use/
│   └── lab_tools_chatgpt.py           # MODULE 2: Tool Use (Function Calling)
├── m03_memory_rag/
│   ├── lab_rag_pipeline_chatgpt.py    # MODULE 3: 금융 규정 RAG Q&A
│   └── make_sample_pdf.py             # 샘플 규정 PDF 생성 스크립트
├── m04_single_agent/
│   └── lab_loan_review_chatgpt.py     # MODULE 4: 단일 에이전트 (여신심사)
├── m05_monitoring/
│   └── lab_security_monitoring_chatgpt.py  # MODULE 5: 보안 & 모니터링
├── m06_multiagent/
│   └── lab_patterns_chatgpt.py        # MODULE 6: 멀티에이전트 협업 패턴 4종
├── data/                              # 규정 PDF 등 입력 문서
├── .env.example                       # 환경변수 양식
├── .gitignore
└── requirements.txt
```

> 벡터 DB(`finance_kb*/`), `.env`, 로그, `__pycache__`는 `.gitignore`로 제외됩니다(코드로 재생성 가능).

---

## 🚀 시작하기

### 1. 가상환경 생성 및 활성화
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # macOS / Linux
```

### 2. 의존성 설치
```powershell
python -m pip install -r requirements.txt
```

### 3. 환경변수 설정
`.env.example`을 복사해 `.env`를 만들고 실제 키를 채웁니다.
```powershell
Copy-Item .env.example .env
```
```dotenv
OPENAI_API_KEY=your-openai-api-key-here
# (선택) LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL  ← m05 모니터링용
```
> 🔒 `.env`는 `.gitignore`에 등록되어 커밋되지 않습니다. API 키를 절대 저장소에 올리지 마세요.
> 💡 한글 출력이 깨지면 실행 전에 `\$env:PYTHONIOENCODING='utf-8'`(PowerShell)을 한 번 설정하세요.

---

## 📚 모듈별 실습

### MODULE 2 — Tool Use (Function Calling)
금융 상담 에이전트가 질문에 따라 4종 도구를 선택·실행하고 결과를 종합해 답변합니다 (ReAct 루프).

| 도구 | 설명 |
|------|------|
| `calculate_loan_payment` | 원리금균등상환 대출 월 상환액 계산 |
| `calculate_dti` | DTI(총부채상환비율) 계산 및 통과 판정 |
| `search_financial_news` | 금융/경제 뉴스 검색 (Mock) |
| `get_exchange_rate` | 환율 조회 (Mock) |
| `check_credit_score` | 고객 신용점수 조회 (Mock) |

```powershell
python m02_tool_use/lab_tools_chatgpt.py               # 전체 테스트 케이스
python m02_tool_use/lab_tools_chatgpt.py --interactive  # 대화형 모드
```
```python
from m02_tool_use.lab_tools_chatgpt import run_financial_agent
print(run_financial_agent("3억원을 연 4.5%로 20년 대출받으면 월 상환액이 얼마인가요?", verbose=True))
```

### MODULE 3 — RAG Q&A (금융 규정)
규정 문서를 청크로 분할 → OpenAI 임베딩 → FAISS 벡터 DB → MMR 검색 → 근거 조항 기반 답변.

```powershell
python m03_memory_rag/lab_rag_pipeline_chatgpt.py                # 데모 Q&A
python m03_memory_rag/lab_rag_pipeline_chatgpt.py --interactive  # 대화형
python m03_memory_rag/make_sample_pdf.py                         # 샘플 규정 PDF 생성(data/)
```
- 근거 조항(예: 제4조) 인용, 규정에 없는 질문은 **"해당 규정에서 확인되지 않습니다"**로 환각 방지
- 텍스트/PDF 양쪽에서 벡터 DB 구축 지원 (`build_knowledge_base_from_text` / `build_knowledge_base_from_pdf`)

### MODULE 4 — 단일 에이전트 (여신심사)
ReAct 루프로 서류 추출 → DTI/LTV 계산 → 규정 준수 확인 → 심사 체크리스트를 생성합니다.

| Tool | 설명 |
|------|------|
| `extract_document_info` | 신청서·신분증·소득·담보 서류에서 필드 추출 (누락 필드 표시) |
| `calculate_dti_ltv` | DTI/LTV 계산, 지역(zone)별 LTV 기준 적용 |
| `check_regulation_compliance` | 규정 코드별 준수 여부 판정 |

```python
from m04_single_agent.lab_loan_review_chatgpt import run_loan_review_agent
print(run_loan_review_agent("LOAN-2025-001"))
```
- 지역별 LTV: 비규제 70% / 규제지역 50% / 투기과열지구 40%
- Tool 호출 **감사 로그**(`audit_log`)로 호출 시각·도구·입력 키·신청건 ID·성공 여부 기록

### MODULE 5 — 보안 & 모니터링
보안 가드레일 + 관측성(LangFuse 연동 옵션)을 갖춘 에이전트 래퍼.

```powershell
python m05_monitoring/lab_security_monitoring_chatgpt.py
```
- **PII 마스킹**: 주민·전화·카드·이메일·사업자등록번호 등 자동 마스킹
- **프롬프트 인젝션 탐지**: 라벨 기반 패턴(예: `KR_IGNORE_PREVIOUS`, `KR_SECURITY_BYPASS`) 차단 + 구조화 JSON 감사 로그
- **성능 벤치마크**: 지연/토큰/비용 측정 + 업무별 **SLA(P95)** 판정 (FAQ 10s / 여신심사 120s / FDS 3s)

### MODULE 6 — 멀티에이전트 협업 패턴 4종
```powershell
python m06_multiagent/lab_patterns_chatgpt.py
```
| 패턴 | 내용 |
|------|------|
| **Sequential** | 여신심사 파이프라인 (단계별 출력이 다음 단계 입력) |
| **Parallel** | 멀티마켓 데이터 동시 수집 (NYSE/NASDAQ/LSE/KRX/HKEX/NEWS) |
| **Hierarchical** | 투자 리서치 오케스트레이터 (뉴스·재무·리스크·밸류에이션 서브에이전트 통합) |
| **Debate** | 대형 여신 토론 심사 (보수·균형·고객중심·규정준수 4개 심사역 + 중재자) |

`recommend_pattern()`은 업무 특성(의존성/병렬화/복잡도/정확도/실시간/고위험/사람검토)에 따라
`Sequential` / `Parallel` / `Hierarchical` / `Debate` / `Debate + HITL` 중 최적 패턴을 추천합니다.

---

## 🔧 공통 모듈

### `common/utils.py`
- **클라이언트 팩토리**: `get_openai_client()`, `get_anthropic_client()`
- **PII 마스킹**: `mask_pii()` — 더 구체적인 패턴을 먼저 적용해 오탐 방지
- **프롬프트 인젝션 탐지**: `detect_prompt_injection()` — `(탐지여부, 라벨코드)` 반환
- **재시도 데코레이터**: `@retry_on_api_error` — 지수 백오프
- **모델 상수/라우팅**: `MODELS`, `get_model()`

### `common/openai_compat.py`
- `create_chat_completion()` — system 프롬프트 병합, **Anthropic 형식 tools(input_schema) → OpenAI 형식(parameters) 자동 변환**
- `finish_reason()`, `completion_text()`, `assistant_message_with_tool_calls()`
- `prompt_tokens()`, `completion_tokens()`

---

## 🛠️ 기술 스택
- Python 3.11+
- OpenAI Python SDK — 채팅 모델 `gpt-4o` / `gpt-4o-mini`, 임베딩 `text-embedding-3-small`
- LangChain · FAISS · pypdf (m03 RAG)
- LangFuse (m05 모니터링, 선택)
- python-dotenv

---

## 📌 참고
- 학습/교육용 저장소입니다. 모의 데이터·계산 결과는 실제 금융 의사결정의 근거가 될 수 없습니다.
- 각 모듈은 독립 실행 가능하며, 프로젝트 루트에서 실행해야 import가 정상 동작합니다.
