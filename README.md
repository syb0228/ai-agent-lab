# AI Agent Lab

금융 도메인을 소재로 **AI 에이전트(Agent)** 개념을 단계별로 실습하는 학습용 저장소입니다.
LLM의 **Tool Use(Function Calling)**, **ReAct 루프**, 그리고 실무에 필요한 **보안 처리(PII 마스킹·프롬프트 인젝션 탐지)** 와 **안정화(재시도)** 패턴을 다룹니다.

> ⚠️ 모든 외부 데이터(뉴스·환율·신용점수 등)는 **교육용 모의(Mock) 데이터**입니다.

---

## 📁 프로젝트 구조

```
ai-agent-lab/
├── common/
│   ├── _init_.py
│   └── utils.py              # 공통 유틸 (클라이언트, PII 마스킹, 인젝션 탐지, 재시도, 모델 라우팅)
├── m02_tool_use/
│   └── lab_tools_chatgpt.py  # MODULE 2: ChatGPT Function Calling(Tool Use) 실습
├── .env.example             # 환경변수 양식
├── .gitignore
└── requirements.txt
```

---

## 🚀 시작하기

### 1. 가상환경 생성 및 활성화
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
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
```
> 🔒 `.env`는 `.gitignore`에 등록되어 커밋되지 않습니다. API 키를 절대 저장소에 올리지 마세요.

---

## 🧪 MODULE 2 — Tool Use 실습

금융 상담 에이전트가 질문에 따라 4종 도구를 선택·실행하고 결과를 종합해 답변합니다.

| 도구 | 설명 |
|------|------|
| `calculate_loan_payment` | 원리금균등상환 대출 월 상환액 계산 |
| `search_financial_news` | 금융/경제 뉴스 검색 (Mock) |
| `get_exchange_rate` | 환율 조회 (Mock) |
| `check_credit_score` | 고객 신용점수 조회 (Mock) |

### 실행
```powershell
# 전체 테스트 케이스
python m02_tool_use/lab_tools_chatgpt.py

# 대화형 모드
python m02_tool_use/lab_tools_chatgpt.py --interactive
```

### 함수 직접 호출 예시
```python
from m02_tool_use.lab_tools_chatgpt import run_financial_agent

answer = run_financial_agent(
    "3억원을 연 4.5%로 20년 대출받으면 월 상환액이 얼마인가요?",
    verbose=True,
)
print(answer)
```

---

## 🔧 공통 유틸리티 (`common/utils.py`)

- **클라이언트 팩토리**: `get_openai_client()`, `get_anthropic_client()`
- **PII 마스킹**: 주민번호·전화번호·계좌·카드·이메일 자동 마스킹 (`mask_pii`)
- **프롬프트 인젝션 탐지**: 위험 패턴 사전 차단 (`detect_prompt_injection`)
- **재시도 데코레이터**: 지수 백오프 API 재시도 (`@retry_on_api_error`)
- **모델 라우팅**: 작업 유형별 최적 모델 선택 (`get_model`)

---

## 🛠️ 기술 스택
- Python 3.11+
- OpenAI Python SDK (`gpt-5` 계열)
- python-dotenv

---

## 📌 참고
학습/교육용 저장소이며, 계산 결과나 모의 데이터는 실제 금융 상담·투자 판단의 근거로 사용할 수 없습니다.
