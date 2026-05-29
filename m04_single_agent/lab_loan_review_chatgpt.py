"""
m04_single_agent/lab_loan_review_chatgpt.py
──────────────────────────────────────────────────────────────────
MODULE 4 실습: 여신 서류 검토 자동화 에이전트
──────────────────────────────────────────────────────────────────
실습 목표:
  1. ReAct 패턴의 Thought-Action-Observation 루프를 구현한다.
  2. 여신심사 3종 Tool을 구현하고 에이전트와 연결한다.
  3. DTI, LTV 계산 및 규정 준수 여부를 자동 검증한다.
  4. 오류 처리, 최대 턴 제한, 디버그 메시지를 설계한다.

실행 방법:
  python m04_single_agent/lab_loan_review_chatgpt.py
──────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.openai_compat import (
    assistant_message_with_tool_calls,
    completion_text,
    create_chat_completion,
    finish_reason,
)
from common.utils import Printer, get_logger, get_openai_client, now_kst_str

logger = get_logger(__name__)
P = Printer()


# ──────────────────────────────────────────────────────────────────
# SECTION 1: 모의 서류 데이터베이스
# ──────────────────────────────────────────────────────────────────

MOCK_DOCUMENTS = {
    "LOAN-2025-001": {
        "application": {
            "applicant_name": "홍길동",
            "birth_date": "1985-03-15",
            "loan_amount": 300_000_000,
            "loan_purpose": "주택구입",
            "application_date": "2025-05-01",
        },
        "id_card": {
            "name": "홍길동",
            "birth": "1985-03-15",
            "expiry": "2030-01-14",
            "type": "주민등록증",
        },
        "income_proof": {
            "annual_income": 60_000_000,
            "employer": "(주)ABC기업",
            "position": "과장",
            "employed_years": 5,
            "doc_type": "근로소득원천징수영수증",
        },
        "property_doc": {
            "address": "서울시 강남구 역삼동 123-45",
            "area_sqm": 84.5,
            "appraisal_value": 500_000_000,
            "appraiser": "한국감정원",
            "zone": "비규제지역",
        },
    },
    "LOAN-2025-002": {
        "application": {
            "applicant_name": "김영희",
            "birth_date": "1990-07-22",
            "loan_amount": 150_000_000,
            "loan_purpose": "신용대출",
            "application_date": "2025-05-02",
        },
        "id_card": {
            "name": "김영희",
            "birth": "1990-07-22",
            "expiry": "2028-06-15",
            "type": "주민등록증",
        },
        "income_proof": {
            "annual_income": 45_000_000,
            "employer": "(주)XYZ코퍼레이션",
            "position": "대리",
            "employed_years": 2,
            "doc_type": "근로소득원천징수영수증",
        },
        "property_doc": None,
    },
    "LOAN-2025-003": {
        "application": {
            "applicant_name": "이철수",
            "birth_date": "1982-11-03",
            "loan_amount": 400_000_000,
            "loan_purpose": "주택구입",
            "application_date": "2025-05-03",
        },
        "id_card": {
            "name": "이철수",
            "birth": "1982-11-03",
            "expiry": "2029-12-31",
            "type": "주민등록증",
        },
        "income_proof": {
            "annual_income": 55_000_000,
            "employer": "(주)DEF테크",
            "position": "차장",
            "employed_years": 7,
            "doc_type": "근로소득원천징수영수증",
        },
        "property_doc": {
            "address": "서울시 송파구 잠실동 100",
            "area_sqm": 76.0,
            "appraisal_value": 600_000_000,
            "appraiser": "한국감정원",
            "zone": "규제지역",
        },
    },
}

REGULATIONS = {
    "DTI-001": {
        "name": "DTI 한도 (일반)",
        "limit": 40.0,
        "unit": "%",
        "basis": "금융감독원 주택담보대출 규제 가이드라인",
    },
    "DTI-002": {
        "name": "DTI 한도 (투기과열지구)",
        "limit": 30.0,
        "unit": "%",
        "basis": "부동산 대책 관련 금융규제",
    },
    "LTV-001": {
        "name": "LTV 한도 (비규제지역)",
        "limit": 70.0,
        "unit": "%",
        "basis": "금융감독원 주택담보대출 규제 가이드라인",
    },
    "LTV-002": {
        "name": "LTV 한도 (규제지역)",
        "limit": 50.0,
        "unit": "%",
        "basis": "부동산 규제지역 금융규제",
    },
    "LTV-003": {
        "name": "LTV 한도 (투기과열지구)",
        "limit": 40.0,
        "unit": "%",
        "basis": "투기과열지구 부동산 금융규제",
    },
    "AGE-001": {
        "name": "최소 신청 연령",
        "min_age": 19,
        "unit": "세",
        "basis": "민법 제4조 (성년)",
    },
    "EMPL-001": {
        "name": "최소 재직기간 (신용대출)",
        "min_years": 1,
        "unit": "년",
        "basis": "내부 여신심사기준 제3조",
    },
}

FIELD_ALIASES = {
    "application": {
        "requested_loan_amount": "loan_amount",
        "requested_amount": "loan_amount",
        "amount": "loan_amount",
        "purpose": "loan_purpose",
        "submitted_at": "application_date",
        "submitted_date": "application_date",
    },
    "id_card": {
        "birth_date": "birth",
        "date_of_birth": "birth",
        "expiration_date": "expiry",
        "expiry_date": "expiry",
        "document_type": "type",
        "id_type": "type",
    },
    "income_proof": {
        "income_type": "doc_type",
        "employment_years": "employed_years",
        "tenure_years": "employed_years",
        "company_name": "employer",
        "issued_by": "employer",
    },
    "property_doc": {
        "property_value": "appraisal_value",
        "property_address": "address",
        "property_area": "area_sqm",
        "appraised_value": "appraisal_value",
        "valuation_amount": "appraisal_value",
    },
}


# ──────────────────────────────────────────────────────────────────
# SECTION 2: 여신심사 Tool 3종 구현
# ──────────────────────────────────────────────────────────────────

def extract_document_info(case_id: str, doc_type: str, fields: list[str]) -> dict:
    """대출 신청 서류에서 필요한 정보를 추출한다."""
    case = MOCK_DOCUMENTS.get(case_id)
    if not case:
        return {"error": f"신청 건을 찾을 수 없습니다: {case_id}"}

    doc = case.get(doc_type)
    if doc is None:
        return {
            "status": "해당_서류_없음",
            "doc_type": doc_type,
            "message": f"{doc_type} 서류가 제출되지 않았습니다",
        }

    result = {}
    missing = []
    aliases = FIELD_ALIASES.get(doc_type, {})
    for field in fields:
        source_field = aliases.get(field, field)
        value = doc.get(source_field)
        if value is None:
            missing.append(field)
            result[field] = "정보 없음 — 추가 서류 요청 필요"
        else:
            result[field] = value

    if missing:
        result["⚠_누락_필드"] = missing

    return result


def check_regulation_compliance(rule_code: str, check_value: float) -> dict:
    """특정 규정 코드에 대한 준수 여부를 확인한다."""
    rule = REGULATIONS.get(rule_code)
    if not rule:
        return {
            "error": f"규정 코드를 찾을 수 없습니다: {rule_code}",
            "available_codes": list(REGULATIONS.keys()),
        }

    result = {
        "rule_code": rule_code,
        "rule_name": rule["name"],
        "basis": rule["basis"],
        "check_value": f"{check_value}{rule['unit']}",
    }

    if "limit" in rule:
        compliant = check_value <= rule["limit"]
        margin = rule["limit"] - check_value
        result.update(
            {
                "compliant": compliant,
                "limit": f"{rule['limit']}{rule['unit']}",
                "margin": f"{margin:+.1f}{rule['unit']}",
                "판정": "기준 통과" if compliant else "기준 초과 (대출 불가)",
            }
        )
    elif "min_age" in rule:
        compliant = check_value >= rule["min_age"]
        result.update(
            {
                "compliant": compliant,
                "min_value": f"{rule['min_age']}{rule['unit']}",
                "판정": "기준 통과" if compliant else "기준 미달 (대출 불가)",
            }
        )
    elif "min_years" in rule:
        compliant = check_value >= rule["min_years"]
        result.update(
            {
                "compliant": compliant,
                "min_value": f"{rule['min_years']}{rule['unit']}",
                "판정": "기준 통과" if compliant else "기준 미달 (한도 감액 가능)",
            }
        )

    return result


# 지역(zone) -> (LTV 규정 코드, 한도%)
LTV_ZONE_RULES = {
    "비규제지역": ("LTV-001", 70.0),
    "규제지역": ("LTV-002", 50.0),
    "투기과열지구": ("LTV-003", 40.0),
}


def calculate_dti_ltv(
    annual_income: float,
    existing_monthly_debt: float,
    loan_amount: float,
    property_value: float = 0,
    annual_rate: float = 4.5,
    loan_months: int = 240,
    zone: str = "비규제지역",
) -> dict:
    """DTI와 LTV를 계산한다. zone에 따라 LTV 기준을 다르게 적용한다."""
    if annual_income <= 0:
        return {"error": "연소득은 0보다 커야 합니다"}

    r = annual_rate / 100 / 12
    if r == 0:
        new_monthly = loan_amount / loan_months
    else:
        new_monthly = loan_amount * r * (1 + r) ** loan_months / ((1 + r) ** loan_months - 1)

    annual_total_debt = (existing_monthly_debt + new_monthly) * 12
    dti = annual_total_debt / annual_income * 100

    result = {
        "신규_월_상환액": f"{round(new_monthly):,}원",
        "기존_월_부채": f"{round(existing_monthly_debt):,}원",
        "연간_총_부채상환": f"{round(annual_total_debt):,}원",
        "DTI": f"{round(dti, 2)}%",
        "DTI_기준": "40% 이하",
        "DTI_통과": dti <= 40.0,
    }

    if property_value > 0:
        ltv = loan_amount / property_value * 100
        ltv_code, ltv_limit = LTV_ZONE_RULES.get(zone, LTV_ZONE_RULES["비규제지역"])
        result.update(
            {
                "LTV": f"{round(ltv, 2)}%",
                "LTV_지역": zone,
                "LTV_규정코드": ltv_code,
                "LTV_기준": f"{ltv_limit}% 이하 ({zone})",
                "LTV_통과": ltv <= ltv_limit,
                "담보_평가액": f"{round(property_value):,}원",
            }
        )
    else:
        result["LTV"] = "해당 없음 (담보 없음)"

    result["종합_통과"] = result["DTI_통과"] and result.get("LTV_통과", True)
    return result


# ──────────────────────────────────────────────────────────────────
# SECTION 3: Tool 스키마 정의
# ──────────────────────────────────────────────────────────────────

REVIEW_TOOLS = [
    {
        "name": "extract_document_info",
        "description": "대출 신청 서류(신청서, 신분증, 소득증빙, 담보서류)에서 특정 정보를 추출합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "description": "신청 건 ID. 예: LOAN-2025-001"},
                "doc_type": {
                    "type": "string",
                    "enum": ["application", "id_card", "income_proof", "property_doc"],
                    "description": "서류 유형",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "추출할 필드 목록. 예: ['applicant_name', 'loan_amount']",
                },
            },
            "required": ["case_id", "doc_type", "fields"],
        },
    },
    {
        "name": "check_regulation_compliance",
        "description": "계산된 수치가 금융 규정 한도를 준수하는지 확인합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_code": {
                    "type": "string",
                    "description": "규정 코드. 예: DTI-001, LTV-001, AGE-001, EMPL-001",
                },
                "check_value": {
                    "type": "number",
                    "description": "확인할 수치. 예: DTI 45.2%면 45.2",
                },
            },
            "required": ["rule_code", "check_value"],
        },
    },
    {
        "name": "calculate_dti_ltv",
        "description": "DTI와 LTV를 계산하고 규정 통과 여부를 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "annual_income": {"type": "number", "description": "연소득(원)"},
                "existing_monthly_debt": {"type": "number", "description": "기존 월 부채 상환액(원)"},
                "loan_amount": {"type": "number", "description": "신청 대출금액(원)"},
                "property_value": {"type": "number", "description": "담보물 감정가(원). 신용대출이면 0"},
                "annual_rate": {"type": "number", "description": "연이율(%). 기본값 4.5"},
                "loan_months": {"type": "integer", "description": "대출 기간(개월). 기본값 240"},
                "zone": {
                    "type": "string",
                    "enum": ["비규제지역", "규제지역", "투기과열지구"],
                    "description": "담보물 소재 지역 규제 구분. property_doc의 zone 값을 그대로 전달. LTV 기준이 달라짐(비규제 70%/규제 50%/투기과열 40%)",
                },
            },
            "required": ["annual_income", "existing_monthly_debt", "loan_amount"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────
# SECTION 4: 에이전트 시스템 프롬프트
# ──────────────────────────────────────────────────────────────────

REVIEW_SYSTEM_PROMPT = """당신은 금융회사 여신심사 보조 AI입니다.

[심사 순서]
반드시 아래 순서로 도구를 사용하세요.
1. extract_document_info(doc_type="application")로 신청서 기본 정보 추출
2. extract_document_info(doc_type="id_card")로 신분증 정보 추출
3. extract_document_info(doc_type="income_proof")로 소득 증빙 확인
4. extract_document_info(doc_type="property_doc")로 담보물 정보 확인 (zone 필드 포함)
5. calculate_dti_ltv로 DTI, LTV 계산 — property_doc의 zone 값을 zone 인자로 반드시 전달
6. check_regulation_compliance로 각 규정 준수 여부 확인

[지역(zone)별 LTV 규정 선택]
담보물 property_doc의 zone 값에 따라 LTV 규정 코드를 다르게 적용하세요.
- zone == "비규제지역": LTV-001 (70% 이하)
- zone == "규제지역": LTV-002 (50% 이하)
- zone == "투기과열지구": LTV-003 (40% 이하)
calculate_dti_ltv 호출 시 zone을 전달하고, check_regulation_compliance에는 위 표의 해당 LTV 코드를 사용하세요.

[출력 형식]
최종 답변은 아래 형식으로 작성하세요.
## 여신심사 체크리스트
**신청 건 ID:** [case_id]  **심사 일시:** [현재 날짜]

| 항목 | 확인 내용 | 판정 |
|------|-----------|------|
| 1. 신청인 기본정보 | ... | ✓/✗ |
| 2. 신분증 확인 | ... | ✓/✗ |
| 3. 소득 증빙 | ... | ✓/✗ |
| 4. 담보물 현황 | ... | ✓/✗ |
| 5. DTI | ...% (기준 40%) | ✓/✗ |
| 6. LTV | ...% (기준 70%) | ✓/✗ |

**종합 심사 의견:** [승인 권고 / 조건부 승인 권고 / 반려 권고]
**주요 이슈:** [있으면 기재]
**담당자 검토 필요 사항:** [목록]

주의:
- 규정에 없는 내용은 추정하지 마세요.
- 정보 부족 시 "추가 서류 요청 필요"라고 명시하세요.
- 이 보고서는 참고용이며 최종 결정은 심사역이 수행한다고 안내하세요.
"""


# ──────────────────────────────────────────────────────────────────
# SECTION 5: Tool 디스패치 및 에이전트 실행
# ──────────────────────────────────────────────────────────────────

# Tool 호출 감사 로그 (금융권 실무: 호출 이력 추적)
audit_log: list[dict] = []


def _record_audit(started_at: str, name: str, inputs: dict, result: dict) -> None:
    """Tool 호출 1건을 감사 로그에 기록한다."""
    audit_log.append(
        {
            "time": started_at,
            "tool": name,
            "input_keys": list(inputs.keys()),
            "case_id": inputs.get("case_id"),
            "success": isinstance(result, dict) and "error" not in result,
        }
    )


def execute_review_tool(name: str, inputs: dict) -> dict:
    """여신심사 Tool 디스패치"""
    started_at = now_kst_str()
    logger.info(f"Tool: {name} | 입력: {list(inputs.keys())}")
    try:
        if name == "extract_document_info":
            result = extract_document_info(**inputs)
        elif name == "check_regulation_compliance":
            result = check_regulation_compliance(**inputs)
        elif name == "calculate_dti_ltv":
            result = calculate_dti_ltv(**inputs)
        else:
            result = {"error": f"알 수 없는 Tool: {name}"}
    except TypeError as exc:
        result = {"error": f"입력 형식 오류: {exc}"}
    except Exception as exc:
        logger.error(f"Tool 오류 [{name}]: {exc}")
        result = {"error": f"{type(exc).__name__}: {exc}"}

    _record_audit(started_at, name, inputs, result)
    return result


def run_loan_review_agent(case_id: str, max_turns: int = 12) -> str:
    """여신심사 에이전트를 ReAct 루프로 실행한다."""
    client = get_openai_client()
    messages = [
        {
            "role": "user",
            "content": f"대출 신청 건 {case_id}를 여신심사 관점에서 안전하게 검토해 주세요.",
        }
    ]

    P.header(f"여신심사 에이전트 시작: {case_id}", "yellow")
    start_time = time.time()

    for turn in range(1, max_turns + 1):
        P.section(f"턴 {turn}/{max_turns}")
        response = create_chat_completion(
            client=client,
            model="gpt-4o-mini",
            max_tokens=3000,
            system=REVIEW_SYSTEM_PROMPT,
            tools=REVIEW_TOOLS,
            messages=messages,
            temperature=0.1,
        )

        reason = finish_reason(response)
        P.kv("finish_reason", reason)

        if reason != "tool_calls":
            final = completion_text(response)
            elapsed = int((time.time() - start_time) * 1000)
            P.success(f"최종 답변 완료! ({elapsed}ms, {turn}턴)")
            return final

        message = response.choices[0].message
        messages.append(assistant_message_with_tool_calls(message))
        for tool_call in message.tool_calls or []:
            inputs = json.loads(tool_call.function.arguments or "{}")
            P.kv(f"  호출 {tool_call.function.name}", str(inputs)[:80])
            result = execute_review_tool(tool_call.function.name, inputs)
            P.kv("  반환 결과", str(result)[:80])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    logger.error(f"최대 턴 초과: {case_id}")
    return (
        f"처리 오류: 심사 프로세스가 최대 반복({max_turns}턴)을 초과했습니다.\n"
        f"건 ID: {case_id}\n"
        "조치: 수동 심사 필요"
    )


# ──────────────────────────────────────────────────────────────────
# SECTION 6: 메인 실행
# ──────────────────────────────────────────────────────────────────

def main():
    P.header("MODULE 4 실습 — 여신심사 에이전트", "blue")

    test_cases = [
        ("LOAN-2025-001", "표준 주택담보대출 — 정상 케이스"),
        ("LOAN-2025-002", "신용대출 — 검토 필요 케이스"),
    ]

    for case_id, description in test_cases:
        print(f"\n{'=' * 60}")
        print(f"케이스: {description}")
        print(f"{'=' * 60}")

        result = run_loan_review_agent(case_id)
        print("\n[심사 체크리스트]\n")
        print(result)
        print()

    P.header("MODULE 4 실습 완료", "green")


if __name__ == "__main__":
    main()