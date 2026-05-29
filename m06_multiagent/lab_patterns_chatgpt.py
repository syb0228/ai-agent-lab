"""
m06_multiagent/lab_patterns_chatgpt.py
──────────────────────────────────────────────────────────────────
MODULE 6 실습: 멀티에이전트 협업 패턴 4종
──────────────────────────────────────────────────────────────────
학습 목표:
  1. Sequential 패턴 — 여신심사 파이프라인
  2. Parallel  패턴 — 멀티마켓 데이터 동시 수집
  3. Hierarchical 패턴 — 투자 리서치 오케스트레이터
  4. Debate 패턴 — 대형 여신 최종 심사
──────────────────────────────────────────────────────────────────
"""

import os, sys, json, asyncio, time
from typing import Any
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.utils import get_openai_client, get_logger, MODELS, Printer
from common.openai_compat import completion_text, create_chat_completion

logger = get_logger(__name__)
P = Printer()

client = None


def llm_call(prompt: str, system: str = "", model: str = MODELS["haiku"],
             max_tokens: int = 500) -> str:
    """단순 LLM 호출 유틸리티"""
    global client
    if client is None:
        client = get_openai_client()
    msgs = [{"role": "user", "content": prompt}]
    kwargs = dict(model=model, max_tokens=max_tokens, messages=msgs)
    if system:
        kwargs["system"] = system
    response = create_chat_completion(client=client, **kwargs)
    return completion_text(response)


# ══════════════════════════════════════════════════════════════════
# PATTERN 1: Sequential (순차) 패턴
# ══════════════════════════════════════════════════════════════════

class SequentialLoanReview:
    """
    여신심사 순차 파이프라인
    각 에이전트의 출력이 다음 에이전트의 입력이 됩니다.
    """

    def document_collector(self, state: dict) -> dict:
        """1단계: 서류 수집 에이전트"""
        # 실제: PDF 파싱, OCR 처리
        state["documents"] = {
            "신청인명":   "홍길동",
            "대출금액":   500_000_000,
            "연소득":     50_000_000,
            "담보평가액": 600_000_000,
            "재직기간":   5,
        }
        state["stage"] = "서류수집 완료"
        P.success(f"  1단계 완료: {list(state['documents'].keys())}")
        return state

    def data_validator(self, state: dict) -> dict:
        """2단계: 데이터 검증 에이전트"""
        docs = state["documents"]
        checks = {
            "신분 확인":    True,
            "소득 증빙":    docs["연소득"] > 0,
            "서류 완비":    all(v for v in docs.values()),
        }
        state["validation"] = checks
        state["stage"] = "데이터검증 완료"
        P.success(f"  2단계 완료: {sum(checks.values())}/{len(checks)} 항목 통과")
        return state

    def dti_ltv_calculator(self, state: dict) -> dict:
        """3단계: DTI/LTV 계산 에이전트"""
        docs = state["documents"]
        loan_amt    = docs["대출금액"]
        income      = docs["연소득"]
        prop_value  = docs["담보평가액"]
        annual_rate = 0.045
        months      = 240
        r   = annual_rate / 12
        pmt = loan_amt * r * (1+r)**months / ((1+r)**months - 1)
        dti = pmt * 12 / income * 100
        ltv = loan_amt / prop_value * 100

        state["ratios"] = {
            "월상환액": round(pmt),
            "DTI":      round(dti, 2),
            "DTI통과":  dti <= 40,
            "LTV":      round(ltv, 2),
            "LTV통과":  ltv <= 70,
        }
        state["stage"] = "DTI/LTV계산 완료"
        P.success(f"  3단계 완료: DTI={dti:.1f}% ({'✓' if dti<=40 else '✗'}), LTV={ltv:.1f}% ({'✓' if ltv<=70 else '✗'})")
        return state

    def report_writer(self, state: dict) -> dict:
        """4단계: 보고서 작성 에이전트"""
        ratios = state["ratios"]
        docs   = state["documents"]
        overall = ratios["DTI통과"] and ratios["LTV통과"]

        report = f"""
## 여신심사 자동화 체크리스트

| 항목       | 결과                          | 판정 |
|------------|-------------------------------|------|
| 신청인     | {docs['신청인명']}             | ✓    |
| 대출금액   | {docs['대출금액']:,}원         | ✓    |
| DTI        | {ratios['DTI']}% (기준 40%)   | {'✓' if ratios['DTI통과'] else '✗'} |
| LTV        | {ratios['LTV']}% (기준 70%)   | {'✓' if ratios['LTV통과'] else '✗'} |
| 월 상환액  | {ratios['월상환액']:,}원       | ✓    |

**종합 의견:** {'승인 권고' if overall else '반려 권고 (규정 초과)'}
⚠️ 최종 결정은 심사역이 수행합니다.
"""
        state["report"] = report.strip()
        state["stage"]  = "완료"
        P.success("  4단계 완료: 보고서 생성")
        return state

    def run(self, case_id: str) -> str:
        P.header(f"Sequential 패턴: 여신심사 파이프라인 [{case_id}]", "cyan")
        state = {"case_id": case_id}
        pipeline = [
            self.document_collector,
            self.data_validator,
            self.dti_ltv_calculator,
            self.report_writer,
        ]
        for step in pipeline:
            state = step(state)
        return state["report"]


# ══════════════════════════════════════════════════════════════════
# PATTERN 2: Parallel (병렬) 패턴
# ══════════════════════════════════════════════════════════════════

async def collect_market_data_async(market: str, ticker: str) -> dict:
    """개별 시장 데이터 수집 (비동기)"""
    await asyncio.sleep(0.1)  # 실제: 외부 API 호출 시뮬레이션
    MOCK_DATA = {
        "NYSE":   {"price_usd": 168.50, "volume": 52_300_000, "pe": 28.5},
        "NASDAQ": {"price_usd": 172.30, "volume": 40_100_000, "pe": 29.1},
        "LSE":   {"price_gbp": 132.20, "volume": 8_100_000,  "pe": 27.8},
        "KRX":   {"price_krw": 225_000, "volume": 12_400_000, "pe": 25.2},
        "HKEX":  {"price_hkd": 1_312.0, "volume": 5_600_000, "pe": 26.1},
        "NEWS":  {"headlines": [f"{ticker} Q1 실적 발표", f"{ticker} 신제품 출시 예정"], "sentiment": "긍정"},
    }
    return {"market": market, "ticker": ticker, "data": MOCK_DATA.get(market, {})}


async def run_parallel_data_collection(ticker: str) -> dict:
    """
    멀티마켓 데이터 병렬 수집
    5개 에이전트가 동시에 실행됩니다.
    """
    P.header(f"Parallel 패턴: 멀티마켓 데이터 수집 [{ticker}]", "cyan")
    start = time.time()

    tasks = [
        collect_market_data_async("NYSE",   ticker),
        collect_market_data_async("NASDAQ", ticker),
        collect_market_data_async("LSE",    ticker),
        collect_market_data_async("KRX",    ticker),
        collect_market_data_async("HKEX",   ticker),
        collect_market_data_async("NEWS",   ticker),
    ]

    # 모든 에이전트를 병렬로 실행
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = int((time.time() - start) * 1000)

    collected = {}
    for result in results:
        if isinstance(result, Exception):
            P.warn(f"  수집 오류: {result}")
        else:
            collected[result["market"]] = result["data"]
            P.success(f"  {result['market']} 수집 완료")

    P.kv("병렬 처리 시간", f"{elapsed}ms (순차 처리 예상: {elapsed * len(tasks)}ms, {len(tasks)}개 시장)")
    return collected


# ══════════════════════════════════════════════════════════════════
# PATTERN 3: Hierarchical (계층적) 패턴
# ══════════════════════════════════════════════════════════════════

class HierarchicalResearchSystem:
    """
    투자 리서치 리포트 생성 계층적 에이전트
    오케스트레이터가 서브에이전트들을 지휘합니다.
    """

    def news_subagent(self, topic: str) -> str:
        """서브에이전트 1: 뉴스 수집"""
        return llm_call(
            f"{topic}에 관한 최신 뉴스 3가지를 한 줄씩 요약하세요.",
            system="금융 뉴스 수집 전문가. 간결하게 요약합니다.",
            model=MODELS["haiku"]
        )

    def financial_data_subagent(self, topic: str) -> str:
        """서브에이전트 2: 재무 데이터 분석"""
        return llm_call(
            f"{topic}의 주요 재무 지표(PER, ROE, 영업이익률)와 현황을 분석하세요.",
            system="재무 분석 전문가. 수치 중심으로 분석합니다.",
            model=MODELS["haiku"]
        )

    def risk_subagent(self, topic: str) -> str:
        """서브에이전트 3: 리스크 분석"""
        return llm_call(
            f"{topic}과 관련된 투자 리스크 요인 3가지를 제시하세요.",
            system="리스크 관리 전문가. 보수적 관점에서 분석합니다.",
            model=MODELS["haiku"]
        )

    def valuation_subagent(self, topic: str) -> str:
        """서브에이전트 4: 밸류에이션 분석"""
        return llm_call(
            f"{topic}의 목표주가, 상승여력(%), 밸류에이션 부담(고평가/적정/저평가)을 분석하세요.",
            system="밸류에이션 분석 전문가. 목표주가와 상승여력, 밸류에이션 부담을 중심으로 평가합니다.",
            model=MODELS["haiku"]
        )

    def orchestrator(self, topic: str) -> str:
        """
        오케스트레이터: 전체 작업 계획 및 서브에이전트 결과 통합
        """
        P.header(f"Hierarchical 패턴: 투자 리서치 [{topic}]", "cyan")

        P.section("서브에이전트 실행")
        P.kv("  1", "뉴스 수집 에이전트 실행 중...")
        news = self.news_subagent(topic)
        P.success("  뉴스 수집 완료")

        P.kv("  2", "재무 데이터 에이전트 실행 중...")
        financial = self.financial_data_subagent(topic)
        P.success("  재무 분석 완료")

        P.kv("  3", "리스크 에이전트 실행 중...")
        risks = self.risk_subagent(topic)
        P.success("  리스크 분석 완료")

        P.kv("  4", "밸류에이션 에이전트 실행 중...")
        valuation = self.valuation_subagent(topic)
        P.success("  밸류에이션 분석 완료")

        # 오케스트레이터가 최종 보고서 통합
        P.section("오케스트레이터 통합")
        final_report = llm_call(
            f"""다음 자료를 바탕으로 투자 리서치 리포트 초안을 작성하세요.

[뉴스 요약]
{news}

[재무 분석]
{financial}

[리스크 요인]
{risks}

[밸류에이션 분석]
{valuation}

주제: {topic}
구조: 요약 → 핵심 이슈 → 투자의견 → 밸류에이션 → 리스크 → 결론""",
            system="투자 리서치 리포트 작성 전문가.",
            model=MODELS["sonnet"],
            max_tokens=800,
        )
        P.success("오케스트레이터 통합 완료")
        return final_report

    def run(self, topic: str) -> str:
        return self.orchestrator(topic)


# ══════════════════════════════════════════════════════════════════
# PATTERN 4: Debate (토론) 패턴
# ══════════════════════════════════════════════════════════════════

class DebateLoanReview:
    """
    대형 여신 토론 심사
    3개의 에이전트가 서로 다른 관점에서 의견을 제시하고
    중재자가 최종 판단을 내립니다.
    """

    def conservative_reviewer(self, case: dict) -> str:
        """보수적 심사역 에이전트"""
        return llm_call(
            f"다음 대출 신청에 대해 엄격하고 보수적인 관점에서 심사 의견을 제시하세요:\n{json.dumps(case, ensure_ascii=False)}",
            system="당신은 리스크에 매우 민감한 보수적 심사역입니다. 규정을 엄격히 적용합니다.",
            model=MODELS["haiku"]
        )

    def balanced_reviewer(self, case: dict) -> str:
        """균형적 심사역 에이전트"""
        return llm_call(
            f"다음 대출 신청에 대해 리스크와 수익성을 균형있게 고려하여 심사 의견을 제시하세요:\n{json.dumps(case, ensure_ascii=False)}",
            system="당신은 균형잡힌 심사역입니다. 리스크와 은행 수익성을 종합 고려합니다.",
            model=MODELS["haiku"]
        )

    def progressive_reviewer(self, case: dict) -> str:
        """고객 중심 심사역 에이전트"""
        return llm_call(
            f"다음 대출 신청에 대해 고객의 상환 가능성을 중심으로 심사 의견을 제시하세요:\n{json.dumps(case, ensure_ascii=False)}",
            system="당신은 고객 중심적 심사역입니다. 상환 능력과 고객 관계를 중시합니다.",
            model=MODELS["haiku"]
        )

    def compliance_reviewer(self, case: dict) -> str:
        """규정 준수 심사역 에이전트"""
        return llm_call(
            f"다음 대출 신청에 대해 금융 규정과 내부 여신 정책 위반 가능성만 검토하여 의견을 제시하세요:\n{json.dumps(case, ensure_ascii=False)}",
            system="당신은 컴플라이언스(준법감시) 심사역입니다. 금융 규정과 내부 여신 정책 위반 가능성만 집중 검토합니다.",
            model=MODELS["haiku"]
        )

    def arbiter(self, opinions: dict, case: dict) -> str:
        """중재자 에이전트: 네 의견을 종합하여 최종 판단"""
        return llm_call(
            f"""네 심사역의 의견을 종합하여 최종 여신심사 의견을 작성하세요.

[보수적 의견]
{opinions['conservative']}

[균형적 의견]
{opinions['balanced']}

[고객 중심 의견]
{opinions['progressive']}

[규정 준수 의견]
{opinions['compliance']}

[신청 내용]
{json.dumps(case, ensure_ascii=False)}

최종 의견은 승인/조건부승인/반려 중 하나를 결정하고, 그 근거를 제시하세요.
특히 규정 준수 의견에서 위반 가능성이 제기되면 반드시 반영하세요.""",
            system="당신은 최고 여신 심사 중재자입니다. 네 관점(보수·균형·고객·규정준수)을 균형있게 반영합니다.",
            model=MODELS["sonnet"],
            max_tokens=600,
        )

    def run(self, case: dict) -> dict:
        P.header(f"Debate 패턴: 대형 여신 심사 [{case.get('case_id', 'N/A')}]", "cyan")

        P.section("4개 에이전트 동시 심사")
        P.kv("  보수적 심사역", "분석 중...")
        conservative = self.conservative_reviewer(case)

        P.kv("  균형적 심사역", "분석 중...")
        balanced = self.balanced_reviewer(case)

        P.kv("  고객 중심 심사역", "분석 중...")
        progressive = self.progressive_reviewer(case)

        P.kv("  규정 준수 심사역", "분석 중...")
        compliance = self.compliance_reviewer(case)

        opinions = {
            "conservative": conservative,
            "balanced":     balanced,
            "progressive":  progressive,
            "compliance":   compliance,
        }

        P.section("중재자 최종 판단")
        final = self.arbiter(opinions, case)
        P.success("토론 심사 완료")

        return {"opinions": opinions, "final_decision": final}


# ══════════════════════════════════════════════════════════════════
# SECTION 5: 패턴 선택 가이드 함수
# ══════════════════════════════════════════════════════════════════

def recommend_pattern(
    has_dependencies: bool,
    is_parallelizable: bool,
    complexity: str,    # "low" | "medium" | "high"
    accuracy_critical: bool,
    realtime_required: bool = False,
    high_risk: bool = False,
    requires_human_review: bool = False,
) -> str:
    """
    업무 특성에 따른 최적 패턴 추천

    Examples:
        >>> recommend_pattern(True, False, "medium", False)
        'Sequential'
        >>> recommend_pattern(False, True, "low", False)
        'Parallel'
        >>> recommend_pattern(False, False, "high", True, high_risk=True, requires_human_review=True)
        'Debate + HITL'
    """
    # 고위험 결정 + 사람 검토 필요 → 토론 후 사람이 최종 승인(HITL)
    if high_risk and requires_human_review:
        return "Debate + HITL"
    if accuracy_critical and complexity == "high":
        return "Debate"
    # 실시간 처리 필요 + 병렬화 가능 → 병렬 처리로 지연 최소화
    if realtime_required and is_parallelizable:
        return "Parallel"
    if complexity == "high" and not is_parallelizable:
        return "Hierarchical"
    if is_parallelizable and not has_dependencies:
        return "Parallel"
    return "Sequential"


# ══════════════════════════════════════════════════════════════════
# SECTION 6: 메인 실행
# ══════════════════════════════════════════════════════════════════

LARGE_LOAN_CASE = {
    "case_id":       "LOAN-LARGE-001",
    "applicant":     "박대한",
    "loan_amount":   5_000_000_000,   # 50억원
    "loan_purpose":  "부동산 개발",
    "annual_income": 300_000_000,      # 3억
    "collateral":    "서울 강남구 오피스빌딩 (평가액 80억)",
    "credit_score":  720,
    "dti":           38.5,
    "ltv":           62.5,
}


def main():
    P.header("MODULE 6 실습 — 멀티에이전트 협업 패턴 4종", "blue")

    # ── Pattern 1: Sequential ──────────────────────────────────
    seq = SequentialLoanReview()
    report = seq.run("LOAN-2025-001")
    print(f"\n[Sequential 결과]\n{report}")

    # ── Pattern 2: Parallel ────────────────────────────────────
    print()
    market_data = asyncio.run(run_parallel_data_collection("005930"))  # 삼성전자
    P.section("수집된 시장 데이터")
    for market, data in market_data.items():
        P.kv(f"  {market}", str(data)[:60])

    # ── Pattern 3: Hierarchical ────────────────────────────────
    hier = HierarchicalResearchSystem()
    research = hier.run("삼성전자 2026년 하반기 투자 전망")
    print(f"\n[Hierarchical 결과]\n{research[:500]}...")

    # ── Pattern 4: Debate ─────────────────────────────────────
    debate = DebateLoanReview()
    result = debate.run(LARGE_LOAN_CASE)
    print(f"\n[Debate 최종 판단]\n{result['final_decision'][:500]}...")

    # ── 패턴 추천 데모 ─────────────────────────────────────────
    P.header("패턴 선택 가이드", "green")
    test_scenarios = [
        (True,  False, "medium", False, "여신심사 파이프라인"),
        (False, True,  "low",    False, "멀티마켓 데이터 수집"),
        (False, False, "high",   False, "투자 리서치 리포트"),
        (False, False, "high",   True,  "100억 이상 대형 여신"),
    ]
    for dep, par, comp, acc, desc in test_scenarios:
        rec = recommend_pattern(dep, par, comp, acc)
        P.kv(f"{rec:15}", desc)

    # 확장 조건 데모 (실시간 병렬 / 고위험 HITL)
    P.kv(
        f"{recommend_pattern(False, True, 'low', False, realtime_required=True):15}",
        "실시간 시세 모니터링 (실시간+병렬)",
    )
    P.kv(
        f"{recommend_pattern(False, False, 'high', True, high_risk=True, requires_human_review=True):15}",
        "100억 이상 대형 여신 (고위험+사람검토)",
    )

    P.header("MODULE 6 실습 완료", "green")


if __name__ == "__main__":
    main()