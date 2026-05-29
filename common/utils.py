"""
common/utils.py
──────────────────────────────────────────────────────────────────
AI Agent 과정 공통 유틸리티 모듈
모든 모듈에서 공통으로 사용하는 함수와 상수를 정의합니다.
──────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any
from functools import wraps

from dotenv import load_dotenv

load_dotenv(override=True)  # .env 값이 OS 환경변수보다 우선하도록 override

# ── 로깅 설정 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 반환"""
    return logging.getLogger(name)


# ── API 클라이언트 팩토리 ──────────────────────────────────────────
def get_anthropic_client():
    """Anthropic 클라이언트 반환 (API 키 자동 로드)"""
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    except ImportError:
        raise ImportError("pip install anthropic 을 먼저 실행하세요.")


def get_openai_client():
    """OpenAI 클라이언트 반환"""
    try:
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except ImportError:
        raise ImportError("pip install openai 를 먼저 실행하세요.")


# ── 모델 상수 ──────────────────────────────────────────────────────
MODELS = {
    "opus":   "claude-opus-4-5",           # 높은 정확도 (여신심사)
    "sonnet": "claude-sonnet-4-6",         # 균형 (투자 리서치)
    "haiku":  "claude-haiku-4-5-20251001", # 빠름 / 저비용 (FDS, FAQ)
    "gpt-5.4":       "gpt-5.4",            # 높은 정확도 (여신심사)
    "gpt-5.4-mini":  "gpt-5.4-mini",       # 균형 (투자 리서치)
    "gpt-5.4-nano":  "gpt-5.4-nano",       # 빠름 / 저비용 (FDS, FAQ)
}

# 작업별 최적 모델 라우팅
MODEL_ROUTING = {
    "loan_review":       MODELS["opus"],
    "large_loan":        MODELS["opus"],
    "investment_report": MODELS["sonnet"],
    "customer_qa":       MODELS["sonnet"],
    "report_draft":      MODELS["sonnet"],
    "fds_check":         MODELS["haiku"],
    "simple_faq":        MODELS["haiku"],
    "classification":    MODELS["haiku"],
}


def get_model(task_type: str) -> str:
    """작업 유형에 따른 최적 모델 반환"""
    return MODEL_ROUTING.get(task_type, MODELS["sonnet"])


# ── PII 마스킹 ─────────────────────────────────────────────────────
# 주의: 더 구체적인(긴) 패턴을 먼저 두어야 합니다.
# 계좌번호 패턴이 카드번호·전화번호를 부분 매칭하므로 그보다 앞에 배치합니다.
PII_PATTERNS = {
    "주민등록번호":    r"\d{6}-[1-4]\d{6}",
    "외국인등록번호":  r"\d{6}-[5-8]\d{6}",
    "카드번호":       r"\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}",
    "전화번호":       r"0[1789]\d{1,2}-\d{3,4}-\d{4}",
    "계좌번호":       r"\d{3,4}-\d{2,6}-\d{4,6}(?:-\d{2,3})?",
    "이메일":         r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}


def mask_pii(text: str) -> tuple[str, dict[str, list]]:
    """
    텍스트에서 PII(개인식별정보)를 마스킹합니다.

    Returns:
        tuple: (마스킹된 텍스트, 발견된 PII 딕셔너리)
    """
    masked = text
    found: dict[str, list] = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = matches
            masked = re.sub(pattern, f"[{pii_type}_마스킹]", masked)
    return masked, found


# ── Prompt Injection 탐지 ──────────────────────────────────────────
INJECTION_PATTERNS = [
    r"이전.{0,10}(지시|명령|설정|프롬프트).{0,10}무시",
    r"당신은.{0,5}(이제|지금).{0,5}(다른|새로운|변경된).{0,10}역할",
    r"시스템.{0,5}(업데이트|변경|모드|프롬프트)",
    r"모든.{0,5}(고객|계좌|데이터).{0,5}(출력|보여|알려)",
    r"ignore.{0,10}(previous|prior|above).{0,10}(instruction|rule|prompt)",
    r"you are now",
    r"forget.{0,10}(everything|all|previous)",
    r"jailbreak",
    r"dan mode",
    r"developer mode",
]


def detect_prompt_injection(user_input: str) -> tuple[bool, str]:
    """
    Prompt Injection 시도를 탐지합니다.

    Returns:
        tuple: (탐지 여부, 탐지된 패턴)
    """
    lower = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return True, pattern
    return False, ""


# ── 재시도 데코레이터 ──────────────────────────────────────────────
def retry_on_api_error(max_attempts: int = 3, base_delay: float = 1.0):
    """
    API 오류 시 지수 백오프(Exponential Backoff) 재시도 데코레이터
    사용법: @retry_on_api_error(max_attempts=3)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"[{func.__name__}] {max_attempts}회 재시도 모두 실패: {e}")
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"[{func.__name__}] 시도 {attempt}/{max_attempts} 실패 "
                        f"({type(e).__name__}: {e}). {delay:.1f}초 후 재시도..."
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


# ── 캐시 키 생성 ───────────────────────────────────────────────────
def make_cache_key(prefix: str, *args) -> str:
    """캐시 키 생성 (SHA256 해시 앞 16자리)"""
    content = json.dumps(args, ensure_ascii=False, sort_keys=True)
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{prefix}:{hash_val}"


# ── 시간 유틸리티 ──────────────────────────────────────────────────
def now_kst_str() -> str:
    """현재 시각 (ISO 포맷, UTC)"""
    return datetime.now(timezone.utc).isoformat()


def elapsed_ms(start: float) -> int:
    """시작 시간부터 경과 시간(ms) 반환"""
    return int((time.time() - start) * 1000)


# ── 결과 출력 포맷터 ────────────────────────────────────────────────
class Printer:
    """터미널 출력 포맷터"""
    COLORS = {
        "blue":   "\033[94m",
        "green":  "\033[92m",
        "yellow": "\033[93m",
        "red":    "\033[91m",
        "cyan":   "\033[96m",
        "bold":   "\033[1m",
        "reset":  "\033[0m",
    }

    @classmethod
    def header(cls, text: str, color: str = "blue") -> None:
        c = cls.COLORS.get(color, "")
        r = cls.COLORS["reset"]
        b = cls.COLORS["bold"]
        print(f"\n{b}{c}{'═' * 60}{r}")
        print(f"{b}{c}  {text}{r}")
        print(f"{b}{c}{'═' * 60}{r}\n")

    @classmethod
    def section(cls, text: str, color: str = "cyan") -> None:
        c = cls.COLORS.get(color, "")
        r = cls.COLORS["reset"]
        print(f"\n{c}── {text} {'─' * (56 - len(text))}{r}")

    @classmethod
    def success(cls, text: str) -> None:
        g = cls.COLORS["green"]
        r = cls.COLORS["reset"]
        print(f"{g}✓ {text}{r}")

    @classmethod
    def warn(cls, text: str) -> None:
        y = cls.COLORS["yellow"]
        r = cls.COLORS["reset"]
        print(f"{y}⚠ {text}{r}")

    @classmethod
    def error(cls, text: str) -> None:
        red = cls.COLORS["red"]
        r = cls.COLORS["reset"]
        print(f"{red}✗ {text}{r}")

    @classmethod
    def kv(cls, key: str, value: Any) -> None:
        b = cls.COLORS["bold"]
        r = cls.COLORS["reset"]
        print(f"  {b}{key:<20}{r} {value}")