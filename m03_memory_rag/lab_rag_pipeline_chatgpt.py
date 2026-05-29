"""
m03_memory_rag/lab_rag_pipeline_chatgpt.py
==================================================================
MODULE 3 실습: 금융 규정 RAG Q&A 에이전트 (ChatGPT)
==================================================================
실습 목표:
  1. 금융 규정 문서를 청크로 분할하고 FAISS 벡터 DB로 구성한다
  2. OpenAI 임베딩 모델을 적용한다
  3. MMR 검색으로 다양한 관련 컨텍스트를 찾는다
  4. OpenAI ChatGPT를 사용해 근거 조항 기반 RAG Q&A를 구현한다

실행 방법:
  python m03_memory_rag/lab_rag_pipeline_chatgpt.py

사전 준비:
  pip install langchain langchain-community langchain-openai
  pip install faiss-cpu pypdf
  .env 파일에 OPENAI_API_KEY 설정 필요
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.utils import get_logger, Printer

logger = get_logger(__name__)
P = Printer()

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)


def _embedding_store_suffix() -> str:
    return OPENAI_EMBEDDING_MODEL.replace("/", "_").replace(":", "_")


# ==================================================================
# SECTION 1: 샘플 금융 규정 텍스트
# ==================================================================

SAMPLE_REGULATION_TEXT = """
대출업무처리기준

제1장 총칙

제1조(목적)
본 기준은 대출 업무의 적정한 처리와 금융소비자 보호를 위하여 필요한 사항을 정함을 목적으로 한다.

제2조(적용 범위)
본 기준은 개인 및 법인을 대상으로 하는 모든 대출 업무에 적용한다.
다만 정책 목적 대출상품 및 특별 정책 대출에는 별도 기준을 우선 적용할 수 있다.

제2장 여신 심사 기준

제3조(신용대출 한도 산정 기준)
신용대출 한도는 다음 각 호의 기준에 따라 산정한다.
  1. 연소득의 150% 이내를 원칙으로 한다.
  2. 신용점수 700점 이상인 경우 연소득의 200%까지 허용할 수 있다.
  3. 기존 대출 잔액을 차감하여 최종 한도를 산정한다.
신용점수 600점 미만인 경우 신용대출을 원칙적으로 제한한다.
재직기간이 1년 미만인 경우 신용대출 한도를 50% 감액할 수 있다.

제4조(DTI 산정 기준)
총부채상환비율(DTI)은 연간 총부채 원리금 상환액을 연소득으로 나눈 비율이다.
주택담보대출의 경우 DTI 40% 이하를 원칙으로 한다.
투기과열지구의 경우 DTI 30% 이하를 적용한다.
DTI 산정 시 모든 금융기관의 대출 원리금을 합산한다.

제5조(LTV 산정 기준)
주택담보대출비율(LTV)은 대출금액을 담보물 평가액으로 나눈 비율이다.
비규제지역의 경우 LTV 70% 이하를 적용한다.
규제지역의 경우 LTV 50% 이하를 적용한다.
투기과열지구의 경우 LTV 40% 이하를 적용한다.
담보물 평가액은 시가감정액 또는 금융감독원이 인정한 감정평가법인의 평가액을 사용한다.

제3장 대출 신청 및 심사 절차

제6조(대출 신청 서류)
대출 신청인은 다음 각 호의 서류를 제출하여야 한다.
  1. 대출신청서
  2. 신분증 사본 (주민등록증 또는 여권)
  3. 소득 증빙 서류: 근로소득원천징수영수증, 사업소득확인원 등
  4. 담보 관련 서류: 등기사항전부증명서, 감정평가서 등
서류 미비 시 영업일 기준 5일 이내 보완을 요청한다.

제7조(심사 기간)
표준 심사 기간은 신청일로부터 영업일 기준 3일 이내로 한다.
추가 서류 요청 기간은 해당 기간에서 제외한다.
대출금액 1억원 초과 건은 영업일 기준 5일 이내로 할 수 있다.

제4장 이자율 및 수수료

제8조(대출 금리 산정)
대출금리는 기준금리에 가산금리를 합산하여 적용금리를 산정한다.
가산금리는 신용등급, 담보 종류, 대출 기간에 따라 차등 적용한다.
우대금리는 급여이체, 자동이체 등록, 카드 실적 등에 따라 최대 0.5%p 적용할 수 있다.

제9조(중도상환 수수료)
대출 실행 후 3년 이내 중도상환 시 중도상환 수수료를 부과한다.
수수료율은 1년 이내 1.5%, 1~2년 이내 1.0%, 2~3년 이내 0.5%로 한다.
정책자금, 전세자금, 서민금융 대출은 중도상환 수수료를 면제한다.

제5장 사후 관리

제10조(연체 관리)
원리금 납입일 다음날부터 연체가 시작된다.
연체금리는 대출금리에 3%p를 가산하여 적용하되 최고 20%를 초과하지 않는다.
90일 이상 연체 시 기한이익상실을 통보하고 전액 상환을 요청할 수 있다.
"""


# ==================================================================
# SECTION 2: 지식 베이스 구축 함수
# ==================================================================

def _create_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def build_knowledge_base_from_text(
    text: str,
    save_path: str = "./finance_kb",
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> "FAISS":
    """텍스트에서 FAISS 벡터 DB를 구성한다."""
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            f"필수 패키지 미설치: {exc}\n"
            "pip install langchain langchain-community langchain-openai faiss-cpu"
        )

    P.section("1단계: 텍스트 청크 분할")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n제10조",
            "\n제9조",
            "\n제8조",
            "\n제7조",
            "\n제6조",
            "\n제5조",
            "\n제4조",
            "\n제3조",
            "\n제2조",
            "\n제1조",
            "\n\n",
            "\n",
            " ",
        ],
        length_function=len,
    )

    docs = [
        Document(
            page_content=text,
            metadata={"source": "대출업무처리기준", "page": 1},
        )
    ]
    chunks = splitter.split_documents(docs)
    P.success(f"{len(chunks)}개 청크 생성 완료")

    for i, chunk in enumerate(chunks[:3]):
        print(f"  청크 {i + 1}: {chunk.page_content[:60]}...")

    P.section("2단계: OpenAI 임베딩 모델 로드")
    print(f"  OpenAI embedding model: {OPENAI_EMBEDDING_MODEL}")
    embeddings = _create_embeddings()
    P.success("OpenAI 임베딩 모델 로드 완료")

    P.section("3단계: FAISS 벡터 DB 생성")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    P.success(f"벡터 DB 생성 완료: {vectorstore.index.ntotal}개 벡터")

    vectorstore.save_local(save_path)
    P.success(f"벡터 DB 저장: {save_path}/")
    return vectorstore


def build_knowledge_base_from_pdf(
    pdf_paths: list[str],
    save_path: str = "./finance_kb",
) -> "FAISS":
    """PDF 파일에서 FAISS 벡터 DB를 구성한다."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(f"필수 패키지 미설치: {exc}")

    P.section("PDF 로드")
    all_docs = []
    for path in pdf_paths:
        if not Path(path).exists():
            P.warn(f"파일 없음, 건너뜀: {path}")
            continue

        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = Path(path).name
        all_docs.extend(docs)
        P.success(f"로드 완료: {path} ({len(docs)}페이지)")

    if not all_docs:
        raise ValueError("유효한 PDF 파일이 없습니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n제", "\n장", "\n절", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(all_docs)
    P.success(f"{len(chunks)}개 청크 생성")

    embeddings = _create_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    P.success(f"벡터 DB 저장: {save_path}/")
    return vectorstore


def load_knowledge_base(save_path: str = "./finance_kb") -> Optional["FAISS"]:
    """저장된 FAISS 벡터 DB를 로드한다."""
    try:
        from langchain_community.vectorstores import FAISS

        if not Path(save_path).exists():
            return None

        embeddings = _create_embeddings()
        vectorstore = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        P.success(f"기존 벡터 DB 로드 완료: {vectorstore.index.ntotal}개 벡터")
        return vectorstore
    except Exception as exc:
        logger.warning("벡터 DB 로드 실패: %s", exc)
        return None


# ==================================================================
# SECTION 3: RAG Q&A 체인 구성
# ==================================================================

REGULATION_QA_PROMPT_TEMPLATE = """당신은 금융 규정 전문가입니다.
아래 제공된 관련 규정 내용을 바탕으로 질문에 정확하게 답변하세요.

[필수 지침]
1. 답변에는 반드시 관련 조항 번호(예: 제3조)를 인용하세요.
2. 제공된 규정 내용에서 확인되지 않는 사항은 "해당 규정에서 확인되지 않습니다"라고 명시하세요.
3. 추측하거나 임의로 규정을 추가하지 마세요.
4. 수치 정보(%, 금액, 기간)는 원문 그대로 정확히 인용하세요.

[관련 규정 내용]
{context}

[질문]
{question}

[답변] (관련 조항 명시 필수):"""


def create_regulation_qa_chain(vectorstore: "FAISS") -> "RetrievalQA":
    """RAG Q&A 체인을 구성한다."""
    try:
        from langchain_classic.chains import RetrievalQA
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            f"필수 패키지 미설치: {exc}\n"
            "pip install langchain langchain-community langchain-openai langchain-classic"
        )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 15,
            "lambda_mult": 0.7,
        },
    )

    prompt = PromptTemplate(
        template=REGULATION_QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ==================================================================
# SECTION 4: 질문-답변 및 출처 표시
# ==================================================================

def ask_regulation(qa_chain: "RetrievalQA", question: str) -> dict:
    """규정 Q&A를 실행하고 답변과 출처를 반환한다."""
    import time

    start = time.time()
    result = qa_chain.invoke({"query": question})
    elapsed = int((time.time() - start) * 1000)

    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "규정 문서"))
        page = doc.metadata.get("page", 1)
        key = f"{source}_p{page}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source,
                "page": page,
                "preview": doc.page_content[:100].replace("\n", " ") + "...",
            }
        )

    return {
        "question": question,
        "answer": result["result"],
        "sources": sources,
        "elapsed_ms": elapsed,
    }


def print_qa_result(result: dict) -> None:
    """Q&A 결과를 보기 좋게 출력한다."""
    print(f"\n{'=' * 60}")
    print(f"질문: {result['question']}")
    print(f"{'=' * 60}")
    print(f"\n답변:\n{result['answer']}")
    print(f"\n참고 출처 ({len(result['sources'])}건):")
    for src in result["sources"]:
        print(f"  - {src['source']} p.{src['page']}")
        print(f"    {src['preview']}")
    P.kv("처리 시간", f"{result['elapsed_ms']}ms")


# ==================================================================
# SECTION 5: 대화 메모리 관리 예시
# ==================================================================

def demo_conversation_memory() -> None:
    """대화 메모리 관리 데모"""
    P.section("대화 메모리 관리 데모")

    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError:
        P.warn("langchain-openai 미설치로 메모리 데모를 건너뜁니다.")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chat_history = [
        HumanMessage(content="DTI媛 臾댁뾿?멸???"),
        AIMessage(content="DTI???곌컙 珥앸?梨??먮━湲??곹솚?≪쓣 ?곗냼?앹쑝濡??섎늿 鍮꾩쑉?낅땲??"),
        HumanMessage(content="二쇳깮?대낫?異쒖쓽 DTI 湲곗???"),
        AIMessage(content="?쇰컲 吏??? 40%, ?ш린怨쇱뿴吏援щ뒗 30% ?댄븯瑜??곸슜?⑸땲??"),
    ]

    summary_prompt = ChatPromptTemplate.from_messages(
        {"input": "DTI가 무엇인가요?"},
        {"output": "DTI는 연간 총부채 원리금 상환액을 연소득으로 나눈 비율입니다."},
    )
    memory.save_context(
        {"input": "주택담보대출의 DTI 기준은?"},
        {"output": "일반 지역은 40%, 투기과열지구는 30% 이하를 적용합니다."},
    )

    mem_state = memory.load_memory_variables({})
    print("현재 메모리 상태:")
    for msg in mem_state.get("chat_history", []):
        role = "Human" if getattr(msg, "type", "") == "human" else "AI"
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  {role}: {content}")


# ==================================================================
# SECTION 6: 메인 실행
# ==================================================================

# NOTE:
# The function below intentionally redefines the earlier demo helper to avoid
# deprecated LangChain memory APIs while keeping the same external behavior.
def demo_conversation_memory() -> None:
    """대화 메모리 관리 데모"""
    P.section("대화 메모리 관리 데모")

    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError:
        P.warn("langchain-openai 미설치로 메모리 데모를 건너뜁니다.")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chat_history = [
        HumanMessage(content="DTI가 무엇인가요?"),
        AIMessage(content="DTI는 연간 총부채 원리금 상환액을 연소득으로 나눈 비율입니다."),
        HumanMessage(content="주택담보대출의 DTI 기준은?"),
        AIMessage(content="일반 지역은 40%, 투기과열지구는 30% 이하를 적용합니다."),
    ]

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "다음 대화 이력을 2문장 이내로 요약하세요. 수치와 기준은 정확히 유지하세요.",
            ),
            ("human", "대화 이력:\n{history}"),
        ]
    )
    history_text = "\n".join(
        f"{'Human' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in chat_history
    )
    summary = (summary_prompt | llm).invoke({"history": history_text}).content

    print("현재 메모리 상태:")
    for msg in chat_history:
        role = "Human" if msg.type == "human" else "AI"
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  {role}: {content}")
    print(f"  Summary: {summary}")


TEST_QUESTIONS = [
    "신용대출 한도 산정 기준은 무엇인가요?",
    "DTI 40%를 초과하면 대출을 받을 수 없나요?",
    "주택담보대출 LTV 한도는 얼마인가요?",
    "대출 신청 시 제출해야 하는 서류는 무엇인가요?",
    "중도상환 수수료는 얼마인가요?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="금융 규정 RAG Q&A 에이전트 데모",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="사용자가 직접 질문을 입력하는 대화형 모드를 실행합니다.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="기본 샘플 질문 실행을 건너뜁니다.",
    )
    return parser.parse_args()


def run_demo_questions(qa_chain) -> None:
    P.section("Demo Q&A")
    for question in TEST_QUESTIONS:
        try:
            result = ask_regulation(qa_chain, question)
            print_qa_result(result)
        except Exception as exc:
            P.error(f"Q&A 실패: {question[:30]}... -> {exc}")


def run_interactive_session(qa_chain) -> None:
    P.section("Interactive Q&A")
    print("질문을 입력하세요. 종료하려면 빈 줄 또는 'exit', 'quit', 'q'를 입력하세요.")

    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화형 모드를 종료합니다.")
            break

        if not question or question.lower() in {"exit", "quit", "q"}:
            print("대화형 모드를 종료합니다.")
            break

        try:
            result = ask_regulation(qa_chain, question)
            print_qa_result(result)
        except Exception as exc:
            P.error(f"Q&A ?ㅽ뙣: {question[:30]}... -> {exc}")


def run_demo_questions(qa_chain) -> None:
    P.section("Demo Q&A")
    for question in TEST_QUESTIONS:
        try:
            result = ask_regulation(qa_chain, question)
            print_qa_result(result)
        except Exception as exc:
            P.error(f"Q&A ?ㅽ뙣: {question[:30]}... -> {exc}")


def main() -> None:
    args = parse_args()
    P.header("MODULE 3 실습 - 금융 규정 RAG Q&A 에이전트 (ChatGPT)", "blue")

    kb_path = f"./finance_kb_demo_{_embedding_store_suffix()}"

    P.section("벡터 DB 준비")
    vectorstore = load_knowledge_base(kb_path)
    if vectorstore is None:
        print("기존 벡터 DB가 없습니다. 샘플 텍스트로 새로 생성합니다.")
        vectorstore = build_knowledge_base_from_text(
            SAMPLE_REGULATION_TEXT,
            save_path=kb_path,
        )
    else:
        print("기존 벡터 DB를 재사용합니다.")

    P.section("RAG Q&A 체인 생성")
    try:
        qa_chain = create_regulation_qa_chain(vectorstore)
        P.success("RAG 체인 준비 완료")
    except Exception as exc:
        P.error(f"체인 생성 실패: {exc}")
        P.warn("OPENAI_API_KEY 및 langchain-openai 설치를 확인하세요.")
        return

    P.section("규정 Q&A 테스트")
    for question in TEST_QUESTIONS:
        try:
            result = ask_regulation(qa_chain, question)
            print_qa_result(result)
        except Exception as exc:
            P.error(f"Q&A 실패: {question[:30]}... -> {exc}")

    demo_conversation_memory()
    P.header("MODULE 3 실습 완료", "green")


def main() -> None:
    args = parse_args()
    P.header("MODULE 3 ?ㅼ뒿 - 湲덉쑖 洹쒖젙 RAG Q&A ?먯씠?꾪듃 (ChatGPT)", "blue")

    kb_path = f"./finance_kb_demo_{_embedding_store_suffix()}"

    P.section("벡터 DB 준비")
    vectorstore = load_knowledge_base(kb_path)
    if vectorstore is None:
        print("기존 벡터 DB가 없습니다. 샘플 텍스트로 새로 생성합니다.")
        vectorstore = build_knowledge_base_from_text(
            SAMPLE_REGULATION_TEXT,
            save_path=kb_path,
        )
    else:
        print("기존 벡터 DB를 재사용합니다.")

    P.section("RAG Q&A 체인 생성")
    try:
        qa_chain = create_regulation_qa_chain(vectorstore)
        P.success("RAG 체인 준비 완료")
    except Exception as exc:
        P.error(f"체인 생성 실패: {exc}")
        P.warn("OPENAI_API_KEY 및 langchain-openai 설치를 확인하세요.")
        return

    if not args.skip_demo:
        run_demo_questions(qa_chain)

    if args.interactive:
        run_interactive_session(qa_chain)

    demo_conversation_memory()
    P.header("MODULE 3 실습 완료", "green")


if __name__ == "__main__":
    main()