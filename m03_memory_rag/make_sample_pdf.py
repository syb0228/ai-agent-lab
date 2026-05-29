"""
샘플 규정 PDF 생성 스크립트 (미션 4용)
SAMPLE_REGULATION_TEXT를 data/대출업무처리기준.pdf 로 변환한다.

실행: python m03_memory_rag/make_sample_pdf.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fpdf import FPDF

from m03_memory_rag.lab_rag_pipeline_chatgpt import SAMPLE_REGULATION_TEXT

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"  # 윈도우 기본 한글 폰트


def make_pdf(out_path: str = "./data/대출업무처리기준.pdf") -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.add_font("malgun", "", FONT_PATH)  # 유니코드 TTF 등록
    pdf.set_font("malgun", size=11)

    width = pdf.epw  # 좌우 여백을 제외한 유효 페이지 폭
    for line in SAMPLE_REGULATION_TEXT.strip().splitlines():
        # 빈 줄은 간격으로, 일반 줄은 멀티셀로 출력
        if line.strip():
            pdf.multi_cell(width, 7, line, wrapmode="CHAR")
        else:
            pdf.ln(3)

    pdf.output(out_path)
    return out_path


if __name__ == "__main__":
    path = make_pdf()
    print(f"PDF 생성 완료: {path}")
