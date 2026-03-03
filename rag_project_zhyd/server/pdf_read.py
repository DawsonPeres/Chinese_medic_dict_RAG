import pdfplumber

import pdfplumber
import re


def extract_dual_column_pdf(pdf_path: str) -> dict:
    """精确提取双栏PDF（如药典格式）"""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]

        # 定义栏位边界（根据实际PDF调整）
        left_col_x_start, left_col_x_end = 40, 320  # 左栏范围
        right_col_x_start, right_col_x_end = 350, 600  # 右栏范围

        left_text = ""
        right_text = ""

        # 按Y坐标排序，确保上下文连续
        blocks = sorted(page._blocks, key=lambda x: x['top'])

        for block in blocks:
            x0, x1 = block['x0'], block['x1']
            text = block['text'].strip()

            if not text:
                continue

            # 判断属于哪一栏
            if x0 >= left_col_x_start and x1 <= left_col_x_end:
                left_text += text + "\n"
            elif x0 >= right_col_x_start and x1 <= right_col_x_end:
                right_text += text + "\n"
            else:
                # 处理跨栏文本（如标题）
                if "【" in text or "】" in text:
                    left_text += text + "\n"
            print()
            return {
                "left": left_text.strip(),
                "right": right_text.strip()
            }

if __name__ == '__main__':
    pdf_ = r"D:\Downloads\2020年药典一部.pdf"
    text = extract_dual_column_pdf(pdf_)
    print(text)

