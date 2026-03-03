import docx
import os
import re

from elasticsearch import exceptions

from es_server import es

def clean_filename(filename):  # 从文件名中提取中文字符并去除末尾的空格

    return ''.join(re.findall(r'[\u4e00-\u9fff]+', filename)).rstrip()

def read_docx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    content_dict = {}
    temp_doc = []
    doc_obj = docx.Document(file_path)

    for paragraph in doc_obj.paragraphs:
        if not paragraph.runs:
            continue
        font_size = paragraph.runs[0].font.size

        # print(font_size)

        if font_size is not None:
            # print(f"Font size: {font_size.pt}, Type: {type(font_size.pt)}")

            # 确保 font_size.pt 是数字
            if isinstance(font_size.pt, (int, float)):
                if font_size.pt == 12:
                    if temp_doc:
                        title = clean_filename(temp_doc[0])
                        if title:
                            content_dict[title] = temp_doc
                        temp_doc = []
            else:
                print(f"Unexpected font size type: {type(font_size.pt)}")

        # 添加调试信息，检查段落文本
        # print(f"Adding paragraph text: {paragraph.text}")

        # 将段落文本添加到临时文档
        temp_doc.append(paragraph.text)


        # 处理最后一个段落
    if temp_doc:
        title = clean_filename(temp_doc[0])
        if title:
            content_dict[title] = temp_doc

    print(content_dict)

    return content_dict


def save_data_to_es(content_dict):
    index_name = "20260213_rag_zhyd_docx_content"
    for title, content in content_dict.items():
        try:
            print(f"正在存储: {title}到{index_name},content: {content}")
            # es.index(index=index_name, id=title, body={'content': '\n'.join(content)})
            # print(f"已存储: {title}到{index_name}")
        except exceptions.ConnectionError as e:
            print(f"连接错误：{e}")
        except exceptions.TransportError as e:
            print(f"存储错误：{e}")


if __name__ == '__main__':
    docx_file = r"D:\Downloads\2020年药典一部.docx"
    content_dict = read_docx(docx_file)
    # print(content_dict)
    save_data_to_es(content_dict)