# elasticsearch
import re

from elasticsearch import Elasticsearch, exceptions



ES_HOST = "localhost"
ES_PORT = 9200

es = Elasticsearch([{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}])


def search_index(index_name, query):
    # response = es.search(index=index_name)
    # response = es.get(index=index_name, id=query)
    response = es.search(index=index_name, body={"query": {"match": {"content": query}}})
    return response

def retrieve_data_from_es(index_name, query):# 从Elasticsearch中检索数据
    response = es.search(index=index_name, body={"query": {"match": {"content": query}}})
    return response['hits']['hits']

def extract_subsections_new(content):
    # 匹配：
    # 【标题】
    # 1标题】
    # （1）标题】
    # 一、标题】
    pattern = re.compile(
        r'(?:【(?P<title1>.+?)】|'           # 标准【标题】
        r'(?:[\d一二三四五六七八九十]+[、.)）]?\s*)(?P<title2>.+?)】)'
    )

    matches = list(pattern.finditer(content))

    subsections = {}

    for i, match in enumerate(matches):

        title = match.group("title1") or match.group("title2")
        title = title.strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        subsections[title] = content[start:end].strip()
    # print(subsections)
    return subsections

def verify_data_in_elasticsearch(index_name, doc_id, sub_titles):
    # 验证数据是否在Elasticsearch中，并且返回结果
    output = []  # 存储输出结果

    try:
        #    检查索引是否存在
        response = es.get(index=index_name, id=doc_id)
        content = response['_source']['content']
        # print("文档内容：", content)
        #    提取所有小标题及其内容
        subsections = extract_subsections_new(content)

        found_content = []

        #    检查每个小标题是否存在于提取的内容中
        for sub_title in sub_titles:
            if sub_title in subsections and subsections[sub_title]:
                found_content.append(f"小标题 '{sub_title}' 的内容:\n{subsections[sub_title]}")

        #    输出结果
        if found_content:
            output.append("\n".join(found_content))
        else:
            #    没有找到任何小标题，输出完整内容
            output.append(f"未找到任何小标题，输出完整内容:\n{content}")

    except exceptions.NotFoundError:
        print(f"文档 ID '{doc_id}' 或索引 '{index_name}' 不存在。")
    except exceptions.TransportError as e:
        print(f"查询错误：{e}")

    return "\n".join(output)  # 以换行符连接输出结果


if __name__ == '__main__':
    index_name = "20260213_rag_zhyd_docx_content"
    query = "山麦冬"
    # search_results = search_index(index_name, query)
    # print(search_results)
    search_results_ = retrieve_data_from_es(index_name, query)
    print(search_results_)