import json

import docx
import os
import re
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)

def download_model(local_model_path):
    if not os.path.exists(local_model_path):
        print("正在下载模型到本地...")
        # 这会下载模型到默认的缓存目录，然后我们复制到本地目录
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(local_model_path)
        print(f"模型已保存到: {local_model_path}")
    else:
        print(f"模型已存在于: {local_model_path}")


class LocalEmbeddingModel:
    def __init__(self, model_path="./local_models/all-MiniLM-L6-v2"):
        """
        从本地路径加载模型
        """
        self.model = SentenceTransformer(model_path)
        print(f"模型从本地加载: {model_path}")

    def encode(self, texts):
        """
        编码文本为向量
        """
        return self.model.encode(texts)





def create_collection(collection_name):
    # 创建集合

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def insert_data(collection_name, points):
    """
    插入数据
    """
    client.upsert(
        collection_name=collection_name,
        points=points
    )

def batch_insert(collection_name, points, batch_size=100):
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"已插入 {i + len(batch)} 条")

def add_index(collection_name, field_name, field_schema):
    client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=field_schema
    )



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

    # print(content_dict)

    return content_dict

# 创建数据集合
# herbs = [
#     {
#         "id": 1,
#         "name": "板蓝根",
#         "prescription": "处方",
#         "method": "制法",
#         "traits": "性状",
#         "identification": "鉴别",
#         "content_determination" : "含量测定",
#         "functional_indications": "功能主治",
#         "dosage_administration": "用法用量",
#         "specifications":"规格",
#         "storage": "贮藏",
#     }
# ]

def splicing_content(title, data_list):
    sections = []
    current_section = None

    for line in data_list[2:]:
        pattern = re.compile(
            r'(?:【(?P<title1>.+?)】|'  # 标准【标题】
            r'(?:[\d一二三四五六七八九十]+[、.)）]?\s*)(?P<title2>.+?)】)'
        )
        match = pattern.match(line)
        if match:
            current_section = match.group(1)
            content = line.split("】", 1)[1].strip()
            sections.append({
                "drug_name": title, # 药品名
                "section": current_section, # 小标题，处方、性状，。。。之类
                "content": content # 小标题对应的内容
            })
        else:
            # 追加到上一个章节
            if sections:
                sections[-1]["content"] += "\n" + line.strip()

    return  sections


def split_question(question):
    classify_template = """
    你是一个药典知识结构解析助手。

任务：
将用户的提问拆分为三个字段：

药品：
需求：
要求：

解析规则：

1. “药品”是问题中出现的中药名称或制剂名称。
2. “需求”必须映射为标准药典章节名称之一：
   - 处方
   - 制法
   - 性状
   - 鉴别
   - 检查
   - 含量测定
   - 功能与主治
   - 用法与用量
   - 规格
   - 贮藏

3. 如果用户问题中使用了同义表达，必须做映射，例如：
   - 形状 / 外观 / 长什么样 → 性状
   - 功效 / 主治 / 治什么 → 功能与主治
   - 怎么吃 / 用多少 / 服用方法 → 用法与用量
   - 成分 / 组成 → 处方
   - 怎么做的 → 制法
   - 质量标准 / 含量 → 含量测定
   - 保存方式 → 贮藏

4. “要求”用于记录额外限定条件，例如：
   - 详细说明
   - 给我全部步骤
   - 用专业术语
   - 用通俗语言解释
   如果没有额外要求，输出空字符串 ""

5. 输出格式必须严格如下（不能添加任何解释）：


你必须输出 JSON，不允许输出任何解释。

格式如下：

{
  "drug": "",
  "section": "",
  "requirement": ""
}

只允许 section 为以下值之一：
["处方","制法","性状","鉴别","检查","含量测定","功能与主治","用法与用量","规格","贮藏"]




"""
    classify_template += f"问题：{question}\n\n请按照上述规则解析问题，并输出 JSON 格式的结果。"

    response = ollama_client.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "user", "content": classify_template}
        ],
        options={
            "temperature": 0,
            "num_predict": 100
        }
    )
    print( response)
    print( response.message)
    return response.message.content.strip().lower()

def search_index(collection_name, query):
    query_vector = embedding_model.encode(query).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    )
    for point in results.points:
        print(point.payload)

def search_index_new(collection_name, query_dict):

    """
    {
      "drug": "",
      "section": "",
      "requirement": ""
    }
    """
    drug = query_dict.get("drug")
    section = query_dict.get("section")
    requirement = query_dict.get("requirement")

    return_data = []

    if len(requirement) > 1:
        query_vector = embedding_model.encode(requirement).tolist()
        print(requirement)
        print("====================")
        print(query_vector)
        results = client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="drug_name",
                        match=MatchValue(value=drug)
                    ),
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section)
                    )
                ]
            ),
            limit=5
        )
        for point in results.points:
            print(point.payload)
            return_data.append(point.payload)

    else:
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="drug_name",
                        match=MatchValue(value=drug)
                    ),
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section)
                    )
                ]
            ),
            limit=10
        )
        points = results[0]
        for point in points:
            print(point.payload)
            return_data.append(point.payload)

    return return_data

def generate_answer(question, drug_name, section, content):
    # 这里可以根据 question_dict 的内容来生成更具体的回答
    # 例如，如果 requirement 中包含 "详细说明"，可以在回答中添加更多细节
    answer_template = f"""
    你是一名中国药典领域的专业药学专家。

下面是从药典数据库中检索得到的结构化内容，请基于该内容回答用户问题。

【用户问题】
{question}

【检索结果】
药品名称：{drug_name}
章节：{section}
原始内容：
{content}

请严格按照以下要求生成最终回答：

1. 仅基于“原始内容”进行回答，不得编造、补充或推测。
2. 若原始内容中包含多个检测项目，请聚焦用户问题所涉及的具体项目。
3. 输出内容应结构清晰、语言专业、符合药典表达习惯。
4. 不要输出 JSON，不要解释数据来源。
5. 不要重复“原始内容”字样。
6. 如为含量测定类问题，回答应包含：
   - 测定方法
   - 色谱条件（如有）
   - 对照品及供试品制备方法（如有）
   - 含量限度要求

请开始生成回答。
"""
    response = ollama_client.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "user", "content": answer_template}
        ],
        options={
            "temperature": 0,
            "num_predict": 300
        }
    )
    print(response)
    print(response.message)
    return response.message.content.strip().lower()

if __name__ == '__main__':
    local_model_path = "./local_models/all-MiniLM-L6-v2"
    docx_file = r"D:\Downloads\2020年药典一部.docx"
    collection_name = "20260303_rag_zhyd_docx_content"

    download_model(local_model_path)
    # 加载模型
    embedding_model = LocalEmbeddingModel()

    # 读取文件内容
    file_data = read_docx(docx_file)

    points = [] # 创建一个空列表，用于存储向量
    global_id = 1

    for title, data_list in file_data.items():
        # 拆分整理文件内容
        sections = splicing_content(title, data_list)

        for idx, section in enumerate(sections):
            # print(f"正在处理: {title}的{section['section']}")

            text_for_embedding = f"""
                药品名称：{section['drug_name']}
                章节：{section['section']}
                内容：{section['content']}
            """

            vector = embedding_model.encode(text_for_embedding).tolist()

            points.append(
                PointStruct(
                    id=global_id,
                    vector=vector,
                    payload={
                        "drug_name": section['drug_name'],
                        "section": section['section'],
                        "content": section['content']
                    }
                )
            )

            global_id += 1

    # 创建集合
    create_collection(collection_name)
    print(f"集合创建成功: {collection_name}")
    # 插入数据
    # insert_data(collection_name, points)
    batch_insert(collection_name, points, batch_size=50)
    print(f"数据插入成功: {collection_name}")
    # 添加索引
    add_index(collection_name, "drug_name","keyword")
    add_index(collection_name, "section","keyword")
    print(f"索引添加成功: {collection_name}")


    ollama_client = ollama.Client(host="http://localhost:11434")

    # question = "板蓝根的性状是什么?"
    # question = "黄连的显微鉴别特征有哪些?"
    question = "当归中阿魏酸的含量测定方法是什么？"
    # question = "连翘的功能与主治范围是什么？"
    # question = "板蓝根颗粒儿童的用法与用量是多少？"
    # question = "一枝黄花有哪些使用禁忌？"
    # question = "人参应如何贮藏？"
    # question = "黄芩在高效液相色谱法下的含量测定条件是什么？"
    # question = "一枝黄花的炮制方法及作用是什么？"
    # question = "一枝黄花的质量控制指标有哪些？"
    question_dict = split_question(question)
    question_dict = json.loads(question_dict)
    print(question_dict)

    # search_index(collection_name, question)
    index_data = search_index_new(collection_name, question_dict)

    drug_name,section = index_data[0]["drug_name"], index_data[0]["section"]
    content = [i["content"] for i in index_data]
    content = "\n".join(content)
    print(drug_name,section)
    print(content)

    answer = generate_answer(question, drug_name, section, content)
    print(">>>>> 最终回答：")
    print(answer)







