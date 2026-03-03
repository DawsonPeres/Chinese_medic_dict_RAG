import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_project_zhyd.server.es_server import retrieve_data_from_es



def extract_subsections(content):  # 提取小标题和内容

    pattern = re.compile(r'(?:【|t)(.+?)(?:】)')
    matches = pattern.finditer(content)

    subsections = {}
    last_position = 0
    last_title = None

    for match in matches:
        # print(match.group())
        title = match.group().strip('【】t]')
        if last_title:
            subsections[last_title] = content[last_position:match.start()].strip()

        last_title = title
        last_position = match.end()

    if last_title:
        subsections[last_title] = content[last_position:].strip()
    # print(subsections)
    return subsections

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


def embedding_function(index_name, query):
    # 将输入文本转换为向量
    # 根据索引查询es中的原始数据
    search_results = retrieve_data_from_es(index_name, query)

    # 原始数据 药材：描述
    index_data_dict = {}
    # [(药材id，小标题，内容)]
    complete_data_list = []
    # [药材id]
    ids_data_list = []
    # [(小标题，内容)]
    texts_data_list = []


    for entry in search_results:
        doc_id = entry['_id']  # 文档ID
        content = entry['_source']['content']  # 文档内容
        # print(f"Document ID: {doc_id} - Content : {content}")
        subsections = extract_subsections_new(content)
        # print(content)
        index_data_dict[doc_id] = content

        # content
        for title, content in subsections.items():
            complete_data_list.append((doc_id, title, content))
            ids_data_list.append(doc_id)
            texts_data_list.append((title, content))
    # return
    # 数据转向量
    embeddings = model.encode([text for _,_,text in complete_data_list], convert_to_numpy=True)

    # 创建FAISS索引
    dimension = embeddings.shape[1] # 向量维度
    index = faiss.IndexFlatL2(dimension) # 创建索引
    index.add(embeddings.astype(np.float32)) # 将向量添加到索引中

    np.savez_compressed(vector_db_path, embeddings=embeddings, index=index, ids=ids_data_list, texts=texts_data_list)
    print("Data processed and saved to disk.")

def retrieve_vector_and_text(input_data, embedding_file_path, top_k=1):
    if not os.path.exists(embedding_file_path):
        raise FileNotFoundError(f"Embedding file not found at: {embedding_file_path}")

    # 将输入文本转换为向量
    model_ = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = model_.encode([input_data], convert_to_numpy=True)
    # 加载预先计算的向量和文本数据
    data = np.load(embedding_file_path, allow_pickle=True)
    # print(data.files)  # 输出文件中的所有数组名称，检查是否包含 'embeddings', 'ids', 'texts'
    embeddings = data['embeddings'] # 加载向量数据
    ids = data['ids'] # 加载ID数据
    texts = data['texts']  # 这里加载文本信息
    dimension = embeddings.shape[1] # 向量维度
    index = faiss.IndexFlatL2(dimension) # 创建索引
    index.add(embeddings.astype(np.float32)) # 将向量添加到索引中
    # 查询相似向量
    D, I = index.search(query_embedding.astype(np.float32), top_k)
    retrieved_ids = ids[I[0]].tolist() # 获取对应的ID
    retrieved_texts = [texts[i] for i in I[0]]  # 获取对应的文本信息
    print(f'Retrieved IDs: {retrieved_ids}')
    print(f'Retrieved Texts: {retrieved_texts}')
    results = [(retrieved_ids[i], retrieved_texts[i][0], retrieved_texts[i][1]) for i in range(top_k)]
    # print("这里是results的打印")

    return results


if __name__ == '__main__':
    index_name = "20260213_rag_zhyd_docx_content"
    query = "山麦冬"
    vector_db_path = f"D:\Study\RAG\Chinese_medic_dict_RAG\.{index_name}.npz"
    # 向量化模型
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # embedding_function(index_name, query)
    message_ = "请给出关于“板蓝根”的性状与贮藏方法"
    # retrieve_vector_and_text(message_, vector_db_path, top_k=1)