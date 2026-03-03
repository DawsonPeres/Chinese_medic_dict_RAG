import re

import ollama

from rag_project_zhyd.server.es_server import verify_data_in_elasticsearch
from rag_project_zhyd.server.vector_server import retrieve_vector_and_text

client = ollama.Client(host="http://localhost:11434")

field_list=["药物名","类别", "鉴别", "贮藏", "指纹图谱", "功能主治", "规格", #文档中会出现的小标题
                             "含量测定",  "性味与归经", "浸岀物",
                             "规定", "制法",  "检査", "用法与用量",
                             "用途",  "触藏", "正丁醇提取物", "特征图谱","禁忌",
                              "效价测定", "正丁醇浸出物",
                             "注意事项", "功能与主治", "制剂",
                             "性状","挥发油","处方",
                             "适应症"]

def question_type_classifier(message):
    classify_template = f"""你是药学专家，请判断以下问题是否与药品或药学相关：
            {message}
            如果这是一个与药品或药学相关的问题（如提问药品的功能、用途、副作用等），返回 "good"；
            如果不是药学相关的问题，返回 "bad"。
            只能返回“good”或者“bad”
            """

    response = client.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "user", "content": classify_template}
        ],
        options={
            "temperature": 0,
            "num_predict": 20
        }
    )
    # print( response)
    # print( response.message)
    return response.message.content.strip().lower()

def analysis_and_cut(message):
    # 分析信息，并依照格式准确输出内容
    extract_template = f"""从以下药物信息中总结字段，回答这句话需要使用到哪些字段，以及该药品的药品名，不用赘述其他：
            {message}
            字段列表：{', '.join(field_list)}

            请返回以下格式的结果：
            提到的药品名：药品名
            标准化输出：
            字段名
            字段名
            例如：
            提到的药品名：八角茴香
            标准化输出：
            功能主治
            性状
            """

    response = client.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "user", "content": extract_template}
        ],
        options={
            "temperature": 0,
            "num_predict": 20
        }
    )
    # print( response)
    # print( response.message)
    return response.message.content.strip()

def analysis_and_output_content(message):
    text = '''问题一般是药品相关的问题，所以字段可能会有近义语句，
            比如制法即是制作方法,有重量的是处方的一部分，“用*制作而成”*也一般是处方，处方中通常只有药材名例如“板蓝根，罂粟壳”，无其他说明
            只有提到的字段才可以出现'''
    all_fields_str = ", ".join(field_list)
    extract_template = f"""你是个语意理解大师，你需要充分理解问题中的内容含义，他的问题提到了哪些信息，他的问题通常答案指向一种药物的名称，
            所以问题中提到的药物有克数的一般为处方中的内容。你需要把问题中的信息分类到字段中提到的内容中
            问题：{message}
            字段：{all_fields_str} 

            用中文回复以及中文字符，回复时参考以下格式，比如 处方：板蓝根1500g,大青叶2250g。将涉及的字段与信息全部输出，顺序为字段名，提到的字段，
            同时，未提到的字段不需要输出字段名：信息
            额外信息：{text}
            未涉及的字段一定不要提到。一定不要出现“字段：None”的类似句子通常来说问题中只有2-3个字段内容，确保你不会输出超过3个字段，字段间换行输出
            """
    response = client.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "user", "content": extract_template}
        ],
        options={
            "temperature": 0,
            "num_predict": 20
        }
    )
    # print( response)
    # print( response.message)
    return response.message.content.strip()

def request_answer(message):
    qa_template = f"""你是一个药典问答机器人，请回答以下问题：
        {message}
        请给出具体的回答。"""
    response = client.chat(
        model="qwen2.5:7b-instruct",
        messages=[{"role": "user", "content": qa_template}],
        options={
            "temperature": 0,
            "num_predict": 20
        }
    )
    print('************************************************************')
    print("这里是response的打印")
    print( response)
    print( response.message)
    return response.message.content.strip()

def extract_drug_info(text):
    # 将标准化后的信息切分
    # 匹配多个药品名和标准化输出
    drug_pattern = re.compile(r'提到的药品名：(.+?)\s+标准化输出：\s*(.+?)(?=(提到的药品名：|$))', re.DOTALL)
    matches = drug_pattern.findall(text)

    drugs = []
    standard_outputs = []

    for match in matches:
        # 提取药品名，可能包含多个药品，以逗号或其他标点分隔
        drug_names = [name.strip() for name in re.split(r'[、,，]', match[0]) if name.strip()]
        # 提取并分割标准化输出的每一行
        outputs = [line.strip() for line in match[1].strip().split('\n') if line.strip()]

        for drug_name in drug_names:
            drugs.append(drug_name)
            standard_outputs.append(outputs)

    return drugs, standard_outputs


if __name__ == '__main__':
    index_name = "20260213_rag_zhyd_docx_content"
    vector_db_path = f"D:\Study\RAG\Chinese_medic_dict_RAG\.{index_name}.npz"
    message_ = "请给出关于“板蓝根”的性状与贮藏方法"
    question_type = question_type_classifier(message_)
    question_result = None
    if question_type == "good":
        question_result = analysis_and_cut(message_)
        print(question_result)
    else:
        print("该问题与药品或药学无关")

    if question_result:
        drugs, standard_outputs = extract_drug_info(question_result)
        print("提到的药品名：", drugs)
        print("子标题：", standard_outputs)
        # 提到的药品名： ['阿司匹林']
        # 子标题： [['用法与用量']]
        combined_results = []
        for id, outputs in zip(drugs, standard_outputs):
            print(f"当前ID: {id}, 输出: {outputs}")
            for sub in outputs:
                print(f"检索子标题: {sub}")
                sub_data = verify_data_in_elasticsearch(index_name=index_name,doc_id=id,sub_titles=[sub])
                print(f"子标题数据: {sub_data}")
                combined_results.append(sub_data)

        unique_content = list(set(combined_results))
        final_output = "\n\n".join(unique_content) + "\n"
        print(f"es的检索结果: {final_output}")  # 打印最终检索结果
        print("----------------------------------------------------------")
        output_content = analysis_and_output_content(message_)
        print(output_content)
        output_lines = []
        # 向量检索的结果
        for line in output_content.splitlines():
            results = retrieve_vector_and_text(line.strip(), vector_db_path, top_k=1)
            for doc_id, title, text in results:
                output_lines.append(f"Document ID: {doc_id}, Title: {title}, Text: {text}")
        output_result = "\n".join(output_lines)  # 向量化检索的结果
        print(final_output)
        final_output += output_result + "\n"  # 向量化检索的结果加入到es的检索结果中

        # context = "\n".join([f"用户: {msg}\n助手: {resp}" for msg, resp in history])
        # final_query = (f"这里是上下文：{context}\n\n这是你要回答的问题{message_}请使用提供的数据信息进行回答:'{final_output}'."
        final_query = (f"这是你要回答的问题{message_}请使用提供的数据信息进行回答:'{final_output}'."
                       f"不要瞎编，涉及的敏感词请替换成同义词。可以润色内容，使其贴切或易懂,只回答我提的问题，问题没问的哪怕给了信息，"
                       f"也不用回答,可以结合用户和助手的对话内容，不要重复回答。")

        print(f"最终查询: {final_query}")  # 调试信息
        answer = request_answer(final_query)
        print(answer)
