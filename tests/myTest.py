from paperqa import Settings, Docs 
# from paperqa.agents import ask, configure_log_verbosity
from paperqa.settings import ParsingSettings,AgentSettings
import os
import argparse
from paperqa.agents.models import AnswerResponse
from paperqa.agents import configure_cli_logging
from paperqa.utils import get_loop
from paperqa.agents.main import agent_query
import logging
from typing import BinaryIO
from paperqa.types import Doc
from paperqa.readers import read_doc
import json


# 设置 PQA_HOME 环境变量
os.environ['PQA_HOME'] = '/home/ubuntu/data/paper-qa-202412'

# Function to read the prompt from a file and return it as a function input
def read_prompt_from_file(file_path):
    """Read the prompt from the specified file and return it."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    except Exception as e:
        print(f"Error reading the prompt file: {e}")
        return None


def myAsk(query: str, settings: Settings, docs : Docs) -> AnswerResponse:
    """Query PaperQA via an agent."""
    configure_cli_logging(settings)
    return get_loop().run_until_complete(
        agent_query(query, settings, agent_type=settings.agent.agent_type,docs=docs)
    )


# 假设 agent_response 是一个 PQASession 实例
def extract_agent_response(agent_response,file_name):
    data = {
        "question": agent_response.question,
        "answer": agent_response.answer,
        "context": {},
        "reference": {}
    }

    # 处理 context
    context_parts = agent_response.context.split("\n\n")
    if len(context_parts) > 1:
        for i, part in enumerate(context_parts, 1):
            data["context"][f"context {i}"] = part.strip()
    else:
        data["context"]["context 1"] = agent_response.context.strip()

    # 处理 reference
    reference_parts = agent_response.references.split("\n\n")
    if len(reference_parts) > 1:
        for i, part in enumerate(reference_parts, 1):
            data["reference"][f"reference {i}"] = part.strip()
    else:
        data["reference"]["reference 1"] = agent_response.references.strip()

    file_name = os.path.splitext(file_name)[0]

    # 指定文件路径
    file_path = f"/home/ubuntu/data/paper-qa-202412/log/response_for_{file_name}.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 将字典写入 JSON 文件
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f"提取信息已保存至 {file_path}")





# Parse command-line arguments
parser = argparse.ArgumentParser(description="Ruamn PaperQA with specified settings.")
parser.add_argument("--ref_lan", type=str, required=True, help="Specify the reference directory (en/english or zh/chinese).")
# parser.add_argument("--disease_type", type=str, required=True, help="Specify the disease type (ACS / STROKE / BOTH).")
args = parser.parse_args()
ref_language = args.ref_lan.strip().lower()
# disease_type = args.disease_type.strip()



# Set the paper directory based on the reference language
if ref_language in ["en", "english"]:
    paper_directory = "/home/ubuntu/data/paper-qa-202412/my_papers/English_ref"
elif ref_language in ["zh", "chinese"]:
    paper_directory = "/home/ubuntu/data/paper-qa-202412/my_papers/Chinese_ref"
else:
    paper_directory = "/home/ubuntu/data/paper-qa-202412/my_papers"

# if disease_type == "ACS":
#     paper_directory = os.path.join(paper_directory, "ACS")
# elif disease_type == "STROKE":
#     paper_directory = os.path.join(paper_directory, "STROKE")

# mainfest_file = os.path.join(paper_directory, "mainfest.csv")

# Create a ParsingSettings instance with document validity check disabled
parsing_settings = ParsingSettings(disable_doc_valid_check=True,use_doc_details=False)

# agent_settings = AgentSettings(agent_llm="gpt-4o-mini", agent_llm_config={"rate_limit": {"gpt-4o-min": "200000 per 1 minute"}},agent_type="fake")
agent_settings = AgentSettings(agent_llm="gpt-4o-mini", agent_llm_config={"rate_limit": {"gpt-4o-mini": "200000 per 1 minute"}})

mySettings = Settings(
        paper_directory=paper_directory,
        # manifest_file=mainfest_file,
        llm="gpt-4o-mini",
        llm_config={"rate_limit": {"gpt-4o-mini": "200000 per 1 minute"}},
        summary_llm="gpt-4o-mini",
        summary_llm_config={"rate_limit": {"gpt-4o-mini": "200000 per 1 minute"}},
        embedding="text-embedding-3-small",
        temperature=0.5,
        parsing = parsing_settings,
        verbosity = 3,
        agent = agent_settings
    )

docs_meta_file_path = '/home/ubuntu/data/paper-qa-202412/my_papers/Chinese_ref/documents_meta_ACS.json'
# 读取 JSON 文件
with open(docs_meta_file_path, "r", encoding="utf-8") as f:
    docs_meta_data = json.load(f)

# print("---------------------------------------agent-------------------------------------------")
# 将 doc_paths 和 citation 组合成三组数据
# data_group = list(zip(doc_paths, citations,docnames,dockeys,titles,dois,authors))
# for doc_path, citation, docname, dockey, title, doi, author in data_group:
#     print(docs.add(path=doc_path, citation=citation, docname=docname, dockey=dockey, title=title, doi = doi, authors = author, settings=mySettings))

# answer = myAsk(
#     query = prompt_input,
#     settings=mySettings,
#     docs = docs
# )
# print("---------------------------------------agent-------------------------------------------")

logger = logging.getLogger(__name__)
configure_cli_logging(mySettings)

print("---------------------------------------docs.query method-------------------------------------------")
# 将 doc_paths 和 citation 组合成三组数据
data_group = list(zip(docs_meta_data['doc_paths'],docs_meta_data['citations'],docs_meta_data['docnames'],docs_meta_data['dockeys'],docs_meta_data['titles'],docs_meta_data['dois'],docs_meta_data['authors']))
docs = Docs()
for doc_path, citation, docname, dockey, title, doi, author in data_group:
    docs.add(path=doc_path, citation=citation, docname=docname, dockey=dockey, title=title, doi = doi, authors = author, settings=mySettings)
    print(f"Add file '{docname}' successfully.")

# Read the prompt back as a function input
directory_path = "/home/ubuntu/data/paper-qa-202412/doctor_advice/format_prompt"
# 获取目录下所有 .txt 文件
file_list = [f for f in os.listdir(directory_path) if f.endswith(".md")]
file_list.sort()  # 按名称排序，确保顺序读取
for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)
    prompt_input = read_prompt_from_file(file_path)
    if prompt_input:
        print(f"Question loaded successfully. File name: {file_name}\n\n")

    session = docs.query(
        query = prompt_input,
        settings=mySettings,
    )
    print(session)
    extract_agent_response(session,file_name)
print("---------------------------------------docs.query method-------------------------------------------")



