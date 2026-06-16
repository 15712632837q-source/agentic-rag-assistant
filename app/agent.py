"""Agentic RAG：ReActAgent 自主决定【何时/是否】检索知识库，再带出处作答。

不是朴素 RAG（每问必检索一次）——agent 会思考要不要查、查什么、够不够再答，
这正是"Agentic"与普通 RAG 的区别（也是岗位想看到的点）。
"""

import os

from . import config


def _load_or_build_index():
    from llama_index.core import StorageContext, load_index_from_storage

    config.configure_llamaindex()
    if os.path.isdir(config.STORAGE_DIR) and os.listdir(config.STORAGE_DIR):
        sc = StorageContext.from_defaults(persist_dir=config.STORAGE_DIR)
        return load_index_from_storage(sc)
    print("[agent] 未发现索引，先建索引 ...")
    from .ingest import build_index
    return build_index()


def build_agent(verbose: bool = True):
    """返回新版 workflow ReActAgent：agent 自主决定是否/如何检索知识库再作答。

    LlamaIndex 0.14+ 用 llama_index.core.agent.workflow.ReActAgent（异步 .run()），
    把知识库索引包成 QueryEngineTool 交给它。豆包端点非 function-calling，
    故用文本式 ReAct（而非 FunctionAgent）。
    """
    from llama_index.core import Settings
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.tools import QueryEngineTool

    index = _load_or_build_index()
    query_engine = index.as_query_engine(similarity_top_k=4)
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="knowledge_base",
        description=(
            "检索用户的个人知识库（第二大脑），回答关于其中笔记、资料、知识点的问题。"
            "输入应是一个具体、自包含的问题。"
        ),
    )
    return ReActAgent(tools=[tool], llm=Settings.llm)
