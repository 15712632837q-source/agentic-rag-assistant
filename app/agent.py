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
    from llama_index.core.tools import FunctionTool, QueryEngineTool

    from . import memory

    index = _load_or_build_index()
    query_engine = index.as_query_engine(similarity_top_k=4)
    kb_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="knowledge_base",
        description=(
            "检索知识库文档，回答关于其中笔记、资料、知识点的问题。"
            "输入应是一个具体、自包含的问题。"
        ),
    )
    remember_tool = FunctionTool.from_defaults(
        fn=memory.remember, name="remember",
        description="把用户明确要你记住的、值得长期保留的事实或偏好存入长期记忆。输入一句要记住的话。",
    )
    recall_tool = FunctionTool.from_defaults(
        fn=memory.recall, name="recall",
        description="回答前，从长期记忆里检索与当前问题/用户相关的、之前记住过的事实或偏好。输入一个查询短语。",
    )
    return ReActAgent(tools=[kb_tool, remember_tool, recall_tool], llm=Settings.llm)
