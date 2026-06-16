"""Streamlit 网页界面 —— Agentic RAG Assistant 的"能点开"版（部署到 Streamlit Cloud）。

云端入口：Streamlit Cloud 的 Main file 填 `streamlit_app.py`。
- 密钥从 st.secrets 注入环境变量（云端在 app 设置里填 ARK_API_KEY / LLM_MODEL）；
- 语料用仓库内的公开示例 `sample_docs/`（不用私人第二大脑，防隐私泄露）；
- 索引在首次访问时按需构建（storage/ 不入库），用 cache_resource 只建一次。
"""

import asyncio
import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

# 把 Streamlit secrets 注入环境变量（供 app.config 在 import 时读取）——必须在导入 app.* 之前
try:
    for _k in ("ARK_API_KEY", "LLM_MODEL", "ARK_BASE_URL", "EMBED_MODEL"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # 本地无 secrets 文件时走 .env
# 线上 demo 固定用公开示例语料
os.environ.setdefault("CORPUS_DIR", str(ROOT / "sample_docs"))

from app.agent import build_agent  # noqa: E402


@st.cache_resource(show_spinner="首次加载：下载 embedding 模型 + 建索引（约 1-2 分钟）…")
def get_agent():
    return build_agent()


def ask(agent, question: str):
    async def _run():
        return await agent.run(question)
    return asyncio.run(_run())


st.set_page_config(page_title="Agentic RAG Assistant", page_icon="📚")
st.title("📚 Agentic RAG Assistant")
st.caption(
    "对一个 markdown 知识库做「自主检索」的问答：agent 自己决定何时检索、查什么，"
    "并给出处引用。本 demo 语料为公开示例文档（换成任意 markdown 目录即可复用）。"
)

with st.sidebar:
    st.subheader("这是什么")
    st.markdown(
        "- **LlamaIndex** ReActAgent（agent 自主决定是否检索）\n"
        "- **豆包/火山方舟** 生成（OpenAI 兼容，引擎可换）\n"
        "- **本地 bge-zh** embedding\n"
        "- 检索命中 hit@4 **100%**、检索延迟 P50 **9ms**（见 README）"
    )
    st.markdown("[GitHub 源码](https://github.com/15712632837q-source/agentic-rag-assistant)")

question = st.text_input(
    "问点什么？",
    placeholder="例如：什么是 Agentic RAG？它和朴素 RAG 有什么区别？",
)

if question:
    agent = get_agent()
    with st.spinner("检索 + 思考中…"):
        resp = ask(agent, question)
    st.markdown("### 回答")
    st.write(str(resp))

    sources = []
    for tc in getattr(resp, "tool_calls", []) or []:
        raw = getattr(getattr(tc, "tool_output", None), "raw_output", None)
        for sn in getattr(raw, "source_nodes", []) or []:
            sources.append((sn.node.metadata.get("file_name", "?"), sn.score or 0.0))
    if sources:
        st.markdown("### 出处")
        for fn, score in sources[:4]:
            st.write(f"- `{fn}`　(相似度 {score:.3f})")
