# Agentic RAG Assistant

对**任意 markdown 知识库**做"会自主检索"的问答助手：agent 自己判断**要不要查、查什么、够不够再答**，并给出**出处引用**——不是每问必检索一次的朴素 RAG。

> 🔗 **在线 Demo**：https://agentic-rag-assistant-uylcdzgzoowxuxmlvhu3kq.streamlit.app
> （Streamlit Community Cloud · 语料为 `sample_docs/` 公开示例 · 首次加载约 1-2 分钟）
> 🌐 **本地网页**：`streamlit run streamlit_app.py`

> Demo 语料用的是作者自建的个人知识库「第二大脑」(42 篇笔记)，但换成任意 markdown 目录即可复用（改 `CORPUS_DIR` 一个变量）。

## 一张图看懂

```
用户提问
   ↓
ReActAgent (思考→行动→观察循环)
   ├─ 决定是否调用 knowledge_base 工具
   │     ↓
   │  VectorStoreIndex 检索 (本地 bge-zh embedding, top-k)
   │     ↓ 命中片段 + 出处
   └─ 综合检索结果作答（带引用）→ 用户
LLM = 豆包 / 火山方舟 (OpenAI 兼容，引擎可换)
```

## 为什么不是"朴素 RAG"
普通 RAG 把"检索→塞进 prompt→生成"写死。这里用 **ReAct agent** 让模型**自主决策**：
简单问题可不检索直接答、复杂问题可多轮检索/换问法，更接近真实助手的行为，也更能体现 agent 编排能力。

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env                                   # 填入 ARK_API_KEY / LLM_MODEL

python -m app.ingest                                     # 建索引（首次会下载本地 embedding 模型）
python -m app.cli "我的知识库里关于 ReAct 模式讲了什么？"   # 命令行问答

uvicorn app.api:app --port 8000                          # 起 HTTP 服务
# POST /ask {"question": "..."}  ->  {"answer": "...", "sources": [...]}
```

Docker：

```bash
docker build -t agentic-rag . && docker run -p 8000:8000 --env-file .env agentic-rag
```

## 技术栈
- **检索/索引**：LlamaIndex（`VectorStoreIndex` + `ReActAgent`）
- **生成**：豆包 / 火山方舟（OpenAI 兼容，`OpenAILike`，可换任意兼容端点）
- **向量化**：本地 `bge-small-zh`（fastembed / ONNX，离线、对中文友好）
- **服务**：FastAPI + Uvicorn + Docker

## 效果指标（实测）
| 指标 | 数值 | 来源 |
|---|---|---|
| 语料规模 | 42 篇 markdown / 63 个向量块 | `app.ingest` |
| 检索命中 hit@4 | **100%（5/5）** | `scripts/eval_retrieval.py` |
| 检索延迟 | **P50 9ms / P95 27ms** | 本地 bge + 内存索引 |
| 答案出处 | 每次作答返回 top-k 源文件 + 相似度 | `/ask` 响应 `sources` |
| 端到端延迟 / 单次成本(token) | 下一步用通用 eval 框架量化 | _进行中_ |

> 评测集是基于已知语料的人工标注小集，跑 `python scripts/eval_retrieval.py` 复现。

## 设计取舍
见 [DECISIONS.md](DECISIONS.md)——为什么用 LlamaIndex 而非自研、为什么本地 embedding、为什么 ReAct agent。
