"""FastAPI 服务：把 Agentic RAG 暴露成 HTTP 接口（生产化第一步）。

启动：uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import build_agent

app = FastAPI(title="Agentic RAG Assistant", version="0.1")
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent(verbose=False)
    return _agent


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(body: AskRequest):
    resp = await get_agent().run(body.question)
    sources = []
    for tc in getattr(resp, "tool_calls", []) or []:
        raw = getattr(getattr(tc, "tool_output", None), "raw_output", None)
        for sn in getattr(raw, "source_nodes", []) or []:
            sources.append({"file": sn.node.metadata.get("file_name", "?"),
                            "score": round(sn.score or 0.0, 3)})
    return {"answer": str(resp), "sources": sources[:4]}
