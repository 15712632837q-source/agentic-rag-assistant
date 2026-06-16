"""长期记忆模块：SQLite 存事实 + 复用 bge embedding 做「语义相关性 × 时间衰减」检索。

落地《从零搭建 AI-Agent 的 10 个模块》里的 Memory 模块：
- 存：用户要记的事实/偏好写进 SQLite（连同它的向量）。
- 取：把查询向量化，对每条记忆算余弦相似度，再乘一个【指数时间衰减】（旧记忆权重低），
  排序取 top-k。
agent 通过 remember / recall 两个工具自主存取（见 app/agent.py）。
embedding 复用项目已配置的本地 bge 模型，不额外引入依赖。
"""

import json
import math
import sqlite3
import time
from pathlib import Path

from . import config

_DB = Path(config.ROOT) / "memory.db"   # 已 gitignore
HALF_LIFE_DAYS = 14.0                    # 记忆半衰期：14 天后语义分权重减半


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories "
        "(id INTEGER PRIMARY KEY, text TEXT, kind TEXT, ts REAL, vec TEXT)"
    )
    return conn


def _embed(text: str) -> list[float]:
    config.configure_llamaindex()
    from llama_index.core import Settings
    return Settings.embed_model.get_text_embedding(text)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _time_decay(age_days: float) -> float:
    """指数时间衰减：age=0→1.0，age=半衰期→0.5。旧记忆语义分被压低。"""
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def add(text: str, kind: str = "fact") -> None:
    vec = _embed(text)
    conn = _conn()
    conn.execute("INSERT INTO memories(text, kind, ts, vec) VALUES(?,?,?,?)",
                 (text, kind, time.time(), json.dumps(vec)))
    conn.commit()
    conn.close()


def search(query: str, top_k: int = 5, min_sim: float = 0.3) -> list[dict]:
    """返回 [{text, kind, sim, age_days, score}]，按 score=sim×时间衰减 降序。"""
    conn = _conn()
    rows = conn.execute("SELECT text, kind, ts, vec FROM memories").fetchall()
    conn.close()
    if not rows:
        return []
    qv = _embed(query)
    now = time.time()
    out = []
    for text, kind, ts, vecjson in rows:
        sim = _cos(qv, json.loads(vecjson))
        if sim < min_sim:
            continue
        age_days = (now - ts) / 86400.0
        out.append({"text": text, "kind": kind, "sim": round(sim, 3),
                    "age_days": round(age_days, 1),
                    "score": sim * _time_decay(age_days)})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:top_k]


# ── 给 agent 用的工具函数（返回给模型看的文本）──
def remember(text: str) -> str:
    """把一句值得长期记住的事实/偏好存入长期记忆。"""
    add(text)
    return f"已记住：{text}"


def recall(query: str) -> str:
    """从长期记忆检索与查询相关的内容（语义×时间衰减）。"""
    results = search(query)
    if not results:
        return "（长期记忆里没有相关内容）"
    lines = [f"- {r['text']}（相关度 {r['sim']}，{r['age_days']} 天前）" for r in results]
    return "相关的长期记忆：\n" + "\n".join(lines)
