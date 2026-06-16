"""技能（Skills）模块：可按需加载的"技能手册"，让 agent 按规定流程/格式做事。

落地《从零搭建 AI-Agent 的 10 个模块》的 Skills：技能多就只把【名称+描述】喂给模型，
模型按需 load 完整流程，避免一股脑塞爆上下文。技能就是 skills/ 目录下的 markdown——
**新增一个技能 = 加一个 .md 文件，无需改代码**（用户可自建 / 装社区技能）。

文件格式（极简，无需 yaml 依赖）：
    ---
    name: 技能名
    description: 一句话说明在什么场景用
    ---
    <技能正文：要 agent 遵循的步骤 / 输出格式 / 模板>
"""

import os
from pathlib import Path

from . import config

SKILLS_DIR = Path(os.getenv("SKILLS_DIR", str(Path(config.ROOT) / "skills")))


def _parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    name, desc, body = path.stem, "", text
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            header, body = parts[1], parts[2]
            for line in header.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip().lower(), v.strip()
                    if k == "name":
                        name = v
                    elif k == "description":
                        desc = v
    return {"name": name, "description": desc, "body": body.strip(), "file": path.name}


def _all() -> list[dict]:
    if not SKILLS_DIR.is_dir():
        return []
    return [_parse(p) for p in sorted(SKILLS_DIR.glob("*.md"))]


def list_skills(query: str = "") -> str:
    """列出所有可用技能的【名称+描述】。遇到任务先调它，看有没有匹配的技能可用。"""
    skills = _all()
    if not skills:
        return "（当前没有可用技能）"
    lines = [f"- {s['name']}：{s['description']}" for s in skills]
    return "可用技能（匹配的话用 load_skill(名称) 加载完整流程再执行）：\n" + "\n".join(lines)


def load_skill(name: str) -> str:
    """按名称加载一个技能的完整流程/格式说明，加载后【严格按它执行】。"""
    key = (name or "").strip().lower()
    skills = _all()
    for s in skills:
        if key and (key == s["name"].lower() or key in s["name"].lower()
                    or s["file"].lower().startswith(key)):
            return f"【技能：{s['name']}】请严格按以下流程与格式完成本次任务：\n\n{s['body']}"
    avail = "、".join(s["name"] for s in skills) or "无"
    return f"没找到技能「{name}」。当前可用技能：{avail}"
