"""命令行问答（先用它跑通，再上 FastAPI）。

用法：python -m app.cli "你的问题"
"""

import asyncio
import sys

from .agent import build_agent


def main():
    if len(sys.argv) < 2:
        print('用法: python -m app.cli "你的问题"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    agent = build_agent()

    async def _ask():
        return await agent.run(question)

    resp = asyncio.run(_ask())

    print("\n=== 回答 ===")
    print(str(resp))

    # 出处：从 agent 本轮调用工具的输出里取检索到的源节点
    try:
        sources = []
        for tc in getattr(resp, "tool_calls", []) or []:
            raw = getattr(getattr(tc, "tool_output", None), "raw_output", None)
            for sn in getattr(raw, "source_nodes", []) or []:
                sources.append((sn.node.metadata.get("file_name", "?"), sn.score or 0.0))
        if sources:
            print("\n=== 出处 ===")
            for fn, score in sources[:4]:
                print(f"- {fn} (score={score:.3f})")
    except Exception:
        pass


if __name__ == "__main__":
    main()
