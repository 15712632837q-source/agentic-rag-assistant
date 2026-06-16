"""轻量检索评测：对一组 (问题, 期望命中的源文件) 测 hit@k + 检索延迟。

证明检索是真的有效（招聘方要"带数字的 eval"），不是凭感觉。
用法：python scripts/eval_retrieval.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import _load_or_build_index  # noqa: E402

# (问题, 期望命中的源文件名子串)——基于已知语料人工标注的小评测集
CASES = [
    ("ReAct 模式的 Thought Action Observation 循环是怎样的？", "ReAct"),
    ("Plan-and-Execute 由哪几个模块组成？", "Plan-and-Execute"),
    ("Agent 是什么？大模型为什么需要工具？", "Agent 是什么"),
    ("ADC 抗体偶联药物的三要素是什么？", "ADC"),
    ("从零打造简化版 Claude Code 的视频讲了什么？", "Agent的概念原理"),
]


def main():
    index = _load_or_build_index()
    retriever = index.as_retriever(similarity_top_k=4)
    hits, latencies = 0, []
    print(f"评测集：{len(CASES)} 题，top_k=4\n")
    for q, expected in CASES:
        t = time.perf_counter()
        nodes = retriever.retrieve(q)
        dt = (time.perf_counter() - t) * 1000
        latencies.append(dt)
        files = [n.node.metadata.get("file_name", "?") for n in nodes]
        hit = any(expected in f for f in files)
        hits += hit
        print(f"{'✓' if hit else '✗'} [{dt:4.0f}ms] {q[:24]}…  期望含「{expected}」")
        print(f"     命中: {files}")
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    print(f"\n命中率 hit@4: {hits}/{len(CASES)} = {hits / len(CASES) * 100:.0f}%")
    print(f"检索延迟 P50: {p50:.0f}ms  P95: {p95:.0f}ms")


if __name__ == "__main__":
    main()
