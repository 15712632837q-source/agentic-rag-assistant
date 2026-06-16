"""离线下载 embedding 模型到 models/（绕开国内 huggingface_hub 下载不稳的问题）。

实测：直接 httpx GET hf-mirror.com 的文件可达（200），但 huggingface_hub 的 HEAD/
缓存逻辑在国内会 LocalEntryNotFoundError。所以这里逐文件直连镜像下载，存到本地目录，
之后 sentence-transformers 直接从本地路径加载，全程不再依赖 hub 的下载。

用法：python scripts/download_embed_model.py   （可用 EMBED_REPO 环境变量换模型）
"""

import os
from pathlib import Path

import httpx

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
REPO = os.environ.get("EMBED_REPO", "BAAI/bge-small-zh-v1.5")
OUT = Path(__file__).resolve().parents[1] / "models" / REPO.split("/")[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = httpx.get(f"{MIRROR}/api/models/{REPO}", timeout=30,
                     follow_redirects=True).json()
    files = [s["rfilename"] for s in info.get("siblings", [])
             if s["rfilename"] != ".gitattributes"
             and not s["rfilename"].endswith(".md")]
    print(f"{REPO}: {len(files)} 个文件 -> {OUT}")
    with httpx.Client(timeout=180, follow_redirects=True) as c:
        for f in files:
            dest = OUT / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  跳过(已存在) {f}")
                continue
            url = f"{MIRROR}/{REPO}/resolve/main/{f}"
            with c.stream("GET", url) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_bytes(chunk_size=1 << 16):
                        fh.write(chunk)
            print(f"  下载 {f} ({dest.stat().st_size} bytes)")
    print("完成:", OUT)


if __name__ == "__main__":
    main()
