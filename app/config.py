"""配置：豆包/火山方舟做生成(OpenAI 兼容) + 本地 bge 中文 embedding。

设计取舍见 DECISIONS.md：
- LLM 用豆包(方舟)，OpenAI 兼容，引擎可换；embedding 用本地 fastembed(ONNX)，
  离线、免再开方舟 embedding 端点、对中文够用。
- 必须绕系统代理：本机系统代理会让直连方舟失败（项目既有实证）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# 绕系统代理：清掉代理环境变量，httpx 直连方舟
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)

# 注意：不在这里强制 HF 镜像。
# - 本地（国内）：用 scripts/download_embed_model.py 预下载到 models/，运行时离线加载，
#   不联网，所以 config 不需要设镜像；若确需镜像，本地运行前自行 export HF_ENDPOINT。
# - 云端（海外，如 Streamlit Cloud）：models/ 不存在，按 HF 仓库名直连官方 huggingface.co
#   下载——这时若强制走国内镜像反而会连不上（实测 Streamlit Cloud OSError 根因）。

ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# embedding 优先用本地已下载的模型目录（国内 huggingface_hub 下载不稳，
# 用 scripts/download_embed_model.py 直连镜像逐文件下到 models/ 下）；
# 本地不存在才退回 HF 仓库名（需联网）。
_LOCAL_EMBED = ROOT / "models" / "bge-small-zh-v1.5"
EMBED_MODEL = os.getenv("EMBED_MODEL") or (
    str(_LOCAL_EMBED) if _LOCAL_EMBED.exists() else "BAAI/bge-small-zh-v1.5")

CORPUS_DIR = os.getenv("CORPUS_DIR", r"D:\项目\第二大脑")
STORAGE_DIR = os.getenv("STORAGE_DIR", str(ROOT / "storage"))

_configured = False


def configure_llamaindex():
    """把全局 LlamaIndex Settings 配成 豆包LLM + 本地embedding（幂等）。"""
    global _configured
    if _configured:
        return
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    Settings.llm = OpenAILike(
        model=LLM_MODEL,
        api_base=ARK_BASE_URL,
        api_key=ARK_API_KEY,
        is_chat_model=True,
        is_function_calling_model=False,  # 走文本式 ReAct，不依赖原生 function calling
        timeout=90,                        # 豆包大上下文调用慢，给足超时
        context_window=32000,
        max_retries=2,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    _configured = True
