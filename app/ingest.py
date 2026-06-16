"""把语料目录里的 .md 读入、切块、向量化，建索引并落盘。

用法：python -m app.ingest    （首次/语料更新后跑一次）
"""

from . import config


def build_index():
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    config.configure_llamaindex()
    docs = SimpleDirectoryReader(
        config.CORPUS_DIR, recursive=True, required_exts=[".md"]
    ).load_data()
    print(f"[ingest] 从 {config.CORPUS_DIR} 读入 {len(docs)} 个文档")
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=config.STORAGE_DIR)
    print(f"[ingest] 索引已落盘: {config.STORAGE_DIR}")
    return index


if __name__ == "__main__":
    build_index()
