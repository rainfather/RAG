# 必须在 langchain/huggingface 导入前设置镜像，否则模型下载直连 huggingface.co 超时
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import *
import json
import hashlib
import jieba
from rank_bm25 import BM25Okapi
from datetime import datetime

def load_documents():
    """加载 data 目录下所有 PDF 和 TXT 文件"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"已创建文档目录 {DATA_DIR}，请放入文档后重新运行")
        return []

    # 加载PDF
    pdf_loader = DirectoryLoader(
        DATA_DIR, glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    # 加载TXT
    txt_loader = DirectoryLoader(
        DATA_DIR, glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = pdf_loader.load() + txt_loader.load()
    print(f"成功加载 {len(documents)} 页文档")
    return documents

def split_documents(documents):
    """按语义边界递归切分文本块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"切分为 {len(chunks)} 个文本块")
    return chunks

def build_vector_db(chunks):
    """生成向量并保存FAISS库到本地"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)
    print(f"向量库已保存至 {VECTOR_DB_PATH}")

    # ========== 构建BM25关键词索引 ==========
    all_texts = [doc.page_content for doc in chunks]
    all_metadatas = [doc.metadata for doc in chunks]
    # 中文分词
    tokenized_corpus = [list(jieba.cut(text)) for text in all_texts]
    # 构建BM25索引
    bm25 = BM25Okapi(tokenized_corpus)

    # 保存索引数据
    bm25_data = {
        "texts": all_texts,
        "metadatas": all_metadatas,
        "tokenized": [list(tokens) for tokens in tokenized_corpus]
    }
    with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)
    print(f"BM25索引已保存至 {BM25_INDEX_PATH}")

    return db

# ========== 文档注册表（增量更新） ==========

def compute_file_hash(filepath: str) -> str:
    """计算文件 SHA256 指纹"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_registry() -> dict:
    """加载文档注册表"""
    if not os.path.exists(DOC_REGISTRY_PATH):
        return {"documents": {}, "last_build": None}
    with open(DOC_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict):
    """保存文档注册表"""
    with open(DOC_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def detect_changes() -> tuple[list[str], list[str], list[str]]:
    """
    检测 data 目录下的文档变更。
    返回: (新增列表, 修改列表, 删除列表)
    """
    registry = load_registry()
    current_files: dict[str, str] = {}

    if os.path.exists(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.pdf', '.txt')):
                current_files[fname] = compute_file_hash(fpath)

    added = [f for f in current_files if f not in registry["documents"]]
    modified = [f for f in current_files
                if f in registry["documents"]
                and registry["documents"][f].get("hash") != current_files[f]]
    deleted = [f for f in registry["documents"] if f not in current_files]

    return added, modified, deleted


def incremental_update() -> dict | None:
    """
    增量更新知识库：
    - 纯新增：增量添加向量 → 快速
    - 有修改/删除：全量重建 → 安全
    返回更新摘要，无需更新返回 None
    """
    added, modified, deleted = detect_changes()

    if not (added or modified or deleted):
        print("📭 没有检测到文档变更，跳过更新")
        return None

    print(f"📋 变更检测：新增 {len(added)}, 修改 {len(modified)}, 删除 {len(deleted)}")

    summary = {
        "added": added, "modified": modified, "deleted": deleted,
        "type": "incremental",
        "timestamp": datetime.now().isoformat()
    }

    # 有修改或删除 → 全量重建（FAISS 不支持安全删除向量）
    if modified or deleted:
        print("⚠️ 有文档被修改/删除，执行全量重建...")
        docs = load_documents()
        if docs:
            chunks = split_documents(docs)
            build_vector_db(chunks)
            # 更新注册表
            _rebuild_registry()
        else:
            print("⚠️ 没有可用的文档，跳过重建")
        summary["type"] = "full_rebuild"
        return summary

    # 纯新增 → 增量追加
    if added:
        print(f"🔄 增量添加 {len(added)} 个新文档...")
        # 只加载新文件
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )

        new_docs = []
        for fname in added:
            fpath = os.path.join(DATA_DIR, fname)
            try:
                if fname.endswith(".pdf"):
                    loader = PyPDFLoader(fpath)
                else:
                    loader = TextLoader(fpath, encoding="utf-8")
                new_docs.extend(loader.load())
                print(f"  ✅ 加载: {fname}")
            except Exception as e:
                print(f"  ❌ 加载失败 {fname}: {e}")

        if new_docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""],
                length_function=len
            )
            new_chunks = splitter.split_documents(new_docs)

            # 加载现有向量库，追加新向量
            db = FAISS.load_local(
                VECTOR_DB_PATH, embeddings,
                allow_dangerous_deserialization=True
            )
            db.add_documents(new_chunks)
            db.save_local(VECTOR_DB_PATH)

            # 重建 BM25 索引（需要全量语料）
            all_docs = load_documents()
            all_chunks = splitter.split_documents(all_docs)
            _save_bm25_index(all_chunks)

            # 更新注册表
            registry = load_registry()
            for fname in added:
                fpath = os.path.join(DATA_DIR, fname)
                registry["documents"][fname] = {
                    "hash": compute_file_hash(fpath),
                    "added_at": datetime.now().isoformat()
                }
            registry["last_build"] = datetime.now().isoformat()
            save_registry(registry)

            print(f"✅ 增量更新完成：新增 {len(new_chunks)} 个文本块，总计 {len(all_chunks)} 个块")
        summary["new_chunks"] = len(new_chunks) if new_docs else 0
        return summary

    return summary


def _save_bm25_index(chunks):
    """保存 BM25 索引（从 build_vector_db 中提取，供增量更新复用）"""
    all_texts = [doc.page_content for doc in chunks]
    all_metadatas = [doc.metadata for doc in chunks]
    tokenized_corpus = [list(jieba.cut(text)) for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_data = {
        "texts": all_texts,
        "metadatas": all_metadatas,
        "tokenized": [list(tokens) for tokens in tokenized_corpus]
    }
    with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)
    print(f"BM25索引已更新 → {BM25_INDEX_PATH}")


def _rebuild_registry():
    """重建注册表（全量构建后调用）"""
    registry = {"documents": {}, "last_build": datetime.now().isoformat()}
    if os.path.exists(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.pdf', '.txt')):
                registry["documents"][fname] = {
                    "hash": compute_file_hash(fpath),
                    "added_at": datetime.now().isoformat()
                }
    save_registry(registry)
    print(f"文档注册表已更新 → {DOC_REGISTRY_PATH}")


if __name__ == "__main__":
    print("=== 开始构建知识库 ===")
    docs = load_documents()
    if not docs:
        exit()
    chunks = split_documents(docs)
    build_vector_db(chunks)
    _rebuild_registry()
    print("=== 构建完成 ===")
