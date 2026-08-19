"""
检索组件（从旧 rag_core.py 拆出，逻辑零改动）
- get_embeddings / get_vector_db / get_bm25 / bm25_retrieve / get_reranker
- 全局单例，避免重复加载模型
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import json

import jieba
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    RERANKER_MODEL_NAME,
    RERANKER_DEVICE,
    VECTOR_DB_PATH,
    BM25_INDEX_PATH,
    BM25_TOP_K,
)

# 全局单例模型，避免重复加载
_embeddings = None
_vector_db = None
_reranker = None
_bm25 = None
_bm25_texts = None
_bm25_metadatas = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # 注：langchain_huggingface 1.x 已移除 HuggingFaceBgeEmbeddings 特化类，
        # 用 HuggingFaceEmbeddings + normalize 参数等价替代（BGE 模型推荐用法，
        # 与旧 data_process.py 一致）
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = FAISS.load_local(
            VECTOR_DB_PATH, get_embeddings(),
            allow_dangerous_deserialization=True
        )
    return _vector_db


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, device=RERANKER_DEVICE)
    return _reranker


def get_bm25():
    """加载BM25索引，全局单例"""
    global _bm25, _bm25_texts, _bm25_metadatas
    if _bm25 is None:
        with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _bm25_texts = data["texts"]
        _bm25_metadatas = data["metadatas"]
        _bm25 = BM25Okapi(data["tokenized"])
    return _bm25, _bm25_texts, _bm25_metadatas


def bm25_retrieve(query, top_k=BM25_TOP_K):
    """BM25关键词检索，返回Document对象列表"""
    bm25, texts, metadatas = get_bm25()
    token_query = list(jieba.cut(query))
    scores = bm25.get_scores(token_query)
    # 按分数排序取top_k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    # 构造Document对象
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # 过滤掉0分的
            results.append(Document(page_content=texts[idx], metadata=metadatas[idx]))
    return results
