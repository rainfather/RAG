"""
RAG问答系统 - FastAPI后端服务
启动命令：python main.py
接口文档：http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from graph.rag_chat import rag_chat
from data_process import load_documents, split_documents, build_vector_db, detect_changes, incremental_update
from cache import get_cached_answer, set_cached_answer, clear_all_cache, get_cache_stats, ping_redis
from auth import verify_api_key, check_rate_limit
from contextlib import asynccontextmanager
import os
import shutil
import json
import logging
from config import DATA_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

# ========== 初始化FastAPI应用 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时检查 Redis，关闭时清理资源"""
    logging.info("正在检查Redis连接...")
    if ping_redis():
        logging.info("✅ Redis连接正常，缓存功能已就绪")
    else:
        logging.warning("⚠️ Redis连接失败！缓存功能将不可用，服务以降级模式运行")
    yield  # 应用运行中...

app = FastAPI(
    title="RAG问答系统API",
    description="基于LangChain + FAISS + BM25 + Reranker的混合检索RAG系统",
    version="1.0.0",
    lifespan=lifespan
)

# ========== 跨域配置（前端页面调用需要） ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源，生产环境改成具体域名
    allow_credentials=True,
    allow_methods=["*"],       # 允许所有HTTP方法
    allow_headers=["*"],       # 允许所有请求头
)

# ========== 请求参数模型 ==========
class ChatRequest(BaseModel):
    """问答接口请求参数"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    history: list[dict] = Field(default_factory=list, description="对话历史 [{\"role\":\"user\",\"content\":\"...\"}, ...]")
    stream: bool = Field(default=False, description="是否使用 SSE 流式返回")

# ========== 1. 问答接口 ==========
@app.post("/api/chat", summary="问答接口")
async def chat(
    request: ChatRequest,
    _auth: bool = Security(verify_api_key),
    _rate: bool = Security(check_rate_limit),
):
    """
    传入问题，返回AI回答和参考资料来源。
    优先走Redis缓存，支持多轮对话和SSE流式输出。

    - **query**: 用户问题（1~2000字）
    - **history**: 可选，多轮对话历史
    - **stream**: 是否 SSE 流式返回（默认 false）
    """
    # ========== SSE 流式模式 ==========
    if request.stream:
        async def sse_generator():
            # 缓存检查（无历史时）
            if not request.history:
                cached = get_cached_answer(request.query)
                if cached:
                    yield f"data: {json.dumps({'type': 'token', 'content': cached['answer']}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'sources': cached.get('sources', []), 'from_cache': True}, ensure_ascii=False)}\n\n"
                    return

            try:
                answer_stream, sources = rag_chat(request.query, history=request.history)
                full_answer = ""

                if isinstance(answer_stream, str):
                    # 无上下文或缓存命中
                    yield f"data: {json.dumps({'type': 'token', 'content': answer_stream}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"
                else:
                    # 真实流式生成
                    for chunk in answer_stream:
                        full_answer += chunk
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
                    # 最后发送来源和完成信号
                    yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'from_cache': False}, ensure_ascii=False)}\n\n"
                    # 存入缓存
                    if sources and not request.history:
                        set_cached_answer(request.query, {"answer": full_answer, "sources": sources})
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",       # 禁用 Nginx 缓冲
                "Connection": "keep-alive",
            }
        )

    # ========== JSON 模式（默认） ==========
    # 第一步：缓存检查（无历史时）
    if not request.history:
        cached_result = get_cached_answer(request.query)
        if cached_result:
            return {
                "code": 0,
                "message": "success",
                "data": cached_result,
                "from_cache": True
            }

    # 第二步：RAG 流程
    try:
        answer_stream, sources = rag_chat(request.query, history=request.history)

        if isinstance(answer_stream, str):
            answer_text = answer_stream
        else:
            answer_text = ""
            for chunk in answer_stream:
                answer_text += chunk

        result_data = {
            "answer": answer_text,
            "sources": sources
        }

        # 第三步：存入缓存（只有无历史时缓存）
        if not isinstance(answer_stream, str) and sources and not request.history:
            set_cached_answer(request.query, result_data)

        return {
            "code": 0,
            "message": "success",
            "data": result_data,
            "from_cache": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答出错：{str(e)}")

# ========== 2. 文档上传接口 ==========
@app.post("/api/upload", summary="上传文档到知识库")
async def upload_document(file: UploadFile = File(...)):
    """
    上传PDF或TXT文档到知识库
    
    - **file**: 要上传的文档文件（仅支持.pdf和.txt）
    """
    # 校验文件类型
    filename = file.filename.lower()
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="仅支持PDF和TXT格式的文档")
    
    # 确保data目录存在
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # 保存文件
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败：{str(e)}")
    
    return {
        "code": 0,
        "message": "文件上传成功",
        "data": {
            "filename": file.filename,
            "size": os.path.getsize(file_path),
            "tip": "上传后需要调用重建知识库接口才能生效"
        }
    }

# ========== 3. 重建知识库接口 ==========
@app.post("/api/rebuild", summary="重建整个知识库（全量/增量）")
async def rebuild_knowledge_base(incremental: bool = False):
    """
    重建向量库和 BM25 索引。支持全量和增量两种模式。

    - **incremental=true**：只处理新增/变更的文档（推荐日常使用）
    - **incremental=false**：完全重建（通过上传文档后首次使用）
    """
    try:
        if incremental:
            # 增量更新模式
            result = incremental_update()
            if result is None:
                return {
                    "code": 0,
                    "message": "没有检测到文档变更",
                    "data": {"changed": False}
                }
            # 增量变更后清空缓存
            cleared_count = clear_all_cache()
            return {
                "code": 0,
                "message": f"知识库更新完成 (类型: {result['type']})",
                "data": {
                    "type": result["type"],
                    "added": result.get("added", []),
                    "modified": result.get("modified", []),
                    "deleted": result.get("deleted", []),
                    "cache_cleared": cleared_count
                }
            }
        else:
            # 全量重建模式
            docs = load_documents()
            if not docs:
                return {
                    "code": 1,
                    "message": "data目录没有找到任何文档",
                    "data": None
                }

            chunks = split_documents(docs)
            build_vector_db(chunks)

            # 重建注册表
            from data_process import _rebuild_registry
            _rebuild_registry()

            # 重建后清空缓存
            cleared_count = clear_all_cache()

            return {
                "code": 0,
                "message": "知识库重建成功，缓存已清空",
                "data": {
                    "document_count": len(docs),
                    "chunk_count": len(chunks),
                    "cache_cleared": cleared_count
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"建库失败：{str(e)}")

# ========== 3.1 检测文档变更 ==========
@app.get("/api/documents/check-updates", summary="检测文档变更")
async def check_document_updates():
    """
    检测 data 目录下的文档变更（新增/修改/删除），不实际更新。
    用于在更新前预览会有哪些变化。
    """
    try:
        added, modified, deleted = detect_changes()
        has_changes = bool(added or modified or deleted)
        return {
            "code": 0,
            "message": "检测完成",
            "data": {
                "has_changes": has_changes,
                "added": added,
                "modified": modified,
                "deleted": deleted,
                "recommendation": "建议执行增量更新" if has_changes and not deleted
                else "需要全量重建" if deleted else "知识库已是最新"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"变更检测失败：{str(e)}")

# ========== 3.2 查看文档注册表 ==========
@app.get("/api/documents/registry", summary="查看文档注册表")
async def view_registry():
    """查看知识库的文档注册表，了解当前索引进度"""
    from data_process import load_registry
    try:
        registry = load_registry()
        return {
            "code": 0,
            "message": "success",
            "data": registry
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取注册表失败：{str(e)}")

# ========== 4. 获取文档列表接口 ==========
@app.get("/api/documents", summary="获取知识库文档列表")
async def list_documents():
    """查看当前知识库中有哪些文档"""
    if not os.path.exists(DATA_DIR):
        return {"code": 0, "message": "success", "data": {"documents": []}}
    
    files = []
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            files.append({
                "name": filename,
                "size": os.path.getsize(file_path)
            })
    
    return {
        "code": 0,
        "message": "success",
        "data": {"documents": files}
    }

# ========== 5. 健康检查接口 ==========
@app.get("/api/health", summary="健康检查")
async def health_check():
    """检查服务是否正常运行，同时验证Redis连接状态"""
    redis_ok = ping_redis()
    return {
        "code": 0,
        "message": "服务运行正常" if redis_ok else "Redis连接异常，缓存功能不可用",
        "data": {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "version": "1.0.0"
        }
    }

# ========== 6. 清空缓存接口 ==========
@app.post("/api/cache/clear", summary="清空所有问答缓存")
async def clear_cache():
    """手动清空所有Redis缓存，知识库有变动时可以调用"""
    count = clear_all_cache()
    return {
        "code": 0,
        "message": f"已清空{count}条缓存",
        "data": {"cleared_count": count}
    }

# ========== 7. 缓存状态接口 ==========
@app.get("/api/cache/stats", summary="查看缓存统计")
async def cache_stats():
    """查看当前有多少条缓存"""
    stats = get_cache_stats()
    return {
        "code": 0,
        "message": "success",
        "data": stats
    }

# ========== 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("RAG问答系统后端启动中...")
    print("接口文档地址：http://localhost:8000/docs")
    print("服务端口：8000")
    print("=" * 50)
    uvicorn.run(
        app,
        host="0.0.0.0",  # 允许外部访问
        port=8000
    )
