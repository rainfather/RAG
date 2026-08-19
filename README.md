# 📚 RAG 知识问答系统（LangGraph 版）

基于 **LangGraph 编排 + 混合检索 + 重排序 + 查询改写** 的 RAG 知识问答系统。
本仓库由原 LangChain LCEL chain 版迁移而来——**功能与接口不变**（API 端点、SSE 协议、缓存语义、鉴权限流、测试行为），编排层改为 LangGraph 图结构，为后续扩展（工具调用、转人工等智能客服能力）预留节点化架构。

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         用户交互层                                    │
│  ┌──────────────────────┐          ┌──────────────────────────────┐  │
│  │   Streamlit UI       │          │   FastAPI REST + Swagger     │  │
│  │   (app.py)           │          │   (main.py)                  │  │
│  └──────────┬───────────┘          └──────────────┬───────────────┘  │
└─────────────┼──────────────────────────────────────┼──────────────────┘
              │                                      ▼
              │                           ┌──────────────────────┐
              │                           │  rag_chat 门面        │
              │                           │  (graph/rag_chat.py)  │
              │                           │  缓存检查 → graph 执行 │
              │                           └──────────┬───────────┘
              ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  LangGraph 图 (graph/build.py)                        │
│                                                                      │
│  rewrite_query ──▶ retrieve_context ──▶ (有上下文?) ──▶ generate      │
│  (LLM 改写 1→4)    (FAISS+BM25 混合      │  否 → END(拒答)   (LLM 流式 │
│                    检索→去重→Reranker)   │                   生成,    │
│                                          │                   节点内   │
│                                          │                   yield)  │
└──────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          存储 & 检索层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  FAISS 向量库 │  │  BM25 索引   │  │  Redis 缓存 (带降级)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **LangGraph 编排** | 有状态图：查询改写→混合检索→重排序→生成，节点独立可测，条件边控制拒答/生成分支 |
| 🔀 **混合检索** | FAISS 稠密向量语义召回 + BM25 稀疏关键词召回，互补提升召回率 |
| ✍️ **查询改写** | 单问题 → 多角度改写（LLM 驱动），解决 query-document mismatch |
| 🎯 **重排序精排** | Cross-Encoder Reranker 对粗召回结果二次排序，提升 Top-3 精准度 |
| ⚡ **流式生成** | LangGraph custom stream：节点内逐 token 发送，SSE 首字延迟低 |
| 💾 **Redis 缓存** | 相同问题秒级返回，带连接池、健康检查、自动降级 |
| 💬 **多轮对话** | 支持追问、指代消解，基于对话历史的上下文理解 |
| 🔐 **鉴权 & 限流** | API Key 白名单 + 滑动窗口频率限制 |
| 📊 **RAGAS 评估** | Faithfulness / Answer Relevancy / Context Precision&Recall 自动评估 |
| 📄 **多格式支持** | PDF + TXT 文档自动解析、分块、向量化（增量更新） |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Redis（可选，用于缓存加速；不装也能正常运行）

### 1. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
copy .env .env
# 编辑 .env，填入你的 API Key（必填）
# DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. 构建知识库

```bash
# 将 PDF/TXT 文档放入 data/ 目录，然后运行建库脚本
python data_process.py
```

> 💡 **模型下载**：建库会自动从 HuggingFace 下载 BGE 中文嵌入模型（BAAI/bge-base-zh-v1.5）。
> 国内网络建议：设置 `HF_ENDPOINT=https://hf-mirror.com`（data_process.py 已内置），
> 或从 ModelScope 手动下载模型后指定本地路径：
> `set EMBEDDING_MODEL_NAME=D:\RAG\models\bge-base-zh-v1.5` 再运行建库。

### 4. 启动服务

**方式一：FastAPI 后端（推荐）**

```bash
python main.py
# API 文档 → http://localhost:8000/docs
# 健康检查 → http://localhost:8000/api/health
```

**方式二：Streamlit Web 界面**

```bash
streamlit run app.py
# 浏览器访问 http://localhost:8501
```

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | 问答（自动走缓存；`stream=true` 时 SSE 流式） |
| `POST` | `/api/upload` | 上传文档（PDF/TXT） |
| `POST` | `/api/rebuild` | 重建知识库（全量/增量） |
| `GET` | `/api/documents/check-updates` | 检测文档变更 |
| `GET` | `/api/documents/registry` | 查看文档注册表 |
| `GET` | `/api/documents` | 查看已上传文档列表 |
| `GET` | `/api/health` | 健康检查（含 Redis 状态） |
| `POST` | `/api/cache/clear` | 手动清空缓存 |
| `GET` | `/api/cache/stats` | 查看缓存统计 |

### SSE 流式协议

```
data: {"type":"token","content":"..."}   # 逐 token
data: {"type":"done","sources":[...]}    # 结束（携带参考来源）
data: {"type":"error","content":"..."}   # 出错
```

## 📁 项目结构

```
D:\RAG\
├── graph\                  # LangGraph 编排核心（本次迁移新增）
│   ├── state.py            # 图状态定义 (RagState)
│   ├── nodes.py            # 节点函数：rewrite/retrieve/generate + 纯函数工具
│   ├── build.py            # 图构建与编译
│   └── rag_chat.py         # 门面：对外接口（签名与旧版一致）
├── retrieval.py            # 检索组件（从旧 rag_core.py 拆出）
├── main.py                 # FastAPI 后端（8 端点，接口不变）
├── app.py                  # Streamlit 前端
├── data_process.py         # 文档加载、分块、建库（含增量更新）
├── cache.py                # Redis 缓存工具类
├── auth.py                 # API Key 鉴权 + 频率限制
├── tracing.py              # LangFuse 可观测性追踪
├── evaluate.py             # RAGAS 质量评估脚本
├── config.py               # 全局配置管理
├── data\                   # 原始文档目录
├── faiss_db\               # FAISS 向量库（构建产物）
└── tests\                  # 测试（46+ 用例）
    ├── test_graph.py       # 节点级 + 图级测试（FakeLLM，不依赖外部服务）
    ├── test_rag.py         # rag_chat 门面测试（缓存/拒答/流式语义）
    ├── test_e2e.py         # 端到端：真实检索 + FakeLLM（不 mock 中间层）
    ├── test_sse.py         # SSE 流式协议端到端测试
    ├── test_api.py         # FastAPI 端点测试
    ├── test_cache.py       # 缓存测试
    └── test_auth.py        # 鉴权限流测试
```

## 🧪 运行测试

```bash
# 单元测试（不需要知识库与 Redis）
pytest tests/ -m "not requires_kb and not requires_redis" -v

# 全部测试（含端到端，需要已构建知识库）
pytest tests/ -v
```

> 端到端测试（test_e2e / test_sse）使用 FakeLLM 替代真实 LLM——不 mock 检索中间层，
> 验证 query → 改写 → 混合检索 → 重排 → 生成 → sources 的完整链路。

## 🔄 与旧版（LangChain chain 版）的差异

| 维度 | 旧版 rag_core.py | 新版 graph/ |
|------|------------------|-------------|
| 编排 | LCEL chain（链式管道） | LangGraph StateGraph（显式状态流 + 条件边） |
| 流式 | `chain.stream()` | 节点内 `get_stream_writer()` + `stream_mode="custom"` |
| 缓存检查 | rag_chat 入口同步 | rag_chat 门面同步（行为一致） |
| 拒答分支 | 函数内 if 判断 | 条件边路由（retrieve → END / generate） |
| 可扩展性 | 加逻辑需改链 | 加节点/加边即可（工具调用、转人工后续接入点） |

## ⚠️ 已知事项

- `deepseek-v4-flash` 模型名沿用旧配置——若直连官方 `api.deepseek.com`，请核实模型名（官方为 `deepseek-chat` / `deepseek-reasoner`）
- 嵌入模型默认 `BAAI/bge-base-zh-v1.5`（config.py 支持 `EMBEDDING_MODEL_NAME` 环境变量覆盖）
- RAGAS 评估依赖单独安装：`pip install -r requirements-eval.txt`（Windows 下 ragas 传递依赖无预编译 wheel）

## 📄 License

MIT
