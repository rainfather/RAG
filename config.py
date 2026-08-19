import os
from dotenv import load_dotenv

load_dotenv()

# ========== 大模型配置 ==========
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL_NAME = "deepseek-v4-flash"
LLM_TEMPERATURE = 0.4  # 越低越严谨，减少幻觉

# ========== 向量模型配置 ==========
# 支持 env 覆盖（如指向本地模型目录 EMBEDDING_MODEL_NAME=D:/RAG/models/bge-base-zh-v1.5）
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")  # BGE中文嵌入模型(C-MTEB前列)，large版本效果更好但需1.3G显存
EMBEDDING_DEVICE = "cpu"  # 有GPU改为 "cuda"

# ========== 重排序模型配置 ==========
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_DEVICE = "cpu"

# ========== 文档分块配置 ==========
CHUNK_SIZE = 500    # 单块字符数
CHUNK_OVERLAP = 100  # 块间重叠字符数

# ========== 检索配置 ==========
RECALL_TOP_K = 10   # 向量粗召回数量
BM25_TOP_K = 10     # BM25关键词召回数量
RERANK_TOP_N = 3    # 最终送入LLM的数量

# ========== 路径配置 ==========
DATA_DIR = "./data"
VECTOR_DB_PATH = "./faiss_db"
BM25_INDEX_PATH = "./bm25_index.json"  # BM25索引文件路径
DOC_REGISTRY_PATH = "./document_registry.json"  # 文档注册表（增量更新用）

# ========== Redis缓存配置 ==========
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)  # 生产环境请设置密码
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
CACHE_EXPIRE_SECONDS = int(os.getenv("CACHE_EXPIRE_SECONDS", "3600"))  # 默认1小时

# ========== 安全配置 ==========
# 逗号分隔的 API Key 白名单，为空则不校验（例如：key1,key2,key3）
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "60"))     # 每窗口最大请求数
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # 限流窗口(秒)

# ========== 多轮对话配置 ==========
CONVERSATION_MAX_TURNS = int(os.getenv("CONVERSATION_MAX_TURNS", "5"))  # 保留最近N轮历史

# ========== LangFuse 可观测性配置 ==========
# https://cloud.langfuse.com 注册获取（留空则不启用追踪）
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
