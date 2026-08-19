"""
RAGAS 评估脚本 - 自动化评估 RAG 管线质量
用法: python evaluate.py

前置条件:
  1. pip install ragas datasets
  2. 已运行 data_process.py 构建知识库
  3. .env 中配置了 DEEPSEEK_API_KEY

评估指标:
  - faithfulness: 答案是否忠实于检索到的上下文
  - answer_relevancy: 答案是否与问题相关
  - context_precision: 检索结果中相关文档的排序质量
  - context_recall: 检索结果是否覆盖了所有相关信息
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import sys
import json
import time
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, VECTOR_DB_PATH
from graph.rag_chat import rag_chat

# ========== 评估数据集 ==========
# 基于 data/ 目录下的文档手工标注
EVAL_DATASET = [
    {
        "question": "什么是向量检索？",
        "ground_truth": "向量检索也叫稠密检索，是将文本转换为高维向量，通过计算余弦相似度来匹配相关内容，能够理解语义。"
    },
    {
        "question": "BM25算法有什么特点？",
        "ground_truth": "BM25基于词频和逆文档频率打分排序，特点是精确匹配专业术语、速度快，但无法理解同义词和改写。"
    },
    {
        "question": "什么是混合检索方案？",
        "ground_truth": "混合检索同时使用向量语义检索和BM25关键词检索，把两路结果合并去重后送入大模型，结合两者优点。"
    },
    {
        "question": "重排序Reranker的作用是什么？",
        "ground_truth": "重排序是在粗召回之后进行的精排步骤，使用交叉编码器对候选文档重新打分排序，精度比向量检索高但速度更慢。"
    },
    {
        "question": "中文Embedding模型有什么推荐？",
        "ground_truth": "中文Embedding模型首选BGE系列，由智源研究院开源；M3E也是常用的中文向量模型，由MokaAI开源。"
    },
    {
        "question": "如何缓解大模型幻觉问题？",
        "ground_truth": "RAG技术通过检索外部知识库让模型基于真实资料回答，是缓解幻觉最有效的方法之一；好的Prompt工程也能减少幻觉。"
    },
    {
        "question": "探索者X100无人机的续航时间是多少？",
        "ground_truth": "电池续航时间长达30分钟。"
    },
    {
        "question": "探索者X100无人机有哪些飞行模式？",
        "ground_truth": "拥有智能跟拍、轨迹飞行和一键返航等多种飞行模式。"
    },
    {
        "question": "FAISS向量数据库有哪些索引类型？",
        "ground_truth": "FAISS提供Flat暴力索引、HNSW层次化导航小世界索引、IVF倒排索引等类型。"
    },
    {
        "question": "文档分块时需要注意什么？",
        "ground_truth": "分块大小和重叠率是两个关键参数，一般中文场景块大小设为500到1000字符。分块太小会丢失上下文，太大又会引入噪音。"
    },
]


def check_prerequisites():
    """检查前置条件"""
    # 1. 检查知识库
    if not os.path.exists(VECTOR_DB_PATH):
        print("❌ 未找到向量知识库，请先运行: python data_process.py")
        sys.exit(1)

    # 2. 检查 API Key
    if not LLM_API_KEY:
        print("❌ 未配置 DEEPSEEK_API_KEY，请检查 .env 文件")
        sys.exit(1)

    # 3. 检查 ragas 是否安装
    try:
        import ragas
        import datasets
        print(f"✅ ragas {ragas.__version__} 已就绪")
    except ImportError:
        print("❌ 请先安装 ragas: pip install ragas datasets")
        sys.exit(1)

    print(f"✅ 知识库已就绪: {VECTOR_DB_PATH}")
    print(f"✅ 评估数据集: {len(EVAL_DATASET)} 条问答")
    print()


def run_pipeline():
    """运行 RAG 管线，生成所有答案"""
    results = []
    total = len(EVAL_DATASET)

    for i, item in enumerate(EVAL_DATASET, 1):
        question = item["question"]
        print(f"[{i}/{total}] 处理: {question}")

        # 走完整 RAG 图（门面），从 sources 取检索上下文
        answer_stream, sources = rag_chat(question, use_cache=False)
        answer = "".join(answer_stream)
        contexts = [s["content"] for s in sources]

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

        # 打印预览
        print(f"    回答: {answer[:100]}...")
        print(f"    召回文档数: {len(contexts)}")
        print()

    return results


def run_ragas_evaluation(results):
    """使用 RAGAS 评估"""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from datasets import Dataset

    print("=" * 60)
    print("开始 RAGAS 评估...")
    print("=" * 60)

    # 构建评估 LLM
    eval_llm = ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL_NAME,
        temperature=0.0,  # 评估用 0 温度，保证可复现
    )
    ragas_llm = LangchainLLMWrapper(eval_llm)

    # 构建数据集
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # 评估
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
    )

    return scores


def print_report(results, scores):
    """输出评估报告"""
    print()
    print("=" * 60)
    print("📊 RAGAS 评估报告")
    print("=" * 60)

    # 汇总指标
    print("\n## 综合指标")
    print("-" * 40)
    score_dict = scores.to_pandas().to_dict(orient="records")[0] if hasattr(scores, 'to_pandas') else scores
    for key, value in score_dict.items():
        if isinstance(value, (int, float)):
            print(f"  {key:25s}: {value:.4f}")

    # 打印 scores 的 DataFrame
    try:
        print("\n## 详细指标 (DataFrame)")
        print("-" * 40)
        df = scores.to_pandas()
        print(df.to_string())
    except Exception:
        pass

    # 保存完整结果
    output = {
        "metrics": {k: v for k, v in score_dict.items() if isinstance(v, (int, float))},
        "details": [
            {
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "num_contexts": len(r["contexts"]),
            }
            for r in results
        ]
    }
    output_path = Path(__file__).parent / "evaluation_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n📁 完整评估报告已保存至: {output_path}")


def main():
    print("=" * 60)
    print("🔍 RAG 管线质量评估")
    print("=" * 60)
    print()

    # 1. 检查前置条件
    check_prerequisites()

    # 2. 运行 RAG 管线，收集答案
    start = time.time()
    results = run_pipeline()
    elapsed = time.time() - start
    print(f"⏱️ 管线运行耗时: {elapsed:.1f}s")

    # 3. RAGAS 评估
    scores = run_ragas_evaluation(results)

    # 4. 输出报告
    print_report(results, scores)


if __name__ == "__main__":
    main()
