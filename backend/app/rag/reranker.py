# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""重排序：对混合召回的候选用 gte-rerank-v2 精排，取 top_n。

本模块实现了 RAG 系统的重排阶段（Reranking Stage）：
- 混合检索（retriever）召回的候选片段数量较多（~40），可能包含噪声；
- 通过调用专门的 Cross-Encoder 重排模型（gte-rerank-v2）对 query 与每个候选
  进行精细的语义匹配打分，比 Embedding 余弦相似度更准确；
- 取分数最高的 top_n 条结果传给压缩（compressor）和生成（LLM）阶段，
  既保证质量又控制 prompt 长度。
"""

from app.llm import client  # LLM/Rerank 客户端，封装了 DashScope gte-rerank-v2 调用


async def rerank_candidates(
    query: str, candidates: list[dict], top_n: int = 5
) -> list[dict]:
    """对候选片段精排，返回带 rerank_score 的 top_n（降序）。

    调用流程：
    1. 提取所有候选片段的 content 文本列表；
    2. 将 query 和 docs 列表传给 client.rerank，调用重排模型批量打分；
    3. 按模型返回的 index 将分数映射回原始候选字典，注入 rerank_score 字段；
    4. 返回按相关性降序排列的 top_n 条结果（client.rerank 已在内部截取）。

    参数：
        query:      用户查询字符串，作为重排的参考问题
        candidates: 候选切片字典列表（来自 retriever.hybrid_search 的返回值）
        top_n:      保留得分最高的 top_n 条结果，默认 5
    返回：
        字典列表，每项为原始候选切片字典加上 "rerank_score" 字段，按 rerank_score 降序排列；
        若 candidates 为空则返回空列表
    """
    if not candidates:  # 候选列表为空，无需调用模型，直接返回空列表
        return []
    docs = [c["content"] for c in candidates]  # 提取所有候选的文本内容，作为重排模型的文档输入
    results = await client.rerank(query, docs, top_n=top_n)  # 异步调用重排模型，返回按相关性降序的 top_n 结果
    reranked = []  # 存放映射回元数据后的重排结果
    for r in results:  # 遍历重排模型返回的每条结果
        item = dict(candidates[r["index"]])  # 通过 index 找到对应的原始候选切片字典，复制一份避免污染
        item["rerank_score"] = r["score"]   # 将重排模型给出的相关性分数写入字典
        reranked.append(item)  # 加入最终结果列表
    return reranked  # 返回带 rerank_score 的 top_n 片段列表（已按分数降序排列）
