# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""语义缓存：对热点/相似问题命中缓存，跳过完整 Agent 链路。

仅缓存通用文档问答(doc_qa)结果，避免缓存与租户/请求强相关的诊断、账单答案。
相似度用 query embedding 余弦比较，阈值见 settings.semantic_cache_sim_threshold。

工作流程：
  1. 对用户 query 做 embedding（向量化），得到查询向量。
  2. 在 Redis List 中取出该租户的全部缓存条目，逐条计算余弦相似度。
  3. 找到相似度最高且超过阈值的条目，直接返回其缓存答案，跳过 LLM 推理。
  4. 若无命中，则在完成 LLM 推理后，将新结果连同 embedding 写入缓存。

每个租户独立维护一个 Redis List，按 LPUSH 顺序保留最新的 MAX_ENTRIES 条，
实现近似 LRU 淘汰，防止缓存无限膨胀。
"""

# 标准库：json 用于将缓存条目序列化为字符串存入 Redis，以及读取时反序列化
import json

# 第三方库：numpy 用于高效进行向量点积和范数计算，实现余弦相似度
import numpy as np

# 项目内部：获取全局共享的异步 Redis 客户端实例
from app.cache.redis_client import get_redis
# 项目内部：读取语义缓存相似度阈值等配置项
from app.config import settings
# 项目内部：调用 embed_one 获取 query 的文本向量表示
from app.llm import client

# 每个租户缓存的最大条目数；超出时通过 ltrim 删除旧条目，防止内存无限增长
MAX_ENTRIES = 200


def _key(tenant_id: str) -> str:
    """生成指定租户的语义缓存 Redis List key。

    使用租户 ID 作为命名空间，保证不同租户的缓存完全隔离，
    避免跨租户的答案污染。

    参数:
        tenant_id (str): 租户唯一标识符。

    返回:
        str: 形如 "semcache:<tenant_id>" 的 Redis key 字符串。
    """
    return f"semcache:{tenant_id}"  # 拼接命名空间前缀与租户 ID，形成隔离 key


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。

    余弦相似度 = (a · b) / (‖a‖ × ‖b‖)，结果范围 [-1, 1]。
    分母加 1e-9 防止零向量时出现除以零的数值错误。

    参数:
        a (np.ndarray): 第一个向量（通常为查询向量）。
        b (np.ndarray): 第二个向量（通常为缓存条目的 embedding）。

    返回:
        float: 两向量的余弦相似度，值越接近 1 表示语义越相似。
    """
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))  # 点积除以两向量模的乘积，+1e-9 防除零


async def get(tenant_id: str, query: str) -> tuple[dict | None, list[float]]:
    """在该租户的语义缓存中查找与 query 最相似的已缓存答案。

    返回 (命中结果或 None, query 向量)。向量回传以便 put 复用，避免重复 embedding。
    若命中，返回的字典中包含 similarity 字段（本次相似度得分），供调用方记录或日志使用。

    参数:
        tenant_id (str): 当前请求的租户 ID，用于查找对应的缓存 List。
        query (str): 用户输入的查询文本。

    返回:
        tuple[dict | None, list[float]]:
            - 若命中缓存，第一个元素为含 answer/citations/card/intent/similarity 的字典；
              未命中则为 None。
            - 第二个元素始终为 query 的 embedding 向量，供调用方在 put 时复用。
    """
    qv = await client.embed_one(query)  # 对用户 query 进行向量化，得到浮点数列表
    r = get_redis()  # 获取全局 Redis 客户端
    raw = await r.lrange(_key(tenant_id), 0, -1)  # 读取该租户缓存 List 的全部条目（0 到末尾）
    if not raw:  # 缓存为空，直接返回未命中
        return None, qv
    qa = np.array(qv)  # 将查询向量转为 numpy 数组，便于后续矩阵运算
    best, best_sim = None, -1.0  # 初始化最优候选条目和最高相似度
    for item in raw:  # 遍历所有缓存条目，逐条计算相似度
        e = json.loads(item)  # 反序列化 JSON 字符串为字典
        sim = _cosine(qa, np.array(e["embedding"]))  # 计算当前条目与查询向量的余弦相似度
        if sim > best_sim:  # 若当前条目相似度更高，则更新最优候选
            best_sim, best = sim, e
    if best is not None and best_sim >= settings.semantic_cache_sim_threshold:
        # 最高相似度达到或超过配置阈值，视为命中，构造并返回缓存答案
        return (
            {
                "answer": best["answer"],              # 缓存的完整答案文本
                "citations": best.get("citations", []),  # 答案引用的文档来源列表，可能为空
                "card": best.get("card"),              # 结构化卡片数据，可能为 None
                "intent": best.get("intent"),          # 意图标签，便于调用方记录统计
                "similarity": round(best_sim, 4),      # 本次命中的相似度得分，保留4位小数
            },
            qv,  # 同时返回查询向量，供 put 时复用
        )
    return None, qv  # 相似度未达阈值，返回未命中（None）和查询向量


async def put(tenant_id: str, query: str, result: dict, embedding: list[float]) -> None:
    """将新的问答结果写入该租户的语义缓存。

    使用调用方传入的 embedding（来自 get 的返回值）避免重复向量化，节省 API 调用开销。
    写入后通过 ltrim 将 List 截断为最多 MAX_ENTRIES 条，实现近似 LRU 淘汰。

    参数:
        tenant_id (str): 当前请求的租户 ID。
        query (str): 用户原始查询文本（保存到缓存条目，便于调试和追踪）。
        result (dict): LLM 生成的完整回答字典，需包含 answer 字段。
        embedding (list[float]): query 的向量表示，由 get 返回后传入，避免重复计算。

    返回:
        None
    """
    r = get_redis()  # 获取全局 Redis 客户端
    # 构造缓存条目，序列化为 JSON 字符串；ensure_ascii=False 保留中文字符
    entry = json.dumps(
        {
            "query": query,                          # 原始查询文本，便于人工审查缓存内容
            "embedding": embedding,                  # query 向量，用于后续相似度比较
            "answer": result["answer"],              # 完整答案文本
            "citations": result.get("citations", []),  # 引用来源列表
            "card": result.get("card"),              # 结构化卡片，可能为 None
            "intent": result.get("intent"),          # 意图标签
        },
        ensure_ascii=False,  # 保留中文等非 ASCII 字符，避免转义后字符串膨胀
    )
    await r.lpush(_key(tenant_id), entry)  # 将新条目插入 List 头部（最新条目排最前）
    # 按租户保留最新 MAX_ENTRIES 条，近似 LRU 防止缓存无限膨胀
    await r.ltrim(_key(tenant_id), 0, MAX_ENTRIES - 1)  # 截断 List，超出部分（旧条目）被自动删除
