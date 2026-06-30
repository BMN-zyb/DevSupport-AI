# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""路由缓存：缓存意图识别结果（intent/route/entities），命中则跳过意图识别 LLM。

仅在无对话历史（首轮）时使用，避免相同 query 在不同上下文下复用错误结果。

本模块提供两个核心操作：
  - get(query)：根据 query 文本从 Redis 中查找已缓存的意图识别结果。
  - put(query, result)：将新的意图识别结果写入 Redis，TTL 为 24 小时。

命中路由缓存可省去一次 LLM 推理调用，显著降低延迟与成本。
缓存 key 使用 query 的 MD5 哈希，兼顾唯一性与存储效率。
"""

# 标准库：hashlib 用于生成缓存 key 的 MD5 哈希值，确保不同 query 映射到不同 key
import hashlib
# 标准库：json 用于将字典序列化为字符串后存入 Redis，以及从 Redis 读取时反序列化
import json

# 项目内部：获取全局共享的异步 Redis 客户端实例
from app.cache.redis_client import get_redis

# 缓存过期时间（秒）：24 小时 = 60s * 60min * 24h
# 路由缓存的时效性要求不高，设置较长 TTL 以提升命中率
TTL = 60 * 60 * 24


def _key(query: str) -> str:
    """根据用户 query 生成唯一的 Redis 缓存 key。

    对 query 做规范化处理（去首尾空格 + 小写化）后计算 MD5，
    使得大小写和多余空格不同的相同语义 query 能命中同一缓存条目。

    参数:
        query (str): 用户输入的原始查询文本。

    返回:
        str: 形如 "routecache:<md5hex>" 的 Redis key 字符串。
    """
    norm = query.strip().lower()  # 规范化：去除首尾空白并转小写，提升缓存命中率
    return "routecache:" + hashlib.md5(norm.encode("utf-8")).hexdigest()  # 拼接命名空间前缀与 MD5 哈希


async def get(query: str) -> dict | None:
    """从路由缓存中查找给定 query 的意图识别结果。

    若 Redis 中存在对应缓存，则反序列化后直接返回；否则返回 None，
    调用方需要继续执行 LLM 意图识别流程。

    参数:
        query (str): 用户输入的查询文本。

    返回:
        dict | None: 命中时返回包含 intent/confidence/entities/route 的字典；
                     未命中时返回 None。
    """
    raw = await get_redis().get(_key(query))  # 异步从 Redis 读取 key 对应的字符串值
    return json.loads(raw) if raw else None  # 若有值则反序列化为字典，否则返回 None


async def put(query: str, result: dict) -> None:
    """将意图识别结果写入路由缓存，设置 24 小时过期。

    只缓存固定字段（intent/confidence/entities/route），
    过滤掉 result 中可能存在的其他临时字段，保持缓存内容精简。

    参数:
        query (str): 原始查询文本（作为缓存 key 的生成源）。
        result (dict): 意图识别结果字典，需包含 intent/confidence/entities/route 字段。

    返回:
        None
    """
    # 只保留与路由决策相关的核心字段，避免缓存无关或过大的数据
    payload = {
        "intent": result["intent"],            # 意图标签，如 "doc_qa"、"diagnose" 等
        "confidence": result["confidence"],    # 意图识别的置信度分数（0~1）
        "entities": result["entities"],        # 从 query 中提取的实体信息（如错误码、产品名）
        "route": result["route"],              # 路由目标，决定交由哪个 Agent 处理
    }
    # 序列化后写入 Redis，ensure_ascii=False 保留中文字符避免被转义；ex 设置过期时间
    await get_redis().set(_key(query), json.dumps(payload, ensure_ascii=False), ex=TTL)
