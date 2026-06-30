# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""会话记忆（Redis）：历史消息窗口 + 已收集实体。

实体记忆让多轮对话中无需重复追问（如已提供的 request_id 后续复用）。

本模块职责：
  - 以 Redis List 存储每个会话的对话历史，支持窗口截断（最近 N 条）。
  - 以 Redis String（JSON 序列化）存储每个会话已抽取到的实体（如 request_id、
    手机号等），避免多轮对话重复追问同一字段。
  - 所有操作均为异步（async/await），适配 FastAPI 异步框架。
"""

# 标准库：用于将 Python 对象与 JSON 字符串互转，以便存入 Redis
import json

# 项目内缓存模块：获取全局共享的 Redis 异步客户端实例
from app.cache.redis_client import get_redis

HISTORY_MAX = 20        # 历史消息窗口：每个会话最多保留最近 20 条消息，超出时自动截断
ENTITY_TTL = 60 * 60 * 6  # 实体记忆 6 小时：历史消息和实体 key 的 Redis 过期时长（秒）


def _hist_key(conv_id: str) -> str:
    """构造历史消息在 Redis 中的 key。

    参数：
        conv_id: 会话 ID，用于唯一标识一路对话。

    返回：
        形如 'mem:hist:<conv_id>' 的 Redis 键名，按命名空间区分不同类型数据。
    """
    return f"mem:hist:{conv_id}"  # 拼接前缀与会话 ID，形成唯一键名


def _ent_key(conv_id: str) -> str:
    """构造实体记忆在 Redis 中的 key。

    参数：
        conv_id: 会话 ID。

    返回：
        形如 'mem:ent:<conv_id>' 的 Redis 键名，与历史消息 key 区分命名空间。
    """
    return f"mem:ent:{conv_id}"  # 实体存储使用独立前缀 'mem:ent:'


async def append_message(conv_id: str, role: str, content: str) -> None:
    """向指定会话追加一条消息，并维护滑动窗口与过期时间。

    参数：
        conv_id:  会话 ID，标识消息归属的对话。
        role:     消息角色，如 'user'、'assistant'、'system'。
        content:  消息正文内容。

    返回：
        无（None）。

    副作用：
        - 将消息序列化为 JSON 后追加到 Redis List 右端。
        - 截断 List 只保留最新的 HISTORY_MAX 条，避免无限增长。
        - 重置 key 的过期时间为 ENTITY_TTL，保持活跃会话不过期。
    """
    r = get_redis()  # 获取全局 Redis 异步客户端
    # 将 {"role": role, "content": content} 序列化为 JSON 字符串后追加至 List 右端
    await r.rpush(_hist_key(conv_id), json.dumps({"role": role, "content": content}, ensure_ascii=False))
    # ltrim 只保留 List 中最后 HISTORY_MAX 条（索引 -HISTORY_MAX 到 -1），实现滑动窗口截断
    await r.ltrim(_hist_key(conv_id), -HISTORY_MAX, -1)
    # 每次写入时刷新 TTL，确保活跃会话不会因长时间未触碰而提前过期
    await r.expire(_hist_key(conv_id), ENTITY_TTL)


async def get_history(conv_id: str) -> list[dict]:
    """获取指定会话的全部历史消息（按时间顺序）。

    参数：
        conv_id: 会话 ID。

    返回：
        消息字典列表，每个元素形如 {"role": "user", "content": "..."}。
        若该会话无历史消息，返回空列表。
    """
    r = get_redis()  # 获取 Redis 客户端
    # lrange 0 -1 取出 List 中所有元素（字节字符串形式）
    items = await r.lrange(_hist_key(conv_id), 0, -1)
    # 将每条 JSON 字节字符串反序列化为 Python dict，返回消息列表
    return [json.loads(x) for x in items]


async def get_entities(conv_id: str) -> dict:
    """获取指定会话已收集到的实体字典。

    参数：
        conv_id: 会话 ID。

    返回：
        实体字典，如 {"request_id": "abc123", "phone": "138****0000"}。
        若尚无实体数据，返回空字典 {}。
    """
    r = get_redis()  # 获取 Redis 客户端
    # get 返回 JSON 字符串或 None（key 不存在时）
    raw = await r.get(_ent_key(conv_id))
    # 有数据则反序列化，否则返回空字典，避免调用方做 None 判断
    return json.loads(raw) if raw else {}


async def update_entities(conv_id: str, new_entities: dict) -> dict:
    """合并新抽取到的非空实体到记忆，返回合并后的实体。

    参数：
        conv_id:      会话 ID。
        new_entities: 本轮新抽取的实体字典，值为 None/空字符串/空集合的字段会被跳过。

    返回：
        合并后的完整实体字典（已持久化至 Redis）。

    设计说明：
        只有非空值才会覆盖/写入，防止新一轮空值意外清除已有数据；
        这样多轮对话中用户一旦提供某字段，后续轮次无需重复追问。
    """
    current = await get_entities(conv_id)  # 先读取当前已有实体，作为合并基础
    for k, v in (new_entities or {}).items():  # new_entities 为 None 时安全降级为空 dict
        if v not in (None, "", [], {}):  # 只保留非空值，避免用空值覆盖已有实体
            current[k] = v  # 将新实体写入（或覆盖）当前字典
    r = get_redis()  # 获取 Redis 客户端准备回写
    # 将合并后的实体序列化为 JSON 并写回 Redis，同时设置过期时间保持与历史消息一致
    await r.set(_ent_key(conv_id), json.dumps(current, ensure_ascii=False), ex=ENTITY_TTL)
    return current  # 返回合并后的实体，供调用方直接使用（无需再次 get）
