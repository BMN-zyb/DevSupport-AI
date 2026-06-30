# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Redis 客户端（异步），用于会话记忆、语义缓存、路由缓存。

本模块封装了与 Redis 的连接管理逻辑，提供全局共享的异步 Redis 客户端实例。
通过 lru_cache 装饰器确保整个应用生命周期内只创建一个客户端连接，避免重复建立连接的开销。
该客户端被语义缓存（semantic_cache.py）和路由缓存（route_cache.py）模块共同使用。
"""

# 标准库：functools 提供 lru_cache，用于将函数结果缓存（此处实现单例模式）
from functools import lru_cache

# 第三方库：redis.asyncio 是 Redis 官方 Python 客户端的异步版本，支持 async/await 非阻塞 I/O
import redis.asyncio as aioredis

# 项目内部配置：读取 Redis 连接地址等运行时配置项
from app.config import settings


@lru_cache  # lru_cache 将返回值缓存起来，同参数第二次调用直接返回缓存，此处无参数，等价于单例
def get_redis() -> aioredis.Redis:
    """创建并返回全局共享的异步 Redis 客户端实例（单例）。

    使用 lru_cache 保证整个进程只初始化一次连接，节省资源。
    decode_responses=True 让 Redis 返回的字节数据自动解码为 Python 字符串，
    无需在调用方手动 decode()。

    返回:
        aioredis.Redis: 可用于 async/await 的 Redis 异步客户端对象。
    """
    # 从配置中读取 Redis URL（如 redis://localhost:6379/0），并启用字符串自动解码
    return aioredis.from_url(settings.redis_url, decode_responses=True)
