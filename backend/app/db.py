# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""数据库连接与会话管理。

- 应用运行时：异步引擎（aiomysql）+ AsyncSession。
- 脚本（建表/灌数据）：同步引擎（pymysql）+ Session。
两者共享同一套 ORM 模型（Base）。

本模块负责：
  1. 声明 ORM 基类 Base，所有模型类均继承自此类；
  2. 创建异步数据库引擎及异步会话工厂，供 FastAPI 路由依赖注入使用；
  3. 提供 get_db() 异步生成器，作为 FastAPI Depends 来自动管理会话生命周期；
  4. 创建同步数据库引擎及同步会话工厂，供脚本（如建表、数据初始化）使用。
"""

# 标准库：用于声明异步生成器的类型注解
from collections.abc import AsyncGenerator

# SQLAlchemy 同步引擎工厂函数
from sqlalchemy import create_engine
# 异步相关：AsyncSession 会话类、async_sessionmaker 异步会话工厂、create_async_engine 异步引擎工厂
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
# DeclarativeBase：ORM 声明式基类；sessionmaker：同步会话工厂
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# 导入全局配置，获取数据库 DSN 等连接参数
from app.config import settings


class Base(DeclarativeBase):
    """ORM 模型基类。

    所有 SQLAlchemy ORM 模型（Tenant、User、Conversation 等）均继承此类。
    DeclarativeBase 会自动将子类注册到元数据（MetaData），
    从而支持 Base.metadata.create_all() 一次性建表。
    """
    pass


# ----- 异步（应用运行时） -----

# 创建异步数据库引擎：使用 aiomysql 驱动，底层基于 asyncio
async_engine = create_async_engine(
    settings.mysql_dsn_async,   # 异步 DSN，格式：mysql+aiomysql://...
    # 注意：aiomysql 0.2 的 ping() 签名与 SQLAlchemy pre_ping 不兼容，故关闭；
    # 用 pool_recycle 回收旧连接以避免 MySQL 主动断连。
    pool_pre_ping=False,        # 关闭连接预检（兼容性问题，见上注释）
    pool_recycle=1800,          # 连接存活超过 1800 秒（30 分钟）后自动回收，避免 MySQL 主动断开空闲连接
    echo=False,                 # 不打印 SQL 日志，生产环境建议保持关闭以减少日志噪音
)

# 创建异步会话工厂：每次调用产生一个新的 AsyncSession 实例
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,          # 绑定到上面创建的异步引擎
    expire_on_commit=False,     # commit 后不使对象过期，避免 lazy load 触发额外查询
    class_=AsyncSession         # 明确指定会话类为异步版本
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：提供异步数据库会话。

    作为 FastAPI 路由的 Depends 使用，框架会在请求开始时获取会话，
    在请求结束（无论成功还是异常）后自动关闭会话并归还连接到连接池。

    用法示例：
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...

    Yields:
        AsyncSession: 与当前请求绑定的异步数据库会话实例
    """
    async with AsyncSessionLocal() as session:  # 创建新的异步会话，离开 with 块时自动关闭
        yield session                           # 将会话提供给路由函数使用


# ----- 同步（脚本） -----

# 创建同步数据库引擎：使用 pymysql 驱动，适合脚本场景（建表、数据初始化等）
sync_engine = create_engine(settings.mysql_dsn_sync, pool_pre_ping=True, echo=False)
# pool_pre_ping=True：同步引擎开启连接预检，确保获取的连接是存活的（脚本场景可接受轻微延迟）
# echo=False：不打印 SQL 语句

# 创建同步会话工厂：供脚本直接使用，无需异步上下文
SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False)
# expire_on_commit=False：与异步版本保持一致，commit 后对象属性依然可直接访问
