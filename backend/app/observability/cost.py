# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Token / 成本统计：按会话与租户记录用量。

本模块职责：
  - 每次 LLM 调用结束后，将消耗的 token 数量异步写入数据库表 token_usage。
  - 提供按租户汇总的 token 用量查询接口，供成本看板/管理后台使用。
  - 通过 SQLAlchemy 异步 ORM 与数据库交互，适配 FastAPI 异步框架。
"""

# SQLAlchemy 聚合函数 func 用于 COUNT/SUM 等 SQL 聚合；select 用于构造查询语句
from sqlalchemy import func, select

# 项目内数据库模块：获取异步数据库会话工厂
from app.db import AsyncSessionLocal
# 项目内 ORM 模型：TokenUsage 对应数据库中记录 token 用量的表
from app.models import TokenUsage


async def record(tenant_id: str, conversation_id: str, model: str, total_tokens: int) -> None:
    """记录一次 LLM 调用的 token 消耗到数据库。

    参数：
        tenant_id:       租户 ID，用于多租户成本隔离统计。
        conversation_id: 会话 ID，用于按对话维度追踪用量。
        model:           调用的模型名称（如 'gpt-4o'、'claude-3-5-sonnet'），便于分模型计费。
        total_tokens:    本次调用消耗的 token 总数（prompt + completion）。

    返回：
        无（None）。

    设计说明：
        total_tokens <= 0 时直接返回，避免写入无效记录（如调用未成功时传入 0）。
    """
    if total_tokens <= 0:  # 过滤掉无效/零用量记录，节省数据库写入
        return
    async with AsyncSessionLocal() as s:  # 使用异步上下文管理器获取数据库会话，会话结束自动关闭
        s.add(
            TokenUsage(  # 构造 ORM 对象，映射到 token_usage 表的一行记录
                tenant_id=tenant_id,           # 租户标识
                conversation_id=conversation_id,  # 所属会话
                model=model,                   # 调用的模型名称
                total_tokens=total_tokens,     # 本次消耗的 token 总数
            )
        )
        await s.commit()  # 提交事务，将记录持久化到数据库


async def summary_by_tenant() -> list[dict]:
    """按租户汇总 token 用量（成本看板用）。

    返回：
        列表，每个元素为一个租户的汇总字典，格式：
            {
                "tenant_id":    str,  # 租户 ID
                "turns":        int,  # 该租户累计调用次数（行数）
                "total_tokens": int,  # 该租户累计消耗 token 总数
            }

    设计说明：
        使用 GROUP BY tenant_id 在数据库侧聚合，避免把大量行全部拉到内存后再统计。
        适用于管理后台成本看板场景，数据量大时建议增加索引或物化视图。
    """
    async with AsyncSessionLocal() as s:  # 获取异步数据库会话
        rows = (
            await s.execute(  # 执行异步查询，返回结果集
                select(
                    TokenUsage.tenant_id,                          # 查询字段：租户 ID
                    func.count().label("turns"),                   # 聚合：统计该租户的记录条数（调用轮次）
                    func.sum(TokenUsage.total_tokens).label("tokens"),  # 聚合：汇总该租户的 token 总量
                ).group_by(TokenUsage.tenant_id)  # 按租户分组，实现分租户统计
            )
        ).all()  # 取出所有分组结果行
        return [
            # 将每一行 Row 对象转换为标准字典，便于 JSON 序列化返回给前端
            {"tenant_id": r.tenant_id, "turns": int(r.turns), "total_tokens": int(r.tokens or 0)}
            # r.tokens 可能为 None（该租户无记录时 SUM 返回 NULL），用 or 0 保证整型安全
            for r in rows
        ]
