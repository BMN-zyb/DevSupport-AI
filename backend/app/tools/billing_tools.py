# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""套餐 / 用量 / 账单查询工具（真实查询 MySQL）。

本模块向 AI Agent 提供三个计费相关的查询工具：
  - query_plan：查询当前租户的套餐信息（QPS 上限、月配额、单价等），
    用于帮助开发者了解自己的资源上限，辅助限流问题诊断。
  - query_usage：查询指定月份（或全部月份）的 API 调用量与超额量，
    用于费用估算和用量异常排查。
  - query_bill：查询指定月份（或全部月份）的账单金额与费用明细，
    用于对账和费用咨询。

三个工具均在模块末尾通过 register() 注册到全局 REGISTRY，应用启动时
load_tools() 会 import 本模块以触发注册。
"""

# ---- 第三方库导入（SQLAlchemy ORM） ----
from sqlalchemy import select  # 构造 SELECT 查询语句

# ---- 项目内部模块导入 ----
from app.db import AsyncSessionLocal  # 异步数据库会话工厂
from app.models import Invoice, Plan, Tenant, UsageRecord  # ORM 模型：账单、套餐、租户、用量记录
from app.tools.registry import ToolContext, ToolSpec, register  # 工具注册相关


async def query_plan(args: dict, ctx: ToolContext) -> dict:
    """查询租户当前套餐（QPS、月配额、单价）。

    先查当前租户的 plan_id，再关联查询对应套餐的详细配置。
    用于 AI 回答"我的 QPS 上限是多少"、"我的月配额还有多少"等问题。

    Args:
        args: 本工具不需要任何参数（空字典即可）。
        ctx:  工具执行上下文，通过 ctx.tenant_id 定位当前租户。

    Returns:
        found=True 时返回套餐详情；found=False 时返回失败原因。
    """
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        tenant = (
            await s.execute(select(Tenant).where(Tenant.id == ctx.tenant_id))
        ).scalar_one_or_none()  # 查询当前租户记录，不存在则返回 None
        if tenant is None or tenant.plan_id is None:
            # 租户不存在，或租户尚未绑定套餐，无法返回套餐信息
            return {"found": False, "reason": "未找到套餐信息"}
        # 根据租户绑定的 plan_id 查询套餐详情（必然存在，用 scalar_one）
        plan = (await s.execute(select(Plan).where(Plan.id == tenant.plan_id))).scalar_one()
        return {
            "found": True,                                      # 标记成功找到套餐
            "plan_name": plan.name,                            # 套餐名称（如"标准版"）
            "qps_limit": plan.qps_limit,                       # QPS 上限（每秒最大请求数）
            "monthly_quota": plan.monthly_quota,               # 月调用量配额（超出按超额单价计费）
            "price_per_call": plan.price_per_call,             # 正常单次调用单价
            "overage_price_per_call": plan.overage_price_per_call,  # 超额部分的单次调用单价
        }


async def query_usage(args: dict, ctx: ToolContext) -> dict:
    """查询某月调用量与超额量。

    查询指定月份（或不指定则查全部月份）的 API 调用量统计，
    包括总调用次数和超出套餐配额的超额次数，用于费用估算和异常排查。

    Args:
        args: 可选参数 "month"（格式 YYYY-MM，如 "2026-06"），不传则查全部月份。
        ctx:  工具执行上下文，通过 ctx.tenant_id 限定租户。

    Returns:
        found=True 时返回各月用量列表；found=False 时返回失败原因。
    """
    month = args.get("month")  # 从参数中提取月份过滤条件（可为 None）
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        # 基础查询：过滤当前租户的用量记录
        stmt = select(UsageRecord).where(UsageRecord.tenant_id == ctx.tenant_id)
        if month:
            stmt = stmt.where(UsageRecord.month == month)  # 如指定月份则添加月份过滤
        stmt = stmt.order_by(UsageRecord.month)            # 按月份升序排列，便于趋势分析
        rows = (await s.execute(stmt)).scalars().all()     # 执行查询，获取所有匹配记录
        if not rows:
            # 没有找到任何用量记录（该租户可能尚未产生调用）
            return {"found": False, "reason": "未找到用量记录"}
        return {
            "found": True,  # 标记成功找到记录
            "usage": [
                # 将每条 UsageRecord 转换为简洁的字典格式
                {"month": r.month, "call_count": r.call_count, "overage_count": r.overage_count}
                for r in rows  # 遍历所有查询结果
            ],
        }


async def query_bill(args: dict, ctx: ToolContext) -> dict:
    """查询某月账单及费用构成。

    查询指定月份（或不指定则查全部月份）的账单信息，
    包括账单金额、支付状态和费用明细（基础费用、超额费用等），
    用于对账或回答"我这个月为什么收费这么多"类问题。

    Args:
        args: 可选参数 "month"（格式 YYYY-MM，如 "2026-06"），不传则查全部月份。
        ctx:  工具执行上下文，通过 ctx.tenant_id 限定租户。

    Returns:
        found=True 时返回账单列表；found=False 时返回失败原因。
    """
    month = args.get("month")  # 从参数中提取月份过滤条件（可为 None）
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        # 基础查询：过滤当前租户的账单记录
        stmt = select(Invoice).where(Invoice.tenant_id == ctx.tenant_id)
        if month:
            stmt = stmt.where(Invoice.month == month)  # 如指定月份则添加月份过滤
        stmt = stmt.order_by(Invoice.month)            # 按月份升序排列
        rows = (await s.execute(stmt)).scalars().all() # 执行查询，获取所有匹配账单
        if not rows:
            # 没有找到任何账单（该租户可能尚未产生费用或对账单）
            return {"found": False, "reason": "未找到账单"}
        return {
            "found": True,  # 标记成功找到记录
            "bills": [
                # 将每条 Invoice 记录转换为字典，items 包含费用明细（JSON 格式）
                {"month": r.month, "amount": r.amount, "status": r.status, "items": r.items}
                for r in rows  # 遍历所有查询结果
            ],
        }


# ---- 将 query_plan 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="query_plan",  # 工具名称
    description="查询当前租户的套餐信息：套餐名、QPS 上限、月调用量配额、单价与超额单价。",
    parameters={"type": "object", "properties": {}},  # 无需任何参数
    func=query_plan,      # 绑定套餐查询函数
    category="billing",   # 工具分类：计费
))

# ---- 将 query_usage 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="query_usage",   # 工具名称
    description="查询租户某月（YYYY-MM）的调用量与超额量；不传 month 则返回全部月份。",
    parameters={
        "type": "object",
        "properties": {"month": {"type": "string", "description": "月份，如 2026-06"}},
        # month 为可选参数，无 required 字段
    },
    func=query_usage,     # 绑定用量查询函数
    category="billing",   # 工具分类：计费
))

# ---- 将 query_bill 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="query_bill",    # 工具名称
    description="查询租户某月账单金额与费用构成（基础费用、超额费用）。",
    parameters={
        "type": "object",
        "properties": {"month": {"type": "string", "description": "月份，如 2026-06"}},
        # month 为可选参数，无 required 字段
    },
    func=query_bill,      # 绑定账单查询函数
    category="billing",   # 工具分类：计费
))
