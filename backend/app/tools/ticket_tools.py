# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""工单工具（真实写入 MySQL）+ 高风险操作占位（AI 不可直接执行）。

本模块向 AI Agent 提供两个工单管理工具：
  - create_ticket：当问题无法自助解决时，创建技术支持工单（写入 MySQL），
    同时附带 AI 的诊断摘要与证据，让人工客服快速了解背景。
  - query_ticket：按工单 ID 查询工单当前状态与优先级，
    用于告知用户工单处理进度。

此外，本模块还注册了三个"高风险占位工具"（reset_api_key、change_plan、refund），
这些工具标记为 high_risk=True，不会暴露给 AI，只能由后台人工审批后执行。
实际函数 _high_risk_blocked 直接返回阻断提示，作为双重保险防止 AI 误调用。

所有工具在模块末尾通过 register() 注册到全局 REGISTRY。
"""

# ---- 标准库导入 ----
import uuid            # 用于生成工单 ID 后缀，保证全局唯一性
from datetime import datetime  # 用于生成工单 ID 时间戳部分（演示模式固定日期）

# ---- 第三方库导入（SQLAlchemy ORM） ----
from sqlalchemy import select  # 构造 SELECT 查询语句

# ---- 项目内部模块导入 ----
from app.db import AsyncSessionLocal  # 异步数据库会话工厂
from app.models import Ticket         # ORM 模型：工单表
from app.tools.registry import ToolContext, ToolSpec, register  # 工具注册相关


async def create_ticket(args: dict, ctx: ToolContext) -> dict:
    """创建技术支持工单（含 AI 诊断摘要与证据）。

    在数据库中插入一条新工单记录，工单 ID 由固定日期前缀 + UUID 后缀组成，
    保证唯一性的同时便于按日期归档检索。工单初始状态为 "new"。

    Args:
        args: 工单参数字典，支持以下字段：
            - title (str, 必填): 工单标题，简述问题。
            - category (str, 必填): 工单类别，如 "API报错/套餐账单/数据质量/故障投诉"。
            - summary (str, 必填): 问题摘要，AI 应填写完整诊断背景。
            - priority (str, 可选): 优先级 P0~P3，默认 P2。
            - user_id (str, 可选): 用户标识，默认空字符串。
            - related_request_ids (list, 可选): 关联的 request_id 列表。
            - related_endpoint (str, 可选): 关联接口路径。
            - error_code (str, 可选): 触发工单的错误码。
            - evidence (str, 可选): AI 收集的证据（查询结果摘要等）。
            - ai_diagnosis (str, 可选): AI 的诊断结论与建议。
            - conversation_id (str, 可选): 关联对话 ID，便于客服回溯上下文。
        ctx:  工具执行上下文，提供 tenant_id。

    Returns:
        包含新建工单 ID、状态和优先级的字典。
    """
    # 构造工单 ID：前缀 "tk_" + 固定演示日期（20260615）+ "_" + 6位 UUID 随机串
    # 使用固定日期而非 datetime.now() 是为了演示数据的可复现性（测试环境一致性）
    ticket_id = "tk_" + datetime(2026, 6, 15).strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:6]
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        s.add(
            Ticket(
                ticket_id=ticket_id,                              # 唯一工单 ID
                tenant_id=ctx.tenant_id,                         # 所属租户 ID（从上下文获取）
                user_id=args.get("user_id", ""),                 # 用户 ID，默认空字符串
                category=args.get("category", "其它"),            # 工单类别，默认"其它"
                priority=args.get("priority", "P2"),              # 优先级，默认 P2（中优先级）
                status="new",                                     # 初始状态固定为"new"（新建）
                title=args.get("title", "技术支持工单"),            # 工单标题
                summary=args.get("summary", ""),                  # 问题摘要
                related_request_ids=args.get("related_request_ids", []),  # 相关请求 ID 列表
                related_endpoint=args.get("related_endpoint"),    # 关联接口路径
                error_code=args.get("error_code"),                # 错误码
                evidence=args.get("evidence", ""),                # 证据文本
                ai_diagnosis=args.get("ai_diagnosis", ""),        # AI 诊断结论
                conversation_id=args.get("conversation_id"),      # 关联对话 ID
            )
        )
        await s.commit()  # 提交事务，将工单持久化到数据库
    # 返回关键字段，让 AI 可以告知用户工单号和当前状态
    return {"ticket_id": ticket_id, "status": "new", "priority": args.get("priority", "P2")}


async def query_ticket(args: dict, ctx: ToolContext) -> dict:
    """按 ticket_id 查询工单状态。

    根据工单 ID 查询数据库，返回工单的当前状态与优先级，
    用于回答用户"我的工单处理到哪一步了"类问题。
    客户侧调用时会进行租户隔离，防止跨租户查询他人工单。

    Args:
        args: 包含 "ticket_id" 字段的参数字典。
        ctx:  工具执行上下文，提供 tenant_id 与 is_internal 标记。

    Returns:
        found=True 时返回工单基本信息；found=False 时表示工单不存在或无权访问。
    """
    ticket_id = args.get("ticket_id")  # 从参数中提取工单 ID
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        t = (
            await s.execute(select(Ticket).where(Ticket.ticket_id == ticket_id))
        ).scalar_one_or_none()  # 按工单 ID 精确查找，不存在则返回 None
        if t is None:
            return {"found": False}  # 工单不存在
        if not ctx.is_internal and t.tenant_id != ctx.tenant_id:
            # 非内部调用且工单归属租户与当前租户不符，拒绝访问（租户隔离）
            return {"found": False, "reason": "无权访问"}
        # 返回工单关键字段（不返回 evidence/ai_diagnosis 等内部诊断信息）
        return {"found": True, "ticket_id": t.ticket_id, "status": t.status,
                "priority": t.priority, "title": t.title}


async def _high_risk_blocked(args: dict, ctx: ToolContext) -> dict:
    """高风险操作的阻断占位函数。

    所有标记为 high_risk=True 的工具（如重置 API Key、变更套餐、退款）
    均绑定此函数作为执行体。即使绕过了 registry.execute() 的高风险拦截，
    此函数也会返回阻断提示，作为第二层防护。

    Args:
        args: 忽略，任何参数均不处理。
        ctx:  忽略。

    Returns:
        始终返回阻断提示字典。
    """
    return {"blocked": True, "reason": "高风险操作必须由人工或后台审批执行"}


# ---- 将 create_ticket 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="create_ticket",
    description="当问题无法自助解决、需人工介入或用户要求人工时，创建技术支持工单，附带 AI 诊断摘要与证据。",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},                                          # 工单标题
            "category": {"type": "string", "description": "如 API报错/套餐账单/数据质量/故障投诉"},
            "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},  # 优先级枚举
            "summary": {"type": "string"},                                        # 问题摘要
            "related_request_ids": {"type": "array", "items": {"type": "string"}},  # 关联请求 ID
            "related_endpoint": {"type": "string"},                               # 关联接口
            "error_code": {"type": "string"},                                     # 错误码
            "evidence": {"type": "string"},                                       # 证据
            "ai_diagnosis": {"type": "string"},                                   # AI 诊断
        },
        "required": ["title", "category", "summary"],  # 这三个字段为必填
    },
    func=create_ticket,   # 绑定工单创建函数
    category="ticket",    # 工具分类：工单
))

# ---- 将 query_ticket 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="query_ticket",
    description="按 ticket_id 查询工单当前状态与优先级。",
    parameters={"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]},
    func=query_ticket,    # 绑定工单查询函数
    category="ticket",    # 工具分类：工单
))

# ---- 高风险工具：注册但标记 high_risk，不暴露给 AI ----
# 遍历高风险操作列表，批量注册为 high_risk=True 的工具
# 这些工具不会出现在 openai_tools() 返回的列表中，AI 无法主动调用
for _name, _desc in [
    ("reset_api_key", "重置 API Key"),     # 高风险：重置密钥会导致原密钥立即失效
    ("change_plan", "变更套餐"),            # 高风险：变更套餐会影响计费和资源配额
    ("refund", "退款/账单调整"),             # 高风险：财务操作，需人工审批
]:
    register(ToolSpec(
        name=_name,                          # 工具名称
        description=_desc,                   # 工具描述
        parameters={"type": "object", "properties": {}},  # 无需参数（占位注册）
        func=_high_risk_blocked,             # 绑定阻断函数，即使被调用也会返回拒绝
        high_risk=True,                      # 标记为高风险，不暴露给 AI
        category="high_risk",               # 工具分类：高风险
    ))
