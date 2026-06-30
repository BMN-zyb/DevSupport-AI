# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""工单查询接口（客户侧：我的工单）+ 反馈接口。

本模块是 DevSupport-AI 项目的工单与用户反馈路由层，提供如下能力：
  - GET  /api/tickets          ：列出当前用户（或租户）最近的工单列表（最多 50 条）
  - GET  /api/tickets/{ticket_id}：查询指定工单的详细信息（含 AI 诊断、证据等字段）
  - POST /api/feedback          ：提交用户对 AI 回答的反馈（满意/不满意/转人工），
                                   并在选择"转人工"时自动创建工单

设计要点：
  1. 工单列表查询实现多租户隔离：内部用户可查看全部工单，普通客户只能查看本租户工单。
  2. 工单详情查询使用 assert_tenant_access 进行跨租户访问校验，防止越权读取。
  3. 反馈接口同时支持满意度记录与一键转人工两种场景，通过 type 字段区分。
  4. 转人工时自动创建工单，附带最近一条用户消息作为标题，减少人工录入成本。
"""

# ── FastAPI 核心依赖 ──────────────────────────────────────────────────────────
from fastapi import APIRouter, Depends, HTTPException  # APIRouter 分组路由；Depends 注入依赖；HTTPException 抛出 HTTP 错误

# ── SQLAlchemy 查询工具 ───────────────────────────────────────────────────────
from sqlalchemy import select  # 构造 SELECT 语句的声明式 API
from sqlalchemy.ext.asyncio import AsyncSession  # 异步数据库会话，配合 await 使用

# ── 标准库 ────────────────────────────────────────────────────────────────────
import uuid  # 用于生成工单 ID 中的随机唯一串（uuid4().hex）
from datetime import datetime  # 用于生成工单 ID 中的日期时间戳部分

# ── SQLAlchemy 排序工具 ───────────────────────────────────────────────────────
from sqlalchemy import desc  # 构造 ORDER BY 降序排列的辅助函数

# ── 项目内部模块 ──────────────────────────────────────────────────────────────
from app.db import get_db  # 获取异步数据库会话的依赖函数
from app.deps import CurrentUser, assert_tenant_access, get_current_user  # CurrentUser：当前用户数据类；assert_tenant_access：租户访问权限校验函数；get_current_user：提取并验证当前用户的依赖
from app.models import Conversation, Feedback, Message, Ticket  # ORM 模型：Conversation 会话、Feedback 反馈记录、Message 消息、Ticket 工单
from app.schemas.chat import FeedbackRequest  # Pydantic Schema：反馈请求体，包含 conversation_id/message_id/type 等字段

# ── 路由器声明 ────────────────────────────────────────────────────────────────
# prefix="/api" 使路由挂载在 /api 下；tags=["tickets"] 用于 OpenAPI 文档界面分组展示
router = APIRouter(prefix="/api", tags=["tickets"])


@router.get("/tickets")
async def my_tickets(
    user: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    """列出当前用户可见的工单列表（按创建时间降序，最多 50 条）。

    多租户隔离逻辑：
      - 内部用户（role=internal）：可查看系统中所有租户的工单，便于运营人员全局监控。
      - 普通客户（role 非 internal）：只能查看本租户（user.tenant_id）下的工单，防止跨租户数据泄露。

    参数：
        user (CurrentUser): 由 get_current_user 依赖注入，含角色与租户信息，用于鉴权与数据过滤。
        db   (AsyncSession): 由 get_db 依赖注入的异步数据库会话。

    返回：
        dict: {"tickets": [<工单概要>, ...]}，每条工单含 ticket_id/title/category/priority/status/error_code/created_at。
    """
    # 构造基础查询：按创建时间降序排列，最多返回 50 条工单（避免数据量过大）
    stmt = select(Ticket).order_by(Ticket.created_at.desc()).limit(50)
    if not user.is_internal:
        # 非内部用户只能查看本租户工单，通过 WHERE tenant_id = <用户租户ID> 实现数据隔离
        stmt = stmt.where(Ticket.tenant_id == user.tenant_id)
    # 执行查询并将结果转换为 ORM 对象列表
    rows = (await db.execute(stmt)).scalars().all()
    # 将 ORM 对象列表序列化为前端所需的字典列表（仅返回列表页需要的精简字段）
    return {
        "tickets": [
            {"ticket_id": t.ticket_id, "title": t.title, "category": t.category,
             "priority": t.priority, "status": t.status, "error_code": t.error_code,
             "created_at": t.created_at.isoformat()}  # isoformat() 将 datetime 转为 ISO 8601 字符串，便于前端解析
            for t in rows
        ]
    }


@router.get("/tickets/{ticket_id}")
async def ticket_detail(
    ticket_id: str, user: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    """获取指定工单的详细信息（含 AI 诊断、证据、关联请求等字段）。

    在返回工单详情前会进行租户访问校验，确保用户只能查看本租户的工单（内部用户除外）。

    参数：
        ticket_id (str)     : 工单唯一标识符（如 "tk_20260615_abc123"）。
        user      (CurrentUser): 由 get_current_user 依赖注入，用于租户权限校验。
        db        (AsyncSession): 由 get_db 依赖注入的异步数据库会话。

    返回：
        dict: 工单完整详情，包含 ticket_id/title/category/priority/status/summary/
              related_request_ids/related_endpoint/error_code/ai_diagnosis/evidence/
              assignee/conversation_id/created_at。

    异常：
        HTTPException(404): 指定 ticket_id 的工单不存在时抛出。
        HTTPException(403): 当前用户无权访问该工单（跨租户）时由 assert_tenant_access 抛出。
    """
    # 按 ticket_id 精确查询单条工单记录，scalar_one_or_none 返回对象或 None
    t = (await db.execute(select(Ticket).where(Ticket.ticket_id == ticket_id))).scalar_one_or_none()
    if t is None:
        # 工单不存在时返回 404，不区分"不存在"与"无权访问"可防止工单 ID 枚举攻击
        raise HTTPException(404, "工单不存在")
    # 校验当前用户是否有权访问该工单所属租户；非内部用户跨租户访问将抛出 403
    assert_tenant_access(user, t.tenant_id)
    # 返回工单详情，包含 AI 诊断结论、证据链、关联 API 请求 ID 等运营分析字段
    return {
        "ticket_id": t.ticket_id,  # 工单唯一标识
        "title": t.title,  # 工单标题（通常来自用户最后一条消息的前 60 字符）
        "category": t.category,  # 工单分类（如 API 异常、账单问题等）
        "priority": t.priority,  # 优先级（P1~P4，P1 最高）
        "status": t.status,  # 工单当前状态（new/in_progress/resolved/closed）
        "summary": t.summary,  # 工单问题摘要
        "related_request_ids": t.related_request_ids,  # 相关 API 请求 ID 列表（用于追踪链路）
        "related_endpoint": t.related_endpoint,  # 出现问题的 API 端点路径
        "error_code": t.error_code,  # 错误码（HTTP 状态码或业务错误码）
        "ai_diagnosis": t.ai_diagnosis,  # AI 自动生成的故障诊断说明
        "evidence": t.evidence,  # 支撑诊断结论的证据（如日志片段、检索到的知识库内容）
        "assignee": t.assignee,  # 当前处理人（人工接管后赋值）
        "conversation_id": t.conversation_id,  # 关联的对话会话 ID，可追溯完整对话记录
        "created_at": t.created_at.isoformat(),  # 工单创建时间，ISO 8601 格式字符串
    }


@router.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest, user: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    """提交用户对 AI 回答的反馈，支持满意度记录与一键转人工两种场景。

    反馈类型（body.type）说明：
      - "resolved"   ：用户认为 AI 已解决问题，记录满意度并标记会话为 AI 解决
      - "unresolved" ：用户认为 AI 未解决问题，记录满意度（不满意）
      - "need_human" ：用户请求转人工，自动创建工单并将最近一条用户消息作为工单标题

    参数：
        body (FeedbackRequest): 请求体，含 conversation_id/message_id/type 字段。
        user (CurrentUser): 由 get_current_user 依赖注入，用于租户权限校验。
        db   (AsyncSession): 由 get_db 依赖注入的异步数据库会话。

    返回：
        dict: {"ok": True, "ticket_id": <工单ID 或 None>}
              - ticket_id 仅在 type="need_human" 时有值，其他情况为 None。

    异常：
        HTTPException(404): 指定 conversation_id 的会话不存在时抛出。
        HTTPException(403): 当前用户无权访问该会话所属租户时由 assert_tenant_access 抛出。
    """
    # 查询目标会话记录，确保会话存在再进行后续操作
    conv = (
        await db.execute(select(Conversation).where(Conversation.id == body.conversation_id))
    ).scalar_one_or_none()
    if conv is None:
        # 会话不存在时返回 404，防止对无效会话提交反馈
        raise HTTPException(404, "会话不存在")
    # 校验当前用户是否有权访问该会话所属租户，防止跨租户提交反馈
    assert_tenant_access(user, conv.tenant_id)
    # 创建 Feedback 记录并添加到数据库会话（尚未提交，将在函数末尾统一 commit）
    db.add(Feedback(conversation_id=body.conversation_id, message_id=body.message_id,
                    tenant_id=conv.tenant_id, type=body.type))
    # 反馈分三类：resolved/unresolved 记满意度，need_human 触发转人工建单
    if body.type in ("resolved", "unresolved"):
        # 更新会话的满意度字段，记录用户最终评价
        conv.satisfaction = body.type
        # resolved 表示 AI 成功解决，需在会话上标记 resolved_by_ai=True 用于指标统计
        conv.resolved_by_ai = body.type == "resolved"

    ticket_id = None  # 默认无工单；仅 need_human 场景下才会赋值
    if body.type == "need_human":
        # 将会话标记为已转人工，后续 AI 不再主动回复该会话
        conv.transferred_to_human = True
        # 一键转人工：真实创建工单，附最近一条用户问题作为上下文
        # 查询该会话中最近一条用户消息，用于自动生成工单标题
        last_user = (
            await db.execute(
                select(Message)
                .where(Message.conversation_id == conv.id, Message.role == "user")
                .order_by(desc(Message.created_at))  # 按创建时间降序取最新一条
            )
        ).scalars().first()
        # 取最近用户消息的前 60 字符作为工单标题；若无消息则使用默认占位标题
        title = (last_user.content[:60] if last_user else "用户请求人工支持")
        # 生成工单 ID：格式为 "tk_<日期>_<6位随机串>"，确保唯一性和可读性
        ticket_id = "tk_" + datetime(2026, 6, 15).strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:6]
        # 创建工单记录并添加到数据库会话（尚未提交）
        db.add(Ticket(
            ticket_id=ticket_id,  # 工单唯一标识
            tenant_id=conv.tenant_id,  # 继承自关联会话的租户 ID，保持数据一致性
            user_id=user.user_id,  # 提交反馈的用户 ID，即工单发起人
            category="人工支持",  # 固定分类：一键转人工创建的工单统一归类为"人工支持"
            priority="P2",  # 默认优先级 P2；后续可由运营人员手动调整
            status="new",  # 工单初始状态为 new，等待人工分配
            title=title,  # 工单标题（来自最近用户消息或默认文本）
            summary=title,  # 摘要与标题相同，作为问题简述（后续人工可补充）
            ai_diagnosis="用户在会话中主动请求人工支持",  # AI 诊断说明：标记为用户主动转人工，区分于 AI 识别后自动转人工的场景
            conversation_id=conv.id,  # 关联原始会话 ID，便于人工处理时查看完整对话记录
        ))
    # 统一提交数据库事务：包含 Feedback 记录、会话状态更新、以及可能的工单创建
    await db.commit()
    return {"ok": True, "ticket_id": ticket_id}  # 返回操作结果；ticket_id 仅转人工场景有值
