# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""技术支持工作台接口（内部角色）。

本模块提供 DevSupport-AI 项目的工作台管理 API，仅对内部支持人员（is_internal=True）开放。
主要功能：
  1. 工单列表查询（支持多维过滤：租户、状态、优先级、错误码、分类）。
  2. 工单详情查看（含关联会话消息上下文）。
  3. 工单状态/负责人更新（变更记录写审计日志）。
  4. AI 推荐回复话术生成（基于会话上下文，供人工编辑后使用）。
  5. 人工回复客户（写入会话消息，同时记录审计日志）。
  6. 接管会话（将会话标记为人工模式，后续消息不再走 AI）。
所有接口均挂载在 /api/workbench 前缀下，受 require_internal 权限守卫保护。
"""

import uuid                    # UUID 生成工具，用于为人工回复消息分配唯一 ID
from datetime import datetime  # 日期时间工具，用于更新工单的 updated_at 字段

from fastapi import APIRouter, Body, Depends, HTTPException  # FastAPI 核心组件
from sqlalchemy import select                               # SQLAlchemy SELECT 查询构造器
from sqlalchemy.ext.asyncio import AsyncSession             # 异步数据库会话类型

from app.db import get_db                          # FastAPI 依赖：获取异步数据库会话
from app.deps import CurrentUser, require_internal # CurrentUser 类型 + 内部角色权限守卫
from app.guardrail import desensitize              # 数据脱敏工具，防止敏感信息泄露给客户
from app.llm import client                         # LLM 客户端，用于调用语言模型生成话术
from app.llm.router import model_for               # 按任务类型选择合适模型的路由函数
from app.models import AuditLog, Conversation, Message, Ticket  # ORM 模型：审计日志、会话、消息、工单
from app.schemas.chat import TicketUpdateRequest   # 工单更新请求体的 Pydantic 模型

# 创建路由器，所有接口挂载在 /api/workbench 前缀下，属于 "workbench" 标签组
router = APIRouter(prefix="/api/workbench", tags=["workbench"])

# 工单状态流转
# 定义合法的工单状态集合，用于在更新接口中校验输入，防止非法状态写入
VALID_STATUS = {"new", "processing", "waiting_customer", "resolved", "closed", "escalated"}


@router.get("/tickets")
async def list_tickets(
    tenant_id: str | None = None,     # 可选过滤：按租户 ID 筛选（多租户隔离场景）
    status: str | None = None,        # 可选过滤：按工单状态筛选（如 new/resolved 等）
    priority: str | None = None,      # 可选过滤：按优先级筛选（如 high/medium/low）
    error_code: str | None = None,    # 可选过滤：按错误码筛选（快速定位同类故障）
    category: str | None = None,      # 可选过滤：按问题分类筛选（如账户/账单/技术）
    user: CurrentUser = Depends(require_internal),  # 仅内部支持人员可访问
    db: AsyncSession = Depends(get_db),             # 注入异步数据库会话
) -> dict:
    """查询工单列表，支持多维过滤条件，按创建时间倒序返回最多 100 条记录。

    Args:
        tenant_id:  按租户 ID 过滤（不传则查全部租户工单）。
        status:     按工单状态过滤。
        priority:   按优先级过滤。
        error_code: 按错误码过滤。
        category:   按问题分类过滤。
        user:       已认证的内部用户（由 require_internal 守卫确保）。
        db:         异步数据库会话。

    Returns:
        包含 tickets 列表的字典，每条记录含工单核心字段。
    """
    # 基础查询：按创建时间倒序，最多返回 100 条，避免超大结果集
    stmt = select(Ticket).order_by(Ticket.created_at.desc()).limit(100)
    if tenant_id:
        stmt = stmt.where(Ticket.tenant_id == tenant_id)  # 追加租户过滤条件
    if status:
        stmt = stmt.where(Ticket.status == status)        # 追加状态过滤条件
    if priority:
        stmt = stmt.where(Ticket.priority == priority)    # 追加优先级过滤条件
    if error_code:
        stmt = stmt.where(Ticket.error_code == error_code)  # 追加错误码过滤条件
    if category:
        stmt = stmt.where(Ticket.category == category)    # 追加分类过滤条件
    rows = (await db.execute(stmt)).scalars().all()  # 执行查询，获取所有匹配工单的 ORM 列表
    return {
        "tickets": [
            # 将 ORM 对象转换为字典，只返回前端列表展示所需的核心字段，不暴露内部详情
            {"ticket_id": t.ticket_id, "tenant_id": t.tenant_id, "title": t.title,
             "category": t.category, "priority": t.priority, "status": t.status,
             "error_code": t.error_code, "assignee": t.assignee,
             "created_at": t.created_at.isoformat()}  # 日期转 ISO 8601 字符串，方便前端解析
            for t in rows
        ]
    }


@router.get("/tickets/{ticket_id}")
async def ticket_with_context(
    ticket_id: str,                                # 工单 ID，作为 URL 路径参数
    user: CurrentUser = Depends(require_internal), # 仅内部支持人员可访问
    db: AsyncSession = Depends(get_db)             # 注入异步数据库会话
) -> dict:
    """获取工单详情，同时返回关联会话的完整消息上下文，便于支持人员全面了解问题背景。

    Args:
        ticket_id: 目标工单的唯一 ID。
        user:      已认证的内部用户。
        db:        异步数据库会话。

    Returns:
        包含 ticket（工单详情字典）和 conversation_messages（消息列表）的字典。

    Raises:
        HTTPException(404): 工单不存在时抛出。
    """
    # 按主键查找工单，scalar_one_or_none 在无结果时返回 None 而非抛出异常
    t = (await db.execute(select(Ticket).where(Ticket.ticket_id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, "工单不存在")  # 工单不存在时返回 404，明确错误语义
    messages = []  # 默认消息列表为空，如果工单没有关联会话则直接返回空列表
    if t.conversation_id:
        # 工单关联了会话：查询该会话下的所有消息，按创建时间正序排列（还原对话顺序）
        msgs = (
            await db.execute(
                select(Message).where(Message.conversation_id == t.conversation_id).order_by(Message.created_at)
            )
        ).scalars().all()
        # 将 ORM 消息对象转换为字典列表，包含角色、内容和元数据
        messages = [{"role": m.role, "content": m.content, "meta": m.meta} for m in msgs]
    return {
        "ticket": {
            # 返回工单的完整字段，包括 AI 诊断、证据、关联请求 ID 等深度信息
            "ticket_id": t.ticket_id, "tenant_id": t.tenant_id, "title": t.title,
            "category": t.category, "priority": t.priority, "status": t.status,
            "summary": t.summary,                          # AI 生成的问题摘要
            "ai_diagnosis": t.ai_diagnosis,                # AI 诊断结论
            "evidence": t.evidence,                        # 支撑诊断的证据数据
            "related_request_ids": t.related_request_ids, # 关联的请求/日志 ID 列表
            "error_code": t.error_code,                    # 触发问题的错误码
            "assignee": t.assignee,                        # 当前负责人
            "conversation_id": t.conversation_id,          # 关联会话 ID
        },
        "conversation_messages": messages,  # 会话消息上下文列表
    }


@router.post("/tickets/{ticket_id}")
async def update_ticket(
    ticket_id: str,                                # 工单 ID，作为 URL 路径参数
    body: TicketUpdateRequest,                     # 请求体：包含可选的 status、assignee、note 字段
    user: CurrentUser = Depends(require_internal), # 仅内部支持人员可访问
    db: AsyncSession = Depends(get_db),            # 注入异步数据库会话
) -> dict:
    """更新工单的状态和/或负责人，并向审计日志写入变更记录。

    Args:
        ticket_id: 目标工单的唯一 ID。
        body:      更新请求体，status/assignee/note 均为可选字段。
        user:      已认证的内部用户，其 user_id 将写入审计日志。
        db:        异步数据库会话。

    Returns:
        包含 ok（成功标志）、ticket_id、status、assignee 的字典。

    Raises:
        HTTPException(404): 工单不存在。
        HTTPException(400): 提供了非法的工单状态值。
    """
    # 查找目标工单，确保其存在
    t = (await db.execute(select(Ticket).where(Ticket.ticket_id == ticket_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(404, "工单不存在")  # 工单不存在返回 404
    if body.status:
        if body.status not in VALID_STATUS:
            # 非法状态值：返回 400 并告知具体错误，防止状态机出现未定义状态
            raise HTTPException(400, f"非法状态: {body.status}")
        t.status = body.status  # 更新工单状态字段
    if body.assignee is not None:
        t.assignee = body.assignee  # 更新负责人（允许设置为空字符串表示取消分配）
    t.updated_at = datetime.utcnow()  # 刷新最后更新时间戳，使用 UTC 时间保证时区一致性
    # 审计
    # 写入审计日志：记录操作人、操作类型和变更详情，满足合规要求、便于事后追溯
    db.add(AuditLog(tenant_id=t.tenant_id, user_id=user.user_id, action="update_ticket",
                    detail=f"ticket={ticket_id} status={body.status} assignee={body.assignee} note={body.note}"))
    await db.commit()  # 原子提交工单变更和审计日志，保证两者同时写入或同时失败
    return {"ok": True, "ticket_id": ticket_id, "status": t.status, "assignee": t.assignee}


async def _conv_messages(db, conv_id):
    """查询指定会话下的所有消息，按创建时间正序排列。

    Args:
        db:      异步数据库会话。
        conv_id: 目标会话的唯一 ID。

    Returns:
        Message ORM 对象列表，按 created_at 升序排序（还原对话时序）。
    """
    return (
        await db.execute(
            # 过滤指定会话 ID，并按消息创建时间升序排列，保证对话顺序正确
            select(Message).where(Message.conversation_id == conv_id).order_by(Message.created_at)
        )
    ).scalars().all()  # scalars() 取出 ORM 对象而非行元组，all() 返回完整列表


@router.get("/conversations/{conv_id}/suggest_reply")
async def suggest_reply(
    conv_id: str,                                  # 会话 ID，作为 URL 路径参数
    user: CurrentUser = Depends(require_internal), # 仅内部支持人员可访问
    db: AsyncSession = Depends(get_db)             # 注入异步数据库会话
) -> dict:
    """基于会话上下文生成 AI 推荐回复话术，供人工编辑后发送。

    调用 LLM 根据最近 8 条对话内容，生成一段专业、礼貌、可直接发送给客户的回复草稿。
    生成结果经脱敏处理后返回，避免敏感信息意外包含在话术中。

    Args:
        conv_id: 目标会话的唯一 ID。
        user:    已认证的内部用户。
        db:      异步数据库会话。

    Returns:
        包含 suggestion（推荐话术文本）的字典。

    Raises:
        HTTPException(404): 会话不存在。
    """
    # 查找目标会话，确保其存在
    conv = (await db.execute(select(Conversation).where(Conversation.id == conv_id))).scalar_one_or_none()
    if conv is None:
        raise HTTPException(404, "会话不存在")  # 会话不存在返回 404
    msgs = await _conv_messages(db, conv_id)  # 获取会话的全部历史消息（按时间正序）
    # 仅取最近 8 条作为话术生成上下文，控制 token 并聚焦近期诉求
    # 将消息列表拼接为"角色: 内容"格式的对话文本，便于 LLM 理解对话结构
    convo = "\n".join(f"{'客户' if m.role == 'user' else '助手'}: {m.content}" for m in msgs[-8:])
    gen = await client.chat(
        [
            # 系统提示：定义 LLM 的角色和任务，约束输出格式（只输出话术正文，不要多余解释）
            {"role": "system", "content": "你是资深技术支持。基于会话上下文，为客服起草一段专业、礼貌、可直接发送给客户的回复话术，给出明确结论与下一步。只输出话术正文。"},
            # 用户消息：提供实际对话内容作为生成上下文；若无历史消息则提供占位提示
            {"role": "user", "content": convo or "（暂无对话内容）"},
        ],
        model=model_for("summarize"),  # 使用摘要/生成场景对应的模型（通常为轻量级模型以控制成本）
        temperature=0.3,               # 较低温度保证话术输出稳定、专业，减少随机发散
    )
    # 对生成结果进行脱敏处理，确保话术中不包含手机号、邮箱等敏感信息
    return {"suggestion": desensitize.desensitize_text(gen.content.strip())}


@router.post("/conversations/{conv_id}/reply")
async def human_reply(
    conv_id: str,                                       # 会话 ID，作为 URL 路径参数
    content: str = Body(..., embed=True),               # 请求体：人工回复的消息内容（必填）
    user: CurrentUser = Depends(require_internal),      # 仅内部支持人员可访问
    db: AsyncSession = Depends(get_db),                 # 注入异步数据库会话
) -> dict:
    """人工回复客户：写入会话（客户可在会话历史看到），并脱敏。

    将支持人员撰写的回复消息以 "assistant" 角色写入会话消息表，
    同时在元数据中记录操作人信息，并写入审计日志。
    消息内容经脱敏处理，防止支持人员无意中在回复中包含敏感信息。

    Args:
        conv_id: 目标会话的唯一 ID。
        content: 人工回复的消息文本（必填，通过 Body embed 模式传入）。
        user:    已认证的内部用户，其 user_id 和 display_name 将写入消息元数据。
        db:      异步数据库会话。

    Returns:
        包含 ok（成功标志）和 message_id（新消息 ID）的字典。

    Raises:
        HTTPException(404): 会话不存在。
    """
    # 查找目标会话，确保其存在
    conv = (await db.execute(select(Conversation).where(Conversation.id == conv_id))).scalar_one_or_none()
    if conv is None:
        raise HTTPException(404, "会话不存在")  # 会话不存在返回 404
    # 构建人工回复消息对象，角色为 assistant（统一与 AI 回复使用相同角色标记，方便前端渲染）
    msg = Message(
        id="msg_" + uuid.uuid4().hex[:12],                      # 为消息分配唯一 ID（12 位十六进制）
        conversation_id=conv_id,                                 # 关联到目标会话
        role="assistant",                                        # 角色为助手（代表支持方的回复）
        content=desensitize.desensitize_text(content),           # 对回复内容脱敏，防止敏感信息外泄
        meta={"by": "human", "agent_id": user.user_id, "agent_name": user.display_name},  # 记录操作人信息
    )
    db.add(msg)                            # 将消息加入待提交队列
    conv.transferred_to_human = True       # 确保会话保持人工接管状态（即使之前未标记也补充标记）
    # 写入审计日志：记录谁对哪个会话执行了人工回复操作，满足合规要求
    db.add(AuditLog(tenant_id=conv.tenant_id, user_id=user.user_id, action="human_reply",
                    detail=f"conversation={conv_id}"))
    await db.commit()  # 原子提交消息记录、会话状态更新和审计日志
    return {"ok": True, "message_id": msg.id}  # 返回成功标志和新消息 ID，供前端更新 UI


@router.post("/conversations/{conv_id}/takeover")
async def takeover_conversation(
    conv_id: str,                                  # 会话 ID，作为 URL 路径参数
    user: CurrentUser = Depends(require_internal), # 仅内部支持人员可访问
    db: AsyncSession = Depends(get_db)             # 注入异步数据库会话
) -> dict:
    """接管指定会话：将会话标记为人工模式，后续客户消息不再由 AI 自动回复。

    支持人员主动接管后，客户端的 chat 接口会检测到 transferred_to_human=True，
    从而跳过 AI 编排，直接落库客户消息并提示等待人工处理。

    Args:
        conv_id: 目标会话的唯一 ID。
        user:    已认证的内部用户，其 user_id 将写入审计日志并作为负责人返回。
        db:      异步数据库会话。

    Returns:
        包含 ok（成功标志）、conversation_id 和 assignee（接管人 ID）的字典。

    Raises:
        HTTPException(404): 会话不存在。
    """
    # 查找目标会话，确保其存在
    conv = (await db.execute(select(Conversation).where(Conversation.id == conv_id))).scalar_one_or_none()
    if conv is None:
        raise HTTPException(404, "会话不存在")  # 会话不存在返回 404
    conv.transferred_to_human = True  # 标记会话为人工接管状态，触发 chat 接口的人工模式分支
    # 写入审计日志：记录接管操作，便于追踪会话的人工介入历史
    db.add(AuditLog(tenant_id=conv.tenant_id, user_id=user.user_id, action="takeover",
                    detail=f"conversation={conv_id}"))
    await db.commit()  # 原子提交会话状态变更和审计日志
    return {"ok": True, "conversation_id": conv_id, "assignee": user.user_id}  # 返回成功信息及接管人 ID
