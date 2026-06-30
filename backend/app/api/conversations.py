# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""会话查询接口。

本模块提供 DevSupport-AI 项目的会话管理 API，挂载在 /api/conversations 前缀下。
面向内外部用户，提供以下功能：
  1. 会话列表查询（内部用户可见全部；外部客户仅见自己的会话）。
  2. 会话详情查询（含完整消息记录，含租户归属校验）。
  3. 客户在人工模式下补充消息（不走 AI，供「我的会话」页面回话使用）。
所有写操作均经脱敏处理，防止敏感信息写入数据库。
"""

import uuid  # UUID 生成工具，用于为客户补充消息分配唯一 ID

from fastapi import APIRouter, Body, Depends, HTTPException  # FastAPI 核心组件
from sqlalchemy import select                               # SQLAlchemy SELECT 查询构造器
from sqlalchemy.ext.asyncio import AsyncSession             # 异步数据库会话类型

from app.db import get_db                                          # FastAPI 依赖：获取异步数据库会话
from app.deps import CurrentUser, assert_tenant_access, get_current_user  # 用户类型、租户校验及身份认证依赖
from app.guardrail import desensitize                              # 数据脱敏工具，防止敏感信息落库
from app.models import Conversation, Message                       # ORM 模型：会话表和消息表

# 创建路由器，所有接口挂载在 /api/conversations 前缀下，属于 "conversations" 标签组
router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(
    user: CurrentUser = Depends(get_current_user), # 通过 JWT 或 API Key 认证的当前用户
    db: AsyncSession = Depends(get_db)             # 注入异步数据库会话
) -> dict:
    """查询会话列表，按最后更新时间倒序返回最多 50 条记录。

    权限逻辑：
      - 内部支持人员（is_internal=True）：可查看所有租户的全部会话。
      - 外部客户（is_internal=False）：仅可查看本租户下属于自己 user_id 的会话。

    Args:
        user: 已认证的当前用户，含 is_internal、tenant_id、user_id。
        db:   异步数据库会话。

    Returns:
        包含 conversations 列表的字典，每条记录含会话核心状态字段。
    """
    # 基础查询：按 updated_at 倒序（最近活跃的排在最前），最多返回 50 条
    stmt = select(Conversation).order_by(Conversation.updated_at.desc()).limit(50)
    # 内部支持人员可见全部会话；外部客户仅见本租户下自己的会话
    if not user.is_internal:
        # 外部客户：追加租户和用户双重过滤，严格隔离不同客户的会话数据
        stmt = stmt.where(Conversation.tenant_id == user.tenant_id, Conversation.user_id == user.user_id)
    rows = (await db.execute(stmt)).scalars().all()  # 执行查询，获取匹配会话的 ORM 列表
    return {
        "conversations": [
            # 将 ORM 对象转换为字典，只返回列表展示所需的关键字段
            {"id": c.id, "tenant_id": c.tenant_id, "status": c.status,
             "latest_intent": c.latest_intent,          # 最近识别的用户意图分类
             "transferred_to_human": c.transferred_to_human,  # 是否已转人工接管
             "satisfaction": c.satisfaction,             # 客户满意度评分（如有）
             "updated_at": c.updated_at.isoformat()}     # 最后更新时间转 ISO 8601 字符串
            for c in rows
        ]
    }


@router.get("/{conv_id}")
async def get_conversation(
    conv_id: str,                                  # 会话 ID，作为 URL 路径参数
    user: CurrentUser = Depends(get_current_user), # 已认证的当前用户
    db: AsyncSession = Depends(get_db)             # 注入异步数据库会话
) -> dict:
    """获取指定会话的详细信息，包含完整的消息记录列表。

    会在返回前通过 assert_tenant_access 校验当前用户是否有权访问该会话，
    防止越权查看他人会话（外部客户跨租户访问场景）。

    Args:
        conv_id: 目标会话的唯一 ID。
        user:    已认证的当前用户，用于租户归属校验。
        db:      异步数据库会话。

    Returns:
        包含 conversation（会话信息字典）和 messages（消息列表）的字典。

    Raises:
        HTTPException(404): 会话不存在。
        HTTPException(403): 无权访问该会话（由 assert_tenant_access 抛出）。
    """
    # 按主键查找会话，scalar_one_or_none 在无结果时返回 None 而非抛出异常
    conv = (
        await db.execute(select(Conversation).where(Conversation.id == conv_id))
    ).scalar_one_or_none()
    if conv is None:
        raise HTTPException(404, "会话不存在")  # 会话不存在时返回 404，明确错误语义
    assert_tenant_access(user, conv.tenant_id)  # 校验当前用户是否属于该会话所在租户，防止越权
    # 查询该会话下的所有消息，按创建时间升序排列，还原对话的时间顺序
    msgs = (
        await db.execute(
            select(Message).where(Message.conversation_id == conv_id).order_by(Message.created_at)
        )
    ).scalars().all()  # 获取完整消息列表
    return {
        "conversation": {"id": conv.id, "tenant_id": conv.tenant_id, "status": conv.status,
                         "latest_intent": conv.latest_intent,             # 最近识别的用户意图
                         "transferred_to_human": conv.transferred_to_human,  # 是否已转人工
                         "satisfaction": conv.satisfaction},              # 满意度评分
        "messages": [
            # 将消息 ORM 对象转换为字典，含消息 ID、角色、内容、元数据和创建时间
            {"id": m.id, "role": m.role, "content": m.content, "meta": m.meta,
             "created_at": m.created_at.isoformat()}  # 时间转 ISO 8601，前端方便解析
            for m in msgs
        ],
    }


@router.post("/{conv_id}/messages")
async def add_customer_message(
    conv_id: str,                                  # 会话 ID，作为 URL 路径参数
    content: str = Body(..., embed=True),          # 请求体：消息内容（必填，embed 模式）
    user: CurrentUser = Depends(get_current_user), # 已认证的当前用户
    db: AsyncSession = Depends(get_db),            # 注入异步数据库会话
) -> dict:
    """客户在人工模式下补充消息（不走 AI），供「我的会话」回话使用。

    适用场景：会话已被转入人工模式（transferred_to_human=True），客户从「我的会话」页面
    发送追加消息时调用。消息不经过 AI 编排，直接以 "user" 角色写入会话消息表，
    等待人工支持人员通过工作台的 human_reply 接口回复。
    消息内容经过脱敏处理，防止客户输入的敏感信息（如密码、银行卡号）被明文存储。

    Args:
        conv_id: 目标会话的唯一 ID。
        content: 客户补充消息的文本内容（必填）。
        user:    已认证的当前用户，用于租户归属校验。
        db:      异步数据库会话。

    Returns:
        包含 ok（成功标志）和 message_id（新消息 ID）的字典。

    Raises:
        HTTPException(404): 会话不存在。
        HTTPException(403): 无权访问该会话（由 assert_tenant_access 抛出）。
    """
    # 查找目标会话，确保其存在
    conv = (
        await db.execute(select(Conversation).where(Conversation.id == conv_id))
    ).scalar_one_or_none()
    if conv is None:
        raise HTTPException(404, "会话不存在")  # 会话不存在时返回 404
    assert_tenant_access(user, conv.tenant_id)  # 校验租户归属，防止越权写入他人会话
    # 构建客户补充消息对象，角色为 "user"（区别于助手/支持人员的 "assistant" 角色）
    msg = Message(
        id="msg_" + uuid.uuid4().hex[:12],              # 为消息分配唯一 ID（12 位十六进制）
        conversation_id=conv_id,                         # 关联到目标会话
        role="user",                                     # 角色为用户（客户发送的消息）
        content=desensitize.desensitize_text(content),   # 对消息内容进行脱敏处理后再存储
    )
    db.add(msg)        # 将消息对象加入数据库会话的待提交队列
    await db.commit()  # 提交事务，将消息持久化到数据库
    return {"ok": True, "message_id": msg.id}  # 返回成功标志和新消息 ID，供前端更新对话 UI
