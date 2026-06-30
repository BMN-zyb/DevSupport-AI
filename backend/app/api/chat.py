# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""智能对话接口（SSE 流式）。

流程：落库会话/用户消息 → 运行多 Agent 编排 → 落库助手消息 → SSE 流式返回答案 + 元信息。

本模块是 DevSupport-AI 项目的核心聊天接口路由，提供 POST /api/chat 端点。
支持两种场景：
  1. AI 模式：消息经多 Agent 编排处理后，以 SSE（Server-Sent Events）流式返回答案、引用、工单等信息。
  2. 人工模式：会话已转人工接管时，客户消息直接落库并返回人工等待提示，不再经过 AI。
"""

import asyncio  # 异步并发库，用于 SSE 流式推送时的非阻塞延迟
import json     # JSON 序列化/反序列化，用于构造 SSE data 字段的结构化载荷
import uuid     # UUID 生成工具，用于为会话和消息创建唯一 ID

from fastapi import APIRouter, Depends                  # FastAPI 路由器及依赖注入
from sqlalchemy import select                           # SQLAlchemy 的 SELECT 查询构造器
from sqlalchemy.ext.asyncio import AsyncSession         # 异步数据库会话类型
from sse_starlette.sse import EventSourceResponse       # SSE（Server-Sent Events）响应封装

from app.agents import supervisor      # 多 Agent 编排调度器，负责路由问题到各子 Agent 并汇总结果
from app.db import get_db              # FastAPI 依赖：获取异步数据库会话
from app.deps import CurrentUser, get_current_user  # 当前用户类型及身份验证依赖
from app.models import Conversation, Message        # ORM 模型：会话表和消息表
from app.schemas.chat import ChatRequest            # 请求体 Pydantic 模型

# 创建路由器，所有接口挂载在 /api 前缀下，属于 "chat" 标签组
router = APIRouter(prefix="/api", tags=["chat"])


def _gen(prefix: str) -> str:
    """生成带前缀的唯一 ID 字符串。

    Args:
        prefix: ID 前缀，例如 "conv" 或 "msg"，用于在数据库中区分不同类型的记录。

    Returns:
        形如 "conv_3a7f9d2c1b4e" 的字符串，前缀 + 12 位十六进制 UUID 片段。
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"  # 取 UUID 的前 12 位十六进制，足够唯一且不过长


async def _get_or_create_conversation(
    db: AsyncSession, conv_id: str | None, user: CurrentUser
) -> Conversation:
    """获取已有会话或按需创建新会话。

    若调用方传入了 conv_id，则优先复用该会话（同时校验租户权限，防止跨租户越权）。
    若未传入 conv_id 或未找到匹配会话，则自动创建一个新会话并写库。

    Args:
        db:      异步数据库会话，用于执行查询和提交事务。
        conv_id: 前端传入的会话 ID（可为 None，表示开启新会话）。
        user:    已认证的当前用户，提供 tenant_id、user_id 和 is_internal 信息。

    Returns:
        找到或新建的 Conversation ORM 实例。
    """
    if conv_id:
        # 尝试通过主键查找已有会话
        conv = (
            await db.execute(select(Conversation).where(Conversation.id == conv_id))
        ).scalar_one_or_none()
        # 复用已有会话前校验租户归属，防止越权访问他人会话
        if conv and (user.is_internal or conv.tenant_id == user.tenant_id):
            return conv  # 校验通过，直接返回已有会话，避免重复创建
    # 未找到可复用的会话，或 conv_id 为空：创建一条新的会话记录
    conv = Conversation(id=_gen("conv"), tenant_id=user.tenant_id, user_id=user.user_id,
                        channel="web", status="active")  # channel 固定为 web，status 初始为 active
    db.add(conv)           # 将新会话对象加入数据库会话的待提交队列
    await db.commit()      # 提交事务，将会话持久化到数据库
    return conv            # 返回刚创建的新会话


@router.post("/chat")
async def chat(
    body: ChatRequest,                              # 请求体，包含 message 和可选的 conversation_id
    user: CurrentUser = Depends(get_current_user),  # 通过 JWT 或 API Key 认证的当前用户
    db: AsyncSession = Depends(get_db),             # 注入异步数据库会话
):
    """处理用户发送的聊天消息，以 SSE 流式返回 AI（或人工提示）答案。

    完整流程：
      1. 获取或创建会话记录。
      2. 判断会话是否已转人工：若是，直接落库用户消息并返回人工等待 SSE 流。
      3. 否则，落库用户消息，调用多 Agent 编排（supervisor.run）获取 AI 答案。
      4. 落库助手消息，更新会话状态。
      5. 以 SSE 格式流式返回 meta（元信息）→ token（逐块答案）→ done（完整结构）。

    Args:
        body: ChatRequest，包含 message（用户消息文本）和 conversation_id（可选会话 ID）。
        user: 已认证用户，含 is_internal（是否内部支持人员）、tenant_id、user_id。
        db:   异步数据库会话。

    Returns:
        EventSourceResponse：SSE 流式响应，依次推送 meta、token、done 三类事件。
    """
    # 获取或创建当前会话（conv_id 为空时自动新建）
    conv = await _get_or_create_conversation(db, body.conversation_id, user)

    # 人工模式：会话已转人工 → 客户消息不再走 AI，仅追加并提示等待人工
    if conv.transferred_to_human and not user.is_internal:
        # 将客户消息持久化到消息表，即使不走 AI 也要保留完整对话记录
        cust_msg = Message(id=_gen("msg"), conversation_id=conv.id, role="user", content=body.message)
        db.add(cust_msg)   # 加入待提交队列
        await db.commit()  # 立即落库，确保消息不丢失
        # 人工等待提示文案，语义友好地告知客户消息已转达
        ack = "您的消息已转达人工技术支持，我们会尽快回复，可在「我的会话」查看进展。"

        async def human_stream():
            """人工模式下的 SSE 事件生成器，模拟流式发送等待提示。"""
            # 先推送 meta 事件，告知前端会话/消息 ID 以及当前为人工接管状态
            yield {"event": "meta", "data": json.dumps(
                {"conversation_id": conv.id, "message_id": cust_msg.id, "intent": "human", "trace_id": None},
                ensure_ascii=False)}  # ensure_ascii=False 保证中文字符不被转义
            # 按每 12 个字符分块模拟打字机效果推送提示文案
            for i in range(0, len(ack), 12):
                yield {"event": "token", "data": ack[i:i + 12]}  # 逐块发送文本片段
                await asyncio.sleep(0.02)  # 每块间隔 20ms，产生可感知的打字机视觉效果
            # 推送 done 事件，标记流结束，并附带人工模式标志供前端判断
            yield {"event": "done", "data": json.dumps({"human_mode": True}, ensure_ascii=False)}

        return EventSourceResponse(human_stream())  # 将异步生成器包装为 SSE 响应返回

    # 落库用户消息（AI 模式）：在调用 AI 之前先持久化，保证即使 AI 调用失败消息也不丢失
    user_msg = Message(id=_gen("msg"), conversation_id=conv.id, role="user", content=body.message)
    db.add(user_msg)   # 将用户消息对象加入数据库会话
    await db.commit()  # 提交事务，完成持久化

    # 运行编排：将用户问题交给多 Agent 调度器处理，获取结构化结果
    result = await supervisor.run(
        query=body.message,          # 用户原始问题文本
        tenant_id=conv.tenant_id,    # 租户 ID，用于隔离各租户的知识库/配置
        user_id=user.user_id,        # 用户 ID，用于个性化上下文或行为追踪
        conversation_id=conv.id,     # 会话 ID，便于 Agent 获取历史消息作为上下文
        is_internal=user.is_internal,  # 是否内部人员，影响部分 Agent 的权限与输出
    )

    # 落库助手消息（含诊断元信息）：将 AI 生成的答案及附属信息持久化
    assistant_msg = Message(
        id=_gen("msg"),                 # 为助手消息分配唯一 ID
        conversation_id=conv.id,        # 关联到当前会话
        role="assistant",               # 角色标记为助手（区别于用户消息）
        content=result["answer"],       # AI 生成的主要答案文本
        meta={
            "intent": result.get("intent"),              # 识别出的用户意图分类
            "citations": result.get("citations", []),    # 知识库引用来源列表
            "card": result.get("card"),                  # 结构化卡片数据（如工单摘要卡片）
            "trace_id": result.get("trace_id"),          # 可观测性追踪 ID，便于日志关联
            "ticket_id": result.get("ticket_id"),        # 若创建了工单，此处为工单 ID
            "need_human": result.get("need_human"),      # AI 是否判断需要转人工处理
            "from_cache": result.get("from_cache", False),  # 答案是否来自语义缓存
        },
    )
    db.add(assistant_msg)   # 将助手消息加入待提交队列
    # 更新会话状态
    conv.latest_intent = result.get("intent")  # 记录最新意图，便于会话列表快速展示
    if result.get("need_human"):
        conv.transferred_to_human = True  # AI 判断需转人工时，标记会话为人工接管状态
    await db.commit()  # 一次性提交助手消息和会话状态变更，保证原子性

    async def event_stream():
        """AI 模式下的 SSE 事件生成器，依次推送 meta、token（答案分块）、done 事件。"""
        # 先发会话/消息元信息，前端可据此更新 UI 状态（如显示会话 ID、追踪链接等）
        yield {"event": "meta", "data": json.dumps(
            {"conversation_id": conv.id, "message_id": assistant_msg.id,
             "intent": result.get("intent"), "trace_id": result.get("trace_id")},
            ensure_ascii=False)}  # ensure_ascii=False 允许 JSON 中包含中文字符
        # 流式发送答案（按字符块模拟打字机）
        answer = result["answer"]  # 取出完整答案文本
        chunk = 18                 # 每块推送 18 个字符，兼顾流畅度与推送频率
        for i in range(0, len(answer), chunk):
            yield {"event": "token", "data": answer[i:i + chunk]}  # 推送当前块的文本片段
            await asyncio.sleep(0.02)  # 每块间隔 20ms，产生打字机视觉效果
        # 末尾发完整结构化信息，前端在 done 事件中获取所有附属数据（卡片、引用、工单等）
        yield {"event": "done", "data": json.dumps(
            {
                "answer": answer,                                    # 完整答案文本（方便前端二次使用）
                "card": result.get("card"),                          # 结构化卡片（如故障诊断卡片）
                "citations": result.get("citations", []),            # 知识库引用来源
                "ticket_id": result.get("ticket_id"),                # 关联工单 ID（若有）
                "need_human": result.get("need_human", False),       # 是否需要转人工
                "need_clarify": result.get("need_clarify", False),   # 是否需要用户补充信息
                "from_cache": result.get("from_cache", False),       # 是否命中语义缓存
                "trace_id": result.get("trace_id"),                  # 可观测性追踪 ID
            }, ensure_ascii=False)}  # ensure_ascii=False 保证中文内容正常编码

    return EventSourceResponse(event_stream())  # 将异步生成器包装为 SSE 响应返回给客户端
