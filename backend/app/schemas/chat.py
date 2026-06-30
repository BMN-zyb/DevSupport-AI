# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""对话/工单相关 DTO（Data Transfer Object，数据传输对象）。

本模块定义了用户对话、反馈和工单操作涉及的请求数据结构，使用 Pydantic BaseModel
提供自动的数据验证、序列化和 OpenAPI 文档生成能力。包含以下三个 schema：
  - ChatRequest：对话消息请求体，携带用户输入和会话 ID。
  - FeedbackRequest：对话反馈请求体，记录用户对答案质量的评价。
  - TicketUpdateRequest：工单更新请求体，用于修改工单状态、分配人或添加备注。
"""

# 第三方库：pydantic 提供基于类型注解的数据验证框架，FastAPI 用它处理请求/响应的数据校验
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """对话消息请求体 DTO。

    用户发送消息时携带此结构。conversation_id 用于关联多轮对话上下文，
    首次对话时可不传（由服务端生成），后续轮次必须传入以维持会话连续性。
    """
    message: str                       # 用户输入的消息文本，不允许为空
    conversation_id: str | None = None  # 会话 ID，None 表示开启新会话；非 None 时继续已有会话


class FeedbackRequest(BaseModel):
    """对话反馈请求体 DTO。

    用户对 AI 回答进行评价时提交此结构，支持三种反馈类型：
      - "resolved"：问题已解决，答案满意。
      - "unresolved"：问题未解决，答案不满意。
      - "need_human"：需要转接人工客服处理。
    反馈数据可用于模型效果评估和人工介入触发。
    """
    conversation_id: str          # 被评价的会话 ID，用于关联具体对话记录
    message_id: str | None = None  # 被评价的具体消息 ID（可选），None 表示对整个会话评价
    type: str                     # 反馈类型：resolved / unresolved / need_human


class TicketUpdateRequest(BaseModel):
    """工单更新请求体 DTO。

    客服或管理员更新工单状态时携带此结构，所有字段均为可选，
    仅传入需要修改的字段，未传入的字段保持原值（PATCH 语义）。
    """
    status: str | None = None    # 工单状态，如 "open"、"in_progress"、"closed"；None 表示不修改
    assignee: str | None = None  # 工单负责人（用户名或 ID）；None 表示不修改分配
    note: str | None = None      # 工单备注或处理说明；None 表示不添加备注
