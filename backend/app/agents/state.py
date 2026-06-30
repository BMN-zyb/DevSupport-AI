# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""LangGraph 编排状态。

本文件定义了 AgentState —— DevSupport-AI 系统中 LangGraph 有向图
在每个节点间共享传递的统一状态结构。

AgentState 使用 TypedDict 描述，字段覆盖完整的请求生命周期：
  - 输入与上下文：租户/用户/会话信息、原始查询、历史对话
  - 意图识别阶段：意图类型、置信度、实体、路由、是否需要追问
  - 处理结果阶段：各专业 Agent 输出、RAG 引用、草稿/最终回复、结构化卡片
  - 工单与人工转接：建单状态、工单 ID、工单提示
  - 可观测性：链路追踪 ID、token 消耗统计
"""

from typing import Any, TypedDict  # Any 用于动态类型字段；TypedDict 构建结构化状态


class AgentState(TypedDict, total=False):
    """LangGraph 图节点间共享的状态字典类型。

    total=False 表示所有字段均为可选（optional），节点只需返回自己修改的字段，
    LangGraph 会自动将返回的部分字典合并到全局状态中。

    各字段按处理阶段分组，反映管线中数据的生命周期。
    """

    # ----------------------------------------------------------------
    # 输入与上下文
    # ----------------------------------------------------------------
    tenant_id: str          # 租户 ID，用于数据隔离、权限控制和成本归因
    user_id: str            # 用户 ID，用于工单关联和审计日志
    conversation_id: str    # 会话 ID，用于多轮对话记忆的唯一标识
    is_internal: bool       # 是否为平台内部员工调用，影响工具调用权限范围
    query: str              # 用户本轮输入的原始问题文本
    history: list[dict]              # 历史消息 [{role, content}]
    collected_entities: dict         # 记忆中已收集的实体

    # ----------------------------------------------------------------
    # 意图识别
    # ----------------------------------------------------------------
    intent: str             # 意图分类结果（如 doc_qa/api_error/billing/rate_limit 等）
    confidence: float       # 意图分类的置信度（0.0 ~ 1.0），低于阈值时触发澄清追问
    entities: dict          # 本轮从查询中抽取并与历史记忆合并后的完整实体字典
    route: list[str]                 # 选中的专业 Agent
    need_clarify: bool      # 是否需要向用户追问以获取更多信息（True 时进入 clarify 节点）
    clarify_question: str   # 追问的具体文本内容，need_clarify=True 时有值

    # ----------------------------------------------------------------
    # 处理结果
    # ----------------------------------------------------------------
    agent_outputs: dict[str, Any]    # 各专业 Agent 输出
    rag_citations: list[dict]        # 所有专业 Agent 返回的文档引用列表（前端用于展示来源）
    draft_answer: str       # summarize 节点生成的初稿文本，待安全审查后变为 final_answer
    final_answer: str       # security 节点脱敏处理后的最终回复文本，直接返回给用户
    card: dict | None       # 结构化回答卡片（含 conclusion/evidence/steps），可为 None
    need_human: bool        # 是否需要转人工（专业 Agent 判断或明确 ticket 意图时为 True）
    pending_ticket: bool        # 是否需要建单（诊断证据不足等）
    ticket_id: str          # 工单创建成功后的工单 ID，透传给用户
    ticket_message: str         # 建单后给客户的友好提示

    # ----------------------------------------------------------------
    # 可观测
    # ----------------------------------------------------------------
    trace_id: str           # 链路追踪唯一 ID，贯穿整个请求生命周期，用于日志关联和排查
    total_tokens: int       # 本次请求所有 LLM 调用累计消耗的 token 总数，用于成本统计
