# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Ticket Agent：兜底创建工单，附带 AI 诊断摘要与证据。

本模块是 DevSupport-AI 多智能体系统中的工单专职子智能体。
职责：
  1. 当其他专职子智能体（API 诊断、计费等）无法自动解决问题时，作为兜底手段创建人工工单。
  2. 根据意图类型自动映射工单类别（如"API报错"、"套餐账单"）和优先级（P1/P2/P3）。
  3. 将 AI 诊断结论、收集到的证据（日志/Key状态/限流数据）随工单一并落库，
     让人工支持团队接手时即可获得完整上下文，减少重复沟通。
  4. 关联 request_id（可选），便于工程师在后台追溯原始调用记录。
"""

# -----------------------------------------------------------------------
# 标准库导入
# -----------------------------------------------------------------------
import json  # 用于将证据字典序列化为 JSON 字符串，存入工单附加信息字段

# dataclass：定义轻量数据容器，存储工单创建结果
from dataclasses import dataclass

# -----------------------------------------------------------------------
# 项目内部模块导入
# -----------------------------------------------------------------------
from app.tools.registry import ToolContext, execute  # 工具注册中心：execute 统一调用各业务工具

# 意图 -> 工单类型 / 默认优先级
# 映射规则：不同用户意图对应不同的业务类别和服务响应优先级
# P1：最高优先级，影响正常使用的故障；P2：中等优先级，账单/数据问题；P3：最低优先级，咨询类
CATEGORY_MAP = {
    "api_error": ("API报错", "P1"),      # API 调用报错，影响业务，最高优先级
    "rate_limit": ("API报错", "P1"),     # 限流问题，影响调用频率，最高优先级
    "billing": ("套餐账单", "P2"),       # 计费/账单疑问，中等优先级
    "data_quality": ("数据质量", "P2"),  # 数据质量/准确性问题，中等优先级
    "ticket": ("故障投诉", "P2"),        # 用户主动投诉/反馈故障，中等优先级
    "doc_qa": ("咨询", "P3"),            # 文档咨询类问题，低优先级（AI 已无法回答）
}


@dataclass
class TicketResult:
    """工单创建结果数据容器，由 create_from_context() 函数填充并返回给上层编排器。

    Attributes:
        ticket_id: 成功创建的工单 ID（如 "TK-2024-001"）；创建失败时为 None。
        message:   展示给用户的提示文本（成功时含工单 ID 和预期处理时效，失败时含错误原因）。
    """
    ticket_id: str | None  # 工单 ID（创建成功时为字符串，失败时为 None）
    message: str           # 用户可见的操作结果提示文本


async def create_from_context(
    *,
    query: str,
    intent: str,
    entities: dict,
    ai_diagnosis: str,
    evidence: dict,
    ctx: ToolContext,
    user_id: str,
    conversation_id: str,
) -> TicketResult:
    """基于当前对话上下文创建人工工单，将 AI 诊断结论与证据随单落库。

    该函数是工单创建的统一入口，整合了以下信息：
    - 用户原始问题（工单标题/摘要）
    - 意图分类（决定工单类别和优先级）
    - AI 诊断结论（帮助人工快速了解问题背景）
    - 收集到的证据（日志/Key状态/限流数据等原始数据）
    - 关联的 request_id（可选，供工程师追溯调用记录）

    Args:
        query:           用户原始问题文本，截断后用作工单标题。
        intent:          意图识别结果字符串（如 "api_error"、"billing"），用于映射工单类别。
        entities:        NER 提取的实体字典，可含 request_id、endpoint、error_code 等键。
        ai_diagnosis:    其他专职子智能体生成的诊断/解释文本，附加到工单供人工参考。
        evidence:        收集到的原始证据字典（调用日志/Key状态/限流统计等）。
        ctx:             工具调用上下文（含鉴权信息、app_id 等），传递给 execute()。
        user_id:         发起请求的用户 ID，用于工单关联用户账户。
        conversation_id: 当前对话 ID，用于工单关联对话历史，便于人工查看完整对话。

    Returns:
        TicketResult 实例，包含工单 ID（成功时）或 None（失败时）及提示消息。
    """
    # 按意图映射工单类型与优先级，未知意图按低优先级咨询处理
    # CATEGORY_MAP.get() 第二参数为默认值，兜底处理未覆盖的意图类型
    category, priority = CATEGORY_MAP.get(intent, ("咨询", "P3"))

    # 提取 request_id：若实体中有 request_id，封装为单元素列表；否则为空列表
    # 工单系统支持关联多个 request_id，此处只关联用户提供的那一个
    request_ids = [entities["request_id"]] if entities.get("request_id") else []

    # 工单标题：取用户问题前 60 个字符（避免标题过长）；若问题为空则使用"[类别]工单"作为默认标题
    title = query.strip()[:60] or f"{category}工单"

    # 组装工单创建参数字典，传给工具中心的 create_ticket 工具
    args = {
        "title": title,                                          # 工单标题（用户问题摘要）
        "category": category,                                    # 工单类别（如"API报错"、"套餐账单"）
        "priority": priority,                                    # 优先级（P1/P2/P3）
        "summary": query.strip()[:500],                         # 问题详细描述（前 500 字符）
        "related_request_ids": request_ids,                     # 关联的 API 调用 request_id 列表
        "related_endpoint": entities.get("endpoint"),           # 关联的接口路径（如 "/v1/chat/completions"）
        "error_code": entities.get("error_code"),               # 关联的错误码（如 "RATE_LIMIT_EXCEEDED"）
        "evidence": json.dumps(evidence, ensure_ascii=False)[:2000],  # 证据 JSON（截断至 2000 字符，防止超限）
        "ai_diagnosis": (ai_diagnosis or "")[:2000],            # AI 诊断文本（截断至 2000 字符，可能为空字符串）
        "user_id": user_id,                                     # 用户 ID，关联工单归属账户
        "conversation_id": conversation_id,                     # 对话 ID，关联对话历史记录
    }
    # 经工具中心建单，证据与诊断随单落库，人工接手即可看到完整上下文
    # 通过工具注册中心统一调用，便于后续替换工单系统实现（如从内部系统迁移到 Jira/Zendesk）
    r = await execute("create_ticket", args, ctx)

    # 处理工单创建失败情况（如工单系统不可用、参数校验失败等）
    if not r["ok"]:
        return TicketResult(ticket_id=None, message=f"工单创建失败：{r.get('error')}")

    # 工单创建成功：提取工单 ID 并构造用户友好的提示消息
    tid = r["data"]["ticket_id"]  # 从工具返回数据中取出工单 ID（如 "TK-2024-001"）
    return TicketResult(
        ticket_id=tid,
        # 提示消息包含工单 ID、类别、优先级，让用户知道已受理并了解处理预期
        message=f"已为你创建工单 {tid}（{category}，优先级 {priority}），技术支持会尽快跟进。",
    )
