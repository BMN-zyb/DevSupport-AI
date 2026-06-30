# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Intent Router Agent：意图分类 + 实体抽取 + 推荐路由。

本文件实现了 DevSupport-AI 系统的意图识别模块（Intent Router），负责：
1. 将用户输入分类为预定义的意图类型（doc_qa / api_error / rate_limit / billing 等）。
2. 从用户输入中抽取关键结构化实体（request_id、error_code、endpoint 等）。
3. 根据意图类型返回推荐的专业 Agent 路由列表，指导 supervisor 分发请求。

设计亮点：
- 使用 LLM（大语言模型）做语义分类，覆盖模糊和复杂的用户表达。
- 使用正则表达式兜底补齐强模式实体（如 request_id、HTTP 状态码），防止 LLM 漏抽。
- 通过精心设计的系统提示（_SYS）包含消歧规则和示例，提升分类准确率。
"""

import json  # 用于解析 LLM 返回的 JSON 格式意图分类结果
import re    # 用于正则兜底抽取实体和清洗 LLM 输出的 ```json 代码块围栏

from app.llm import client          # LLM 统一调用客户端（封装了模型 API 调用）
from app.llm.router import model_for  # 根据任务类型动态选择合适的 LLM 模型

# 意图 -> 推荐专业 Agent
# 映射关系：每种意图对应需要调用哪些专业 Agent 模块处理
ROUTE_MAP = {
    "doc_qa": ["doc_rag"],                           # 文档问答：只需 RAG 检索文档
    "api_error": ["api_diagnostic", "doc_rag"],       # API 报错诊断：诊断 + 文档参考
    "rate_limit": ["api_diagnostic", "billing", "doc_rag"],  # 429 复合问题
    "billing": ["billing", "doc_rag"],               # 账单类：账单 Agent + 文档参考
    "data_quality": ["api_diagnostic", "doc_rag"],   # 数据质量问题：API 诊断 + 文档
    "ticket": ["ticket"],                            # 明确建单/转人工：只走工单节点
    "chitchat": [],                                  # 闲聊：不调用任何专业 Agent，由通用回复处理
}

INTENTS = list(ROUTE_MAP.keys())  # 所有合法意图类型列表，用于系统提示和意图合法性校验

# 意图识别系统提示：包含意图说明、消歧规则、示例和实体定义
# 精心设计的提示是分类准确率的核心，通过 f-string 动态插入 INTENTS 列表确保与代码同步
_SYS = (
    "你是 API 平台技术支持的意图识别器。判断用户问题的意图类型并抽取关键实体。\n"
    f"意图类型（只能选其一）：{INTENTS}\n"
    "- doc_qa: 询问文档/概念/用法/如何做/如何排查/错误码含义。包括「Webhook 回调如何排查」「签名怎么生成」等操作指导。\n"
    "- api_error: 针对某次具体调用的报错要做诊断定位（401/403/500/签名/参数等），通常带 request_id 或明确错误码/状态码。\n"
    "- rate_limit: 只要涉及 429 / QPS 超限 / 限流 / 大量请求被拒，一律归此类（即使提到了接口名）。\n"
    "- billing: 套餐/调用量/余额/账单/发票/费用/QPS上限与配额规则。\n"
    "- data_quality: 返回为空/数据不一致/字段缺失等数据质量问题。\n"
    "- ticket: 明确要求人工、投诉、查询工单。\n"
    "- chitchat: 与业务无关的闲聊。\n"
    "消歧规则：提到 429/限流→rate_limit；问「怎么排查/如何配置/是什么含义」的指导类→doc_qa；"
    "只有要定位某次具体失败(带request_id或明确错误码)才用 api_error；"
    "退款/改价/套餐升降级/变更等商业诉求→billing（由账单模块说明并转人工），不要归 ticket；"
    "ticket 仅用于明确要求人工、投诉或查询既有工单。\n"
    "示例：\n"
    "  「下午很多429是不是挂了」→ rate_limit\n"
    "  「Webhook 回调收不到怎么排查」→ doc_qa\n"
    "  「SIGN_INVALID 是什么原因」→ doc_qa\n"
    "  「接口返回401，request_id是req_x」→ api_error\n"
    "实体字段（无则留空字符串）：request_id, error_code, http_status, endpoint, app_id, month(YYYY-MM), webhook_event_id, invoice_id, ticket_id\n"
    "实体规则：http_status 只放数字状态码(如 401/429/500)；error_code 只放大写字母下划线错误码(如 AUTH_KEY_EXPIRED/SIGN_INVALID)，"
    "不要把数字状态码填进 error_code；endpoint 形如 /v1/idcard/verify。\n"
    "只输出 JSON：{\"intent\":..., \"confidence\":0~1, \"entities\":{...}}"
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)  # 贪婪匹配 LLM 输出中的 JSON 对象（跨行），用于提取嵌套在文本中的 JSON

# 强模式实体的正则兜底（LLM 偶尔漏抽，规则补齐更稳）
# 各正则模式对应特征明显、格式固定的实体类型，正则比 LLM 更稳定
_ENTITY_RE = {
    "request_id": re.compile(r"\breq_[0-9A-Za-z_]+\b"),           # request_id 以 req_ 开头，如 req_abc123
    "error_code": re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"),   # 全大写+下划线错误码，如 SIGN_INVALID
    "endpoint": re.compile(r"/v\d+/[A-Za-z0-9/_-]+"),             # API 路径，如 /v1/idcard/verify
    "http_status": re.compile(r"\b[1-5]\d{2}\b"),                  # 3 位 HTTP 状态码，如 401/429/500
}


def _regex_entities(query: str) -> dict:
    """使用正则表达式从查询文本中兜底抽取强模式实体。

    当 LLM 漏抽某个实体时，此函数作为补充手段。每个正则只取第一个匹配，
    因为用户通常在单条消息中只提及一个 request_id 或状态码。

    Args:
        query: 用户输入的原始查询文本。
    Returns:
        抽取到的实体字典，key 为实体名，value 为匹配字符串；未匹配则不含该 key。
    """
    found = {}  # 存放找到的实体，key 为实体名
    for key, pat in _ENTITY_RE.items():  # 遍历所有正则模式
        m = pat.search(query)  # 在查询文本中搜索该模式
        if m:
            found[key] = m.group(0)  # 命中则取第一个匹配值存入字典
    return found  # 返回所有找到的强模式实体


def _parse_json(text: str) -> dict:
    """从 LLM 返回文本中稳健解析 JSON，兼容代码块围栏和多余文本包裹。

    LLM 有时会在 JSON 前后加 ```json 围栏或说明文字，此函数处理这些情况
    并在解析失败时安全返回空字典，避免上层代码崩溃。

    Args:
        text: LLM 原始输出文本。
    Returns:
        解析成功返回 dict；任何解析失败均返回空字典 {}。
    """
    text = text.strip()  # 去除首尾空白字符
    if text.startswith("```"):
        # 去除 ```json ... ``` 代码块围栏，flags=re.MULTILINE 让 ^ $ 匹配每行
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = _JSON_RE.search(text)  # 从清洗后的文本中查找 JSON 对象
    if not m:
        return {}  # 未找到 JSON 结构，返回空字典
    try:
        return json.loads(m.group(0))  # 尝试解析找到的 JSON 字符串
    except json.JSONDecodeError:
        return {}  # JSON 格式错误（如截断或转义问题），安全返回空字典


async def classify(query: str, history: list[dict] | None = None) -> dict:
    """对单条 query 做意图分类 + 实体抽取，返回意图/置信度/实体/推荐路由。

    处理流程：
    1. 构建包含系统提示和（可选）历史上下文的消息列表。
    2. 调用 LLM 做意图识别，temperature=0 保证稳定可复现。
    3. 解析 JSON 输出，进行意图合法性兜底和实体正则补齐。
    4. 返回结构化结果供 supervisor 的 intent_node 使用。

    Args:
        query: 当前用户输入的问题文本。
        history: 可选的对话历史列表（[{role, content}]），用于多轮指代消解。
    Returns:
        包含以下 key 的字典：
            - intent (str): 分类后的意图类型
            - confidence (float): 意图置信度 0~1
            - entities (dict): 抽取并补齐后的实体字典
            - route (list[str]): 推荐的专业 Agent 列表
            - tokens (int): 本次 LLM 调用消耗的 token 数
    """
    msgs = [{"role": "system", "content": _SYS}]  # 始终以意图识别系统提示开头
    if history:
        # 仅带最近 4 条历史，兼顾多轮指代消解与 token 成本
        # 例如 "上面那个接口" 需要历史才能知道是哪个 endpoint
        hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])  # 格式化历史消息为纯文本
        msgs.append({"role": "user", "content": f"对话历史：\n{hist_text}"})  # 将历史作为 user 消息注入
    msgs.append({"role": "user", "content": f"用户问题：{query}"})  # 最后追加当前用户问题

    # temperature=0 保证分类稳定可复现
    r = await client.chat(msgs, model=model_for("intent"), temperature=0.0)  # 调用 LLM 进行意图分类，温度为 0 保证一致性
    parsed = _parse_json(r.content)  # 解析 LLM 返回的 JSON 格式分类结果

    intent = parsed.get("intent", "doc_qa")  # 获取分类意图，LLM 未返回时默认为 doc_qa（最安全的兜底）
    if intent not in ROUTE_MAP:  # LLM 给出非法意图时回落到文档问答
        intent = "doc_qa"  # 非法意图（不在 ROUTE_MAP 中）时强制回落，防止后续路由崩溃
    entities = {k: v for k, v in (parsed.get("entities") or {}).items() if v}  # 过滤空值实体，仅保留非空字段
    # 正则兜底：强模式实体未被 LLM 抽到时补齐
    for k, v in _regex_entities(query).items():
        entities.setdefault(k, v)  # setdefault：仅当该实体 key 不存在时才补充（不覆盖 LLM 已抽到的值）
    try:
        confidence = float(parsed.get("confidence", 0.5))  # 将置信度转为浮点数，LLM 未返回时默认 0.5
    except (TypeError, ValueError):
        confidence = 0.5  # 置信度字段格式异常时安全回落到 0.5（中性值）

    return {
        "intent": intent,            # 最终确定的意图类型
        "confidence": confidence,    # 意图分类的置信度
        "entities": entities,        # 补齐后的实体字典
        "route": ROUTE_MAP[intent],  # 根据意图查表得到推荐的专业 Agent 路由列表
        "tokens": r.total_tokens,    # 本次 LLM 调用的 token 消耗，用于成本追踪
    }
