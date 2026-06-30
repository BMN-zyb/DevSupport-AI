# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Billing Agent：套餐/调用量/账单解释；高风险商业操作转人工。

本模块是 DevSupport-AI 多智能体系统中的计费专职子智能体。
职责：
  1. 查询用户当前套餐、历史用量、账单数据等真实业务数据。
  2. 结合计费规则文档，由 LLM 解释账单构成、套餐差异、超额原因等问题。
  3. 对涉及退款、改价、套餐升降级等高风险商业操作，识别后标记 need_human=True，
     禁止自动执行，改为转人工/商务团队跟进，避免自动化误操作。
"""

# -----------------------------------------------------------------------
# 标准库导入
# -----------------------------------------------------------------------
import json  # 用于将证据字典序列化为 JSON 字符串，传给 LLM 进行分析

# dataclass：用于定义轻量数据容器，存储计费查询结果
from dataclasses import dataclass, field

# -----------------------------------------------------------------------
# 项目内部模块导入
# -----------------------------------------------------------------------
from app.agents import doc_rag  # 文档 RAG 子智能体，用于检索计费规则等知识库文档
from app.agents.util import normalize_card, parse_json, render_card  # 卡片规范化、JSON 解析、卡片渲染工具
from app.llm import client  # LLM 客户端，封装向大模型发请求的接口
from app.llm.router import model_for  # 模型路由函数，根据场景名称返回对应模型配置
from app.tools.registry import ToolContext, execute  # 工具注册中心：execute 统一调用各业务工具

# 高风险商业意图关键词（不直接执行，转人工）
# 这些操作涉及资金变动或合同变更，AI 不具备授权，必须由人工核实后处理
HIGH_RISK_KEYWORDS = [
    "退款", "退费", "改价", "调价", "重置",
    "降套餐", "降级", "降到", "升到", "改成", "套餐变更", "变更套餐",
    "改套餐", "换套餐", "退订", "解约", "取消套餐",
]


@dataclass
class BillingResult:
    """计费查询结果数据容器，由 handle() 函数填充并返回给上层编排器。

    Attributes:
        answer:      格式化后的账单解释卡片文本，可直接展示给用户。
        evidence:    收集到的原始证据字典（套餐/用量/账单数据），供调试与归档。
        citations:   文档引用列表，每项含 doc_title/section/score 等字段。
        need_human:  是否需要转人工处理（命中高风险关键词时为 True）。
        tokens:      本次处理消耗的 LLM token 总数（解释 + 文档检索之和）。
        card:        LLM 输出的结构化 JSON 卡片原始数据，供上层进一步处理。
    """
    answer: str                                      # 渲染后的账单解释文本（展示给用户）
    evidence: dict = field(default_factory=dict)     # 原始证据字典（套餐/用量/账单，默认空字典）
    citations: list[dict] = field(default_factory=list)  # 文档引用列表（默认空列表）
    need_human: bool = False                         # 是否需要转人工处理（高风险操作标记）
    tokens: int = 0                                  # 本次总 token 消耗量
    card: dict | None = None                         # LLM 生成的 JSON 卡片原始对象


async def handle(query: str, entities: dict, ctx: ToolContext) -> BillingResult:
    """查套餐/用量/账单真实数据 + 计费文档，LLM 解释；高风险操作标记转人工。

    完整处理流程：
      1. 检测用户问题是否命中高风险商业关键词，标记 high_risk 标志。
      2. 依次查询套餐配置、调用用量、账单明细（串行，依赖同一用户上下文）。
      3. 调用文档 RAG 检索计费规则文档，作为 LLM 的知识背景。
      4. 将证据与文档交给 LLM，生成 JSON 格式的解释卡片。
      5. 若命中高风险关键词，在 prompt 中注入提示，并在返回结果中标记 need_human=True。

    Args:
        query:    用户原始问题文本（如"为什么这个月账单涨了这么多"）。
        entities: NER/意图识别提取的实体字典，可含 month 等键（用于过滤指定月份数据）。
        ctx:      工具调用上下文（含鉴权信息、app_id 等），传递给 execute()。

    Returns:
        BillingResult 实例，包含账单解释文本、证据、引用、人工转接标记等完整结果。
    """
    # 命中高风险商业关键词则只解释、不执行，最终标记 need_human
    # any() 短路求值：一旦命中一个关键词即停止遍历，效率较高
    high_risk = any(k in query for k in HIGH_RISK_KEYWORDS)

    evidence: dict = {}  # 初始化证据字典，后续各查询步骤将数据写入此字典

    # 查询套餐：获取用户当前订阅的套餐名称、配额、有效期等信息
    rp = await execute("query_plan", {}, ctx)  # 不传额外参数，查当前用户默认套餐
    if rp["ok"] and rp["data"].get("found"):
        evidence["plan"] = rp["data"]  # 套餐信息写入证据字典

    # 用量（指定月或全部）：从 entities 中提取 month 参数，若无则查全量用量
    month = entities.get("month")  # 用户可能指定"上个月"等月份，NER 已解析为 YYYY-MM 格式
    ru = await execute("query_usage", {"month": month} if month else {}, ctx)  # 有月份则按月过滤
    if ru["ok"] and ru["data"].get("found"):
        evidence["usage"] = ru["data"]["usage"]  # 用量数据（按月分布的调用次数/token数等）

    # 账单：查询账单明细（包含基础费用、超额费用、折扣等）
    rb = await execute("query_bill", {"month": month} if month else {}, ctx)  # 同样支持按月过滤
    if rb["ok"] and rb["data"].get("found"):
        evidence["bills"] = rb["data"]["bills"]  # 账单列表写入证据字典

    # 计费规则文档：固定查询计费相关文档，为 LLM 提供规则背景（套餐配额/超额计费/账单构成）
    doc = await doc_rag.answer("套餐计费规则、账单费用构成与超额计费")
    citations = doc.citations  # 提取文档引用，用于前端展示"参考来源"

    # 构造 System prompt：限定 LLM 角色、数据使用规范和输出格式
    sys = (
        "你是 API 平台账单助手。基于【证据】(真实套餐/用量/账单数据)与【文档】解释账单/套餐问题。"
        "必须直接使用证据中的真实数字（套餐配额、各月调用量、基础/超额费用），不要说无法查看数据。"
        "账单上涨要对比月度用量并说明费用构成(基础费用 vs 超额费用)。"
        "退款、改价、套餐升降级等商业操作不能直接执行，需说明将转人工/商务。\n"
        "输出 JSON：{\"conclusion\":\"一句话结论\", \"evidence\":[\"引用到的真实数据点\"], \"steps\":[\"建议操作\"]}。只输出 JSON。"
    )
    # 若命中高风险关键词，注入额外提示，要求 LLM 在结论中明确告知用户不可直接执行
    risk_note = "\n注意：用户请求涉及高风险商业操作，结论中需明确告知不能直接执行、将转人工/商务跟进。" if high_risk else ""
    # 构造 User prompt：拼接问题、高风险提示（可选）、证据数据、文档内容
    user = (
        f"【用户问题】{query}{risk_note}\n\n"                           # 用户原始问题 + 可选高风险注意
        f"【证据】{json.dumps(evidence, ensure_ascii=False)}\n\n"       # 序列化证据字典（中文字符不转义）
        f"【文档】{doc.answer}"                                         # RAG 检索到的计费文档内容
    )
    # 调用 LLM 生成账单解释（temperature=0.2 保证数字引用准确性和输出稳定性）
    gen = await client.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model_for("billing_explain"),  # 路由到账单解释场景对应的模型
        temperature=0.2,                     # 低温度减少幻觉，确保数字引用准确
    )
    card = normalize_card(parse_json(gen.content))  # 解析并规范化 LLM 输出的 JSON 卡片

    # 兜底处理：若 LLM 未能生成有效结论（JSON 解析失败或 conclusion 为空），用原始文本截断填充
    if not card["conclusion"]:
        card["conclusion"] = gen.content.strip()[:200]  # 截取前 200 字符作为兜底结论

    return BillingResult(
        answer=render_card(card),               # 将卡片渲染为格式化文本（展示给用户）
        evidence=evidence,                      # 原始证据字典（套餐/用量/账单数据）
        citations=citations,                    # 文档引用列表（来自 RAG）
        need_human=high_risk,                   # 命中高风险关键词则转人工（True/False）
        tokens=gen.total_tokens + doc.tokens,   # 累加账单解释 LLM 与文档检索的 token 消耗
        card=card,                              # 结构化卡片原始数据（供上层使用）
    )
