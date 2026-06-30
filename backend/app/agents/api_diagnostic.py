# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""API Diagnostic Agent：基于调用日志/Key状态/限流统计 + 文档，辅助定位 API 报错。

输出结构：诊断结论 / 证据摘要 / 可能原因 / 建议操作 / 关联文档 / 是否需要工单。

本模块是 DevSupport-AI 多智能体系统中的 API 诊断专职子智能体。
职责：
  1. 收集多维度证据：调用日志、API Key 状态、限流统计。
  2. 并行调用文档 RAG 检索相关错误码说明。
  3. 将证据与文档交给 LLM，组装结构化 JSON 诊断卡片返回给上层编排器。
  4. 若证据不足以定位问题，标记 need_ticket=True，触发工单创建流程。
"""

# -----------------------------------------------------------------------
# 标准库导入
# -----------------------------------------------------------------------
import asyncio  # 用于并发执行多个异步 I/O 操作（如同时查 Key/限流/文档）
import json     # 用于将证据字典序列化为 JSON 字符串，传给 LLM

# dataclass：用于定义轻量数据容器，避免手写 __init__/__repr__ 等样板代码
from dataclasses import dataclass, field

# -----------------------------------------------------------------------
# 项目内部模块导入
# -----------------------------------------------------------------------
from app.agents import doc_rag  # 文档 RAG 子智能体，负责知识库检索与问答
from app.agents.util import normalize_card, parse_json, render_card  # 卡片数据规范化、JSON 解析、卡片渲染工具
from app.db import AsyncSessionLocal  # 异步数据库 Session 工厂，用于查询 ErrorCode 表
from app.llm import client  # LLM 客户端，封装了向大模型发请求的接口
from app.llm.router import model_for  # 模型路由函数，根据场景名称返回对应模型配置
from app.models import ErrorCode  # ORM 模型：错误码表（code/name/cause/fix_steps）
from app.tools.registry import ToolContext, execute  # 工具注册中心：execute 统一调用各业务工具


async def _error_doc_fast(error_code: str):
    """已知错误码直接查 error_code 表，跳过完整 RAG（性能优化热路径）。

    对于系统内已维护的标准错误码（如 AUTH_KEY_EXPIRED、RATE_LIMIT_EXCEEDED），
    直接从数据库读取预定义的原因与处理步骤，避免走完整的 embed/rerank/generate 流程，
    大幅降低延迟与 token 消耗。

    Args:
        error_code: 错误码字符串，例如 "AUTH_KEY_EXPIRED"。

    Returns:
        若命中：返回 (answer_str, citations_list) 二元组；
        若未命中（表中无此记录）：返回 None，由调用方回退到完整 RAG。
    """
    async with AsyncSessionLocal() as s:  # 打开一个异步数据库 Session（用完自动关闭）
        row = await s.get(ErrorCode, error_code)  # 按主键（error_code）查询错误码记录
    if not row:
        return None  # 未命中则返回 None，由调用方决定是否走完整 RAG
    # 将数据库记录拼接为自然语言答案，格式：错误码（名称）：原因 处理步骤：步骤
    answer = f"{row.code}（{row.name}）：{row.cause} 处理步骤：{row.fix_steps}"
    # 构造引用列表，供前端展示"引用来源"（index 固定为 1，score 固定为满分 1.0）
    citations = [{"index": 1, "doc_title": "错误码手册", "section": row.code, "version": "v1", "score": 1.0}]
    return answer, citations  # 返回答案文本与引用列表


@dataclass
class DiagResult:
    """API 诊断结果数据容器，由 diagnose() 函数填充并返回给上层编排器。

    Attributes:
        answer:      格式化后的诊断卡片文本，可直接展示给用户。
        evidence:    收集到的原始证据字典（日志/Key状态/限流统计），供调试与归档。
        citations:   文档引用列表，每项含 doc_title/section/score 等字段。
        error_code:  从日志或实体中提取/推断出的错误码（可能为 None）。
        need_ticket: 是否建议创建工单（证据不足或需人工介入时为 True）。
        tokens:      本次诊断消耗的 LLM token 总数（诊断 + 文档检索之和）。
        card:        LLM 输出的结构化 JSON 卡片原始数据，供上层进一步处理。
    """
    answer: str                                      # 渲染后的诊断文本（展示给用户）
    evidence: dict = field(default_factory=dict)     # 原始证据字典（默认空字典）
    citations: list[dict] = field(default_factory=list)  # 文档引用列表（默认空列表）
    error_code: str | None = None                    # 识别到的错误码（可为空）
    need_ticket: bool = False                        # 是否需要创建人工工单
    tokens: int = 0                                  # 本次总 token 消耗量
    card: dict | None = None                         # LLM 生成的 JSON 卡片原始对象


async def diagnose(query: str, entities: dict, ctx: ToolContext, *, is_rate_limit: bool = False) -> DiagResult:
    """收集证据（日志/Key/限流统计）+ 文档，交 LLM 组装结构化诊断卡片。

    完整诊断流程：
      1. 若有 request_id，先查调用日志，从日志中补全 error_code/endpoint/app_id。
      2. 并行执行：查询 API Key 状态（仅 Key 相关错误）、限流统计（仅限流相关）、
         文档检索（已知错误码走快速路径，否则走完整 RAG）。
      3. 将所有证据与文档内容拼接成 prompt，调用 LLM 生成 JSON 诊断卡片。
      4. 根据 need_ticket 字段及证据完整性决定是否标记需要创建工单。

    Args:
        query:         用户原始问题文本。
        entities:      NER/意图识别提取的实体字典，可含 error_code/endpoint/request_id 等键。
        ctx:           工具调用上下文（含鉴权信息、app_id 等），传递给 execute()。
        is_rate_limit: 上层编排器判断该意图为限流时传 True，强制拉取限流统计。

    Returns:
        DiagResult 实例，包含诊断文本、证据、引用、工单标记等完整结果。
    """
    evidence: dict = {}  # 初始化证据字典，后续各步骤将证据写入此字典

    # 从实体字典中提取三个关键实体（可能均为 None，需后续补全）
    error_code = entities.get("error_code")  # 错误码，如 "RATE_LIMIT_EXCEEDED"
    endpoint = entities.get("endpoint")      # 接口路径，如 "/v1/chat/completions"
    request_id = entities.get("request_id")  # 请求 ID，用于精确查单次调用日志

    # 1. 有 request_id → 先查调用日志（后续步骤依赖其 error_code/endpoint/app_id）
    app_id = None  # app_id 需从日志中获取，初始为 None
    if request_id:
        # 调用"查询调用日志"工具，传入 request_id 精确匹配
        r = await execute("query_call_log", {"request_id": request_id}, ctx)
        if r["ok"] and r["data"].get("found"):
            log = r["data"]               # 日志命中，取出日志数据
            evidence["call_log"] = log    # 将日志存入证据字典，供 LLM 分析
            # 若 entities 未提供 error_code/endpoint，则从日志中补全
            error_code = error_code or log.get("error_code")
            endpoint = endpoint or log.get("endpoint")
            app_id = log.get("app_id")    # 从日志中获取应用 ID，供后续查 Key 状态
        else:
            # 日志未命中，记录"未找到"标记，后续可据此判断是否需要工单
            evidence["call_log"] = {"found": False, "request_id": request_id}

    # 2. 性能优化：Key 状态查询、限流统计、文档支撑 三者并行
    # 判断是否需要查 API Key 状态：仅当错误码为 Key 相关且有 app_id 时才查
    need_apikey = error_code in ("AUTH_KEY_EXPIRED", "AUTH_KEY_INVALID") and app_id
    # 判断是否需要查限流统计：外部标记为限流、或错误码为限流/配额相关、或有 endpoint 时查
    need_stats = is_rate_limit or error_code in ("RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED") or bool(endpoint)

    async def _apikey():
        """条件性查询 API Key 状态：仅在 need_apikey 为 True 时执行实际查询。"""
        return await execute("query_apikey_status", {"app_id": app_id}, ctx) if need_apikey else None

    async def _stats():
        """条件性查询最近 60 分钟限流统计：仅在 need_stats 为 True 时执行实际查询。"""
        return await execute("query_recent_call_stats", {"endpoint": endpoint, "minutes": 60}, ctx) if need_stats else None

    async def _doc():
        """文档检索：已知错误码走快速路径，否则走完整 RAG。"""
        # 热路径：已知错误码直查 error_code 表，跳过 RAG 的 embed/rerank/generate
        if error_code:
            fast = await _error_doc_fast(error_code)  # 尝试从数据库快速获取错误码说明
            if fast:
                return ("fast", fast[0], fast[1], 0)  # ("fast", 答案文本, 引用列表, 消耗token=0)
        # 完整 RAG 流程：构造查询（若有错误码则聚焦错误码解释，否则用原始问题）
        d = await doc_rag.answer(
            f"{error_code} 含义、原因与处理步骤" if error_code else query,  # 针对错误码优化查询文本
            error_code=error_code or None,  # 传递错误码给 RAG，辅助过滤相关文档
        )
        return ("rag", d.answer, d.citations, d.tokens)  # ("rag", 答案文本, 引用列表, token消耗)

    # 并行执行三个异步任务，最大化 I/O 效率（避免串行等待）
    rk, rs, doc_res = await asyncio.gather(_apikey(), _stats(), _doc())

    # 处理 API Key 状态查询结果
    if rk and rk["ok"]:
        evidence["apikey_status"] = rk["data"]  # 将 Key 状态写入证据字典

    # 处理限流统计查询结果（仅当有实际调用数据时才写入证据，避免空数据干扰 LLM）
    if rs and rs["ok"] and rs["data"].get("total"):
        evidence["recent_stats"] = rs["data"]  # 将限流统计写入证据字典

    # 解包文档检索结果（忽略第一个字段"fast"/"rag"路径标识）
    _, doc_answer, citations, doc_tokens = doc_res

    # 5. LLM 组装结构化诊断（JSON 卡片）
    # System prompt：限定 LLM 角色、输入范围和输出格式，防止编造
    sys = (
        "你是 API 平台诊断助手。基于【证据】(真实调用日志/Key状态/限流统计)和【文档】输出结构化诊断。"
        "只能依据证据与文档，不得编造；涉及密钥只展示脱敏值。\n"
        "输出 JSON：{\"conclusion\":\"一句话诊断结论\", \"evidence\":[\"关键证据(含状态码/错误码/时间/脱敏Key等)\"], "
        "\"steps\":[\"可执行修复步骤\"], \"need_ticket\":true/false}。need_ticket：证据不足以定位或需人工时为 true。只输出 JSON。"
    )
    # User prompt：将用户问题、收集的证据、文档内容拼接成结构化输入
    user = (
        f"【用户问题】{query}\n\n"
        f"【证据】{json.dumps(evidence, ensure_ascii=False)}\n\n"  # ensure_ascii=False 保留中文字符
        f"【文档】{doc_answer}"
    )
    # 调用 LLM 生成诊断（temperature=0.2 保证输出稳定性，减少随机性）
    gen = await client.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model_for("diagnose"),  # 根据场景名称路由到合适的模型（如 GPT-4o/Claude 等）
        temperature=0.2,              # 低温度确保诊断结论一致性
    )
    parsed = parse_json(gen.content)   # 解析 LLM 输出的 JSON 字符串，容错处理格式异常
    card = normalize_card(parsed)      # 规范化卡片结构，补全缺失字段为默认值
    need_ticket = bool(parsed.get("need_ticket", False))  # 读取 LLM 判断的工单需求

    # 证据不足（无有效日志）也建议工单：即使 LLM 未标记 need_ticket，
    # 若用户提供了 request_id 但日志未找到，说明需要人工排查
    if request_id and not evidence.get("call_log", {}).get("found", False):
        need_ticket = True  # 强制标记需要工单，人工核查日志缺失原因

    # 兜底处理：若 LLM 未能生成有效结论（如 JSON 解析失败），用原始文本截断填充
    if not card["conclusion"]:
        card["conclusion"] = gen.content.strip()[:200]  # 截取前 200 字符作为兜底结论

    # 组装最终返回结果
    return DiagResult(
        answer=render_card(card),           # 将卡片渲染为格式化文本（Markdown 或纯文本）
        evidence=evidence,                  # 原始证据字典（含日志/Key/限流数据）
        citations=citations,                # 文档引用列表（来自 RAG 或快速路径）
        error_code=error_code,              # 最终确定的错误码（可能从日志中补全）
        need_ticket=need_ticket,            # 是否需要创建工单
        tokens=gen.total_tokens + doc_tokens,  # 累加诊断 LLM 与文档检索的 token 消耗
        card=card,                          # 结构化卡片原始数据（供上层使用）
    )
