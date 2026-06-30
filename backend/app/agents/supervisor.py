# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Supervisor：用 LangGraph 编排多 Agent，串行保证安全审查在最终回复前。

管线：load_context → intent → [clarify?] → specialists(并行) → ticket → summarize → security
每个节点记录 AgentTrace；任一专业 Agent 异常被隔离降级，不影响整体。

本文件是整个 DevSupport-AI 系统的编排核心（Supervisor），负责：
1. 定义并连接 LangGraph 有向图中的所有节点（上下文加载、意图识别、澄清、
   专业 Agent 并行分发、工单、摘要、安全审查）。
2. 对外暴露 run() 接口，作为单次完整对话请求的入口，包含语义缓存、
   fallback 兜底、记忆持久化、成本统计等横切关注点。
"""

import asyncio  # 用于并行调度多个协程（asyncio.gather）

from langgraph.graph import END, START, StateGraph  # LangGraph 图构建工具：起止节点 + 状态图

# 导入各专业子 Agent 模块，每个模块负责特定领域的回答与处理
from app.agents import api_diagnostic, billing, doc_rag, intent_router, security, ticket
from app.agents.state import AgentState  # 整个图共享的 TypedDict 状态定义
from app.agents.util import render_card  # 将结构化卡片渲染为 markdown 文本
from app.cache import route_cache, semantic_cache  # 路由缓存 + 语义相似度缓存，用于降低重复 LLM 调用
from app.config import settings  # 全局配置，例如意图置信度阈值
from app.guardrail import fallback  # 规则化兜底回复，管线崩溃时使用
from app.llm import client  # LLM 统一调用客户端
from app.llm.router import model_for  # 根据任务类型选择合适的 LLM 模型
from app.memory import session  # 会话记忆模块，读写历史消息与实体
from app.observability import cost  # Token 成本统计记录
from app.observability.trace import TraceCollector, timer  # 链路追踪工具：收集器 + 计时器
from app.tools.registry import ToolContext, load_tools  # 工具注册表 + 工具上下文

load_tools()  # 程序启动时立即注册所有工具，确保后续节点可用


async def _general_reply(query: str) -> tuple[str, int]:
    """处理闲聊/通用问题，引导用户提出技术支持请求。

    Args:
        query: 用户原始输入文本。
    Returns:
        (回复文本, 消耗 token 数) 的二元组。
    """
    r = await client.chat(  # 调用 LLM 客户端，发送消息列表
        [
            {"role": "system", "content": "你是 API 平台技术支持助手。对与业务无关的闲聊，礼貌简短回应，并引导用户提出 API 接入/报错/账单等技术支持问题。"},
            {"role": "user", "content": query},  # 将用户原始问题放入 user 消息
        ],
        model=model_for("chitchat"),  # 闲聊场景使用轻量级模型节省成本
        temperature=0.5,  # 适当提高 temperature 让回复更自然多样
    )
    return r.content.strip(), r.total_tokens  # 去除首尾空白并返回内容与 token 消耗


def build_graph(ctx: ToolContext, trace: TraceCollector):
    """按请求构建图（闭包捕获 ctx 与 trace）。

    每次请求动态构建一个新的 LangGraph 编排图，通过闭包将 ctx（工具上下文）
    和 trace（链路追踪器）注入到各节点函数中，从而隔离不同请求的状态。

    Args:
        ctx: 工具上下文，包含租户 ID、trace_id 等权限信息。
        trace: 链路追踪收集器，记录每个节点的耗时与 token。
    Returns:
        编译好的 LangGraph CompiledGraph，可直接调用 ainvoke。
    """

    async def load_context(state: AgentState) -> dict:
        """节点1：从会话记忆中加载历史消息与已收集的实体，注入到图状态。

        Args:
            state: 当前图状态，需包含 conversation_id。
        Returns:
            含 history（历史对话列表）和 collected_entities（已知实体字典）的更新字典。
        """
        conv = state["conversation_id"]  # 取出本次对话 ID，作为记忆查询 key
        history = await session.get_history(conv)  # 异步拉取该会话的历史消息列表
        entities = await session.get_entities(conv)  # 异步拉取该会话已收集的实体（如 app_id、request_id）
        return {"history": history, "collected_entities": entities}  # 将结果写回图状态

    async def intent_node(state: AgentState) -> dict:
        """节点2：意图识别 + 实体抽取 + 路由决策 + 澄清判断。

        优先查询路由缓存（无历史时命中率更高），未命中则调用 intent_router 分类。
        根据意图和置信度判断是否需要向用户追问澄清信息。

        Args:
            state: 当前图状态，需包含 query、conversation_id。
        Returns:
            含 intent、confidence、entities、route、need_clarify、clarify_question 的更新字典。
        """
        history = state.get("history")  # 获取已加载的历史消息，可能为空
        with timer() as t:  # 开启计时器，记录本节点耗时
            # 无历史时才查路由缓存（有历史的多轮对话语义更复杂，缓存命中率低且易误用）
            cached_route = None if history else await route_cache.get(state["query"])
            if cached_route:
                res = {**cached_route, "tokens": 0}  # 命中缓存则 token 计为 0（未实际调用 LLM）
            else:
                res = await intent_router.classify(state["query"], history)  # 调用 LLM 做意图分类
                if not history:
                    await route_cache.put(state["query"], res)  # 无历史时把结果写入路由缓存供后续复用
        # 合并记忆中的实体（新抽取覆盖/补充）
        merged = await session.update_entities(state["conversation_id"], res["entities"])  # 将本轮抽到的实体合并到会话记忆
        trace.step(
            "intent_router",
            input_summary=state["query"],  # 记录本节点的输入摘要，用于链路排查
            output_summary=f"intent={res['intent']} conf={res['confidence']} entities={merged}",  # 记录输出摘要
            duration_ms=t.ms,  # 节点耗时（毫秒）
            token_usage=res["tokens"],  # 本节点 LLM 消耗的 token 数
        )
        # 澄清判定：API 报错/数据质量类缺关键定位信息时追问
        need_clarify = False  # 默认不需要追问
        clarify_q = ""  # 追问文本，空时表示不追问
        if res["intent"] == "api_error" and not merged.get("request_id") and not merged.get("error_code"):
            # api_error 意图但缺少 request_id 和 error_code，无法精准诊断，需要追问
            need_clarify = True
            clarify_q = "为帮你准确定位，请提供以下任一信息：request_id、出错的接口名，或返回的错误码/HTTP状态码（最多 3 项即可）。"
        elif res["confidence"] < settings.intent_confidence_threshold and res["intent"] != "chitchat":
            # 意图置信度低于阈值且不是闲聊，说明分类不确定，需要用户补充场景描述
            need_clarify = True
            clarify_q = "我不太确定你的问题类型，能补充一下具体场景吗？比如是接口报错、账单费用，还是文档用法？"
        return {
            "intent": res["intent"],  # 最终识别的意图类型
            "confidence": res["confidence"],  # 意图分类的置信度 0~1
            "entities": merged,  # 合并后的完整实体字典
            "route": res["route"],  # 推荐的专业 Agent 列表
            "need_clarify": need_clarify,  # 是否需要向用户追问
            "clarify_question": clarify_q,  # 追问的具体内容
        }

    def after_intent(state: AgentState) -> str:
        """条件路由函数：根据是否需要澄清，决定下一步跳到 clarify 还是 specialists 节点。

        Args:
            state: 当前图状态，需包含 need_clarify 字段。
        Returns:
            字符串 "clarify" 或 "specialists"，由 LangGraph 路由到对应节点。
        """
        return "clarify" if state.get("need_clarify") else "specialists"  # need_clarify 为 True 则追问，否则直接进入专业 Agent

    async def clarify_node(state: AgentState) -> dict:
        """节点3（可选）：直接返回澄清追问作为最终答案，结束本轮对话。

        当意图识别认为需要追问时进入此节点，记录 trace 后把追问文本写入 final_answer。

        Args:
            state: 当前图状态，需包含 query 和 clarify_question。
        Returns:
            含 final_answer（追问文本）和 need_human=False 的字典。
        """
        trace.step("clarify", input_summary=state["query"], output_summary=state["clarify_question"])  # 记录追问节点的链路信息
        return {"final_answer": state["clarify_question"], "need_human": False}  # 将追问文本作为最终回复，无需转人工

    async def _run_agent(name: str, coro):
        """辅助函数：包装单个 Agent 协程，实现异常隔离（单 Agent 失败不影响其他 Agent）。

        Args:
            name: Agent 名称字符串，用于 trace 记录和结果索引。
            coro: 该 Agent 对应的异步协程。
        Returns:
            四元组 (name, result, timer, error_str)。
            正常时 error_str 为 None；异常时 result 为 None，error_str 为异常描述。
        """
        with timer() as t:  # 计时：记录单个 Agent 的执行耗时
            try:
                result = await coro  # 等待 Agent 协程执行完毕
                return name, result, t, None  # 成功：返回 (名称, 结果, 计时, None)
            except Exception as e:  # noqa: BLE001  单 Agent 异常隔离
                return name, None, t, f"{type(e).__name__}: {e}"  # 失败：捕获异常，返回异常信息而非抛出

    async def specialists_node(state: AgentState) -> dict:
        """节点4：并行分发到选中的专业 Agent，汇总输出结果、引用文档与建单需求。

        根据 intent_node 确定的 route 列表，并行运行对应的专业 Agent
        （api_diagnostic / billing / doc_rag），收集各 Agent 结果。

        Args:
            state: 当前图状态，需包含 route、query、entities、intent 等字段。
        Returns:
            含 agent_outputs、rag_citations、need_human、pending_ticket 的字典。
        """
        route = state.get("route", [])  # 获取推荐路由列表，默认空列表
        query, entities = state["query"], state.get("entities", {})  # 取出用户问题和已知实体
        outputs: dict = {}  # 收集各 Agent 的输出结果，key 为 Agent 名
        citations: list = []  # 汇总所有 Agent 返回的引用文档列表
        need_human = False  # 是否需要转人工，任一 Agent 认为需要则为 True
        need_ticket = False  # 是否需要创建工单

        if not route:  # chitchat 场景：route 为空，使用通用回复
            with timer() as t:
                reply, tok = await _general_reply(query)  # 调用闲聊回复函数
            trace.step("general", input_summary=query, output_summary=reply[:120], duration_ms=t.ms, token_usage=tok)
            return {"agent_outputs": {"general": reply}, "rag_citations": []}  # 直接返回通用回复

        # 并行运行选中的专业 Agent
        coros = []  # 存储待并行执行的协程列表
        if "api_diagnostic" in route:
            # API 诊断 Agent：传入查询、实体、工具上下文和是否为限流意图
            coros.append(_run_agent("api_diagnostic", api_diagnostic.diagnose(
                query, entities, ctx, is_rate_limit=state["intent"] == "rate_limit")))
        if "billing" in route:
            # 账单 Agent：处理套餐/余额/发票等账单类问题
            coros.append(_run_agent("billing", billing.handle(query, entities, ctx)))
        # doc_rag 仅在没有诊断/账单作为主答时单独使用，避免重复（诊断/账单内部已含文档）
        if "doc_rag" in route and not ("api_diagnostic" in route or "billing" in route):
            # 文档 RAG Agent：在没有其他专业 Agent 处理时，基于向量检索回答文档类问题
            coros.append(_run_agent("doc_rag", doc_rag.answer(query, history=state.get("history"))))

        results = await asyncio.gather(*coros)  # 并行执行所有 Agent 协程，等待全部完成
        for name, result, t, err in results:  # 遍历每个 Agent 的执行结果
            if err:
                # Agent 执行出错：记录错误 trace 后跳过，不影响其他 Agent 结果
                trace.step(name, input_summary=query, status="error", duration_ms=t.ms, error=err)
                continue
            tok = getattr(result, "tokens", 0)  # 获取该 Agent 消耗的 token 数，无则为 0
            hit = [c.get("doc_title") for c in getattr(result, "citations", [])]  # 提取引用的文档标题列表
            trace.step(name, input_summary=query, output_summary=getattr(result, "answer", "")[:160],
                       duration_ms=t.ms, token_usage=tok, hit_docs=hit)  # 记录 Agent 节点的链路信息
            outputs[name] = result  # 将成功结果存入 outputs 字典
            citations.extend(getattr(result, "citations", []))  # 汇总引用文档
            if getattr(result, "need_human", False):
                need_human = True  # 任一 Agent 认为需要转人工，则标记为 True
            if getattr(result, "need_ticket", False):
                need_ticket = True  # 任一 Agent 认为需要创建工单，则标记为 True

        return {
            "agent_outputs": outputs,  # 所有成功 Agent 的输出结果
            "rag_citations": citations,  # 汇总的引用文档列表
            "need_human": need_human or state["intent"] == "ticket",  # 明确 ticket 意图也要转人工
            "pending_ticket": need_ticket,  # 是否触发后续工单节点
        }

    async def ticket_node(state: AgentState) -> dict:
        """节点5：根据诊断结果创建工单，并返回工单 ID 和友好提示。

        仅在 need_human 或 pending_ticket 为 True 时执行实际工单创建，
        否则直接返回空字典（跳过）。

        Args:
            state: 当前图状态，需包含 agent_outputs、query、intent、entities 等字段。
        Returns:
            含 ticket_id 和 ticket_message 的字典，或空字典（无需建单时）。
        """
        outputs = state.get("agent_outputs", {})  # 获取各专业 Agent 的输出
        need_ticket = state.get("pending_ticket") or state.get("need_human")  # 两种条件均可触发建单
        if not need_ticket:
            return {}  # 不需要建单，直接返回空字典，节点透明跳过
        # 汇总现有诊断作为工单上下文
        ai_diag = ""  # AI 诊断结论文本，用于工单描述
        evidence = {}  # 诊断证据字典，用于工单附加信息
        if "api_diagnostic" in outputs:
            # 优先使用 API 诊断结果作为工单上下文
            ai_diag = outputs["api_diagnostic"].answer
            evidence = outputs["api_diagnostic"].evidence
        elif "billing" in outputs:
            # 其次使用账单 Agent 结果作为工单上下文
            ai_diag = outputs["billing"].answer
            evidence = outputs["billing"].evidence
        with timer() as t:  # 计时：记录建单操作耗时
            tk = await ticket.create_from_context(
                query=state["query"], intent=state["intent"], entities=state.get("entities", {}),
                ai_diagnosis=ai_diag, evidence=evidence, ctx=ctx,  # 工具上下文，含租户权限
                user_id=state.get("user_id", ""), conversation_id=state["conversation_id"])
        trace.step("ticket", input_summary=state["query"], output_summary=tk.message, duration_ms=t.ms)
        return {"ticket_id": tk.ticket_id, "ticket_message": tk.message}  # 返回工单 ID 和给用户的提示

    async def summarize_node(state: AgentState) -> dict:
        """节点6：将多个专业 Agent 的结构化卡片合并为统一的 draft_answer。

        处理三种情况：
        1. 只有通用回复（闲聊）→ 直接使用。
        2. 单卡片 → 直接渲染为 markdown。
        3. 多卡片（复合问题如 429）→ 调用 LLM 合并结论，证据和步骤取并集。

        Args:
            state: 当前图状态，需包含 agent_outputs 字段。
        Returns:
            含 draft_answer（文本草稿）和 card（结构化卡片）的字典。
        """
        outputs = state.get("agent_outputs", {})  # 获取所有 Agent 的输出结果
        cards = []  # 收集各 Agent 返回的结构化卡片
        for name in ("api_diagnostic", "billing", "doc_rag"):
            o = outputs.get(name)  # 获取该 Agent 的输出对象
            if o is not None and getattr(o, "card", None):
                cards.append(o.card)  # 若有结构化卡片则收集起来
        general = outputs.get("general")  # 闲聊通用回复（若有）

        card = None  # 最终输出的单一结构化卡片（合并后）
        if general and not cards:
            draft = general  # 情况1：只有通用回复，直接作为草稿
        elif len(cards) == 1:
            card = cards[0]  # 情况2：仅一个卡片，直接渲染
            draft = render_card(card)  # 将结构化卡片渲染为 markdown 文本
        elif len(cards) > 1:
            # 复合问题（如 429）：小模型合并多条结论，证据/步骤取并集
            with timer() as t:  # 计时：记录合并摘要的耗时
                gen = await client.chat(
                    [
                        {"role": "system", "content": "把以下多条结论合并成一句连贯、无重复、结论先行的中文结论，只输出结论本身。"},
                        {"role": "user", "content": "\n".join(c["conclusion"] for c in cards)},  # 将所有结论拼接后发给 LLM
                    ],
                    model=model_for("summarize"), temperature=0.2)  # 摘要任务用低 temperature 保证确定性
            trace.step("summarize", output_summary=gen.content[:160], duration_ms=t.ms, token_usage=gen.total_tokens)
            card = {
                "conclusion": gen.content.strip(),  # LLM 合并后的统一结论
                "evidence": [e for c in cards for e in c["evidence"]],  # 所有卡片证据的并集（列表推导式展开）
                "steps": [s for c in cards for s in c["steps"]],  # 所有卡片步骤的并集
            }
            draft = render_card(card)  # 将合并后的卡片渲染为 markdown
        else:
            draft = "这个问题我先帮你转接人工技术支持，请稍候。"  # 兜底：无任何 Agent 成功时，给出转人工提示
        return {"draft_answer": draft, "card": card}  # 返回草稿和结构化卡片（用于后续缓存和脱敏）

    async def security_node(state: AgentState) -> dict:
        """节点7（最终节点）：对 draft_answer 进行安全审查与脱敏，写入 final_answer。

        串行放在最后，保证所有输出都经过安全过滤后才暴露给用户。
        脱敏操作同时作用于文本和结构化卡片（card）。

        Args:
            state: 当前图状态，需包含 draft_answer 和 card 字段。
        Returns:
            含 final_answer（脱敏后文本）和 card（脱敏后卡片）的字典。
        """
        from app.guardrail import desensitize  # 局部导入脱敏模块（避免循环导入）
        with timer() as t:  # 计时：记录安全审查耗时
            res = security.review_output(state.get("draft_answer", ""))  # 调用安全审查，返回脱敏后的结果对象
            card = state.get("card")  # 获取结构化卡片（可能为 None）
            clean_card = desensitize.desensitize_obj(card) if card else None  # 若有卡片则对其中的敏感字段脱敏
        trace.step("security", output_summary=f"脱敏类型={res.sensitive_found}", duration_ms=t.ms)  # 记录脱敏类型到链路
        return {"final_answer": res.clean_text, "card": clean_card}  # 返回脱敏后的最终文本和卡片

    # 创建 LangGraph 状态图，绑定共享状态类型 AgentState
    g = StateGraph(AgentState)
    # 注册所有节点到图中
    g.add_node("load_context", load_context)  # 节点1：加载会话历史和实体
    g.add_node("intent", intent_node)  # 节点2：意图识别与路由决策
    g.add_node("clarify", clarify_node)  # 节点3：澄清追问（可选路径）
    g.add_node("specialists", specialists_node)  # 节点4：专业 Agent 并行处理
    g.add_node("ticket", ticket_node)  # 节点5：工单创建
    g.add_node("summarize", summarize_node)  # 节点6：多 Agent 结果汇总
    g.add_node("security", security_node)  # 节点7：安全审查与脱敏

    # 定义图的边（节点间的流转关系）
    g.add_edge(START, "load_context")  # 图起点 → 加载上下文
    g.add_edge("load_context", "intent")  # 加载上下文完成 → 意图识别
    g.add_conditional_edges("intent", after_intent, {"clarify": "clarify", "specialists": "specialists"})  # 条件路由：需澄清 → clarify，否则 → specialists
    g.add_edge("clarify", END)  # 澄清追问后直接结束本轮，等待用户回复
    g.add_edge("specialists", "ticket")  # 专业处理完成 → 工单节点（内部判断是否真正建单）
    g.add_edge("ticket", "summarize")  # 工单处理完成 → 摘要合并
    g.add_edge("summarize", "security")  # 摘要完成 → 安全审查
    g.add_edge("security", END)  # 安全审查后结束，输出 final_answer
    return g.compile()  # 编译并返回可执行的 CompiledGraph 对象


async def run(
    *, query: str, tenant_id: str, user_id: str, conversation_id: str, is_internal: bool = False
) -> dict:
    """执行一次完整编排，返回结果并持久化链路与记忆。

    本函数是系统对外的核心入口，封装了完整的请求处理流程：
    语义缓存查询 → 图编排执行 → 异常 fallback → 结果后处理 → 记忆持久化 → 缓存写入。

    Args:
        query: 用户输入的问题文本。
        tenant_id: 租户 ID，用于权限隔离和成本归因。
        user_id: 用户 ID，用于工单关联和审计。
        conversation_id: 会话 ID，用于多轮对话记忆关联。
        is_internal: 是否为内部（平台员工）调用，影响工具权限。
    Returns:
        包含 answer、intent、confidence、citations、card、need_human、
        ticket_id、trace_id、total_tokens、entities、need_clarify、from_cache 的结果字典。
    """
    trace = TraceCollector(tenant_id=tenant_id, conversation_id=conversation_id)  # 初始化链路追踪收集器
    ctx = ToolContext(tenant_id=tenant_id, trace_id=trace.trace_id, is_internal=is_internal)  # 构建工具上下文，含权限信息

    # 语义缓存：命中热点问题则跳过完整链路
    with timer() as ct:  # 计时：记录语义缓存查询耗时
        cached, query_emb = await semantic_cache.get(tenant_id, query)  # 查询语义缓存，同时返回查询向量（后续写缓存复用）
    if cached:
        # 语义缓存命中：记录 trace 后直接返回缓存结果，跳过完整 LangGraph 编排
        trace.step("semantic_cache", input_summary=query,
                   output_summary=f"命中 similarity={cached['similarity']}", duration_ms=ct.ms)
        await session.append_message(conversation_id, "user", query)  # 仍需记录用户消息到会话记忆
        await session.append_message(conversation_id, "assistant", cached["answer"])  # 记录缓存回复到会话记忆
        await trace.persist()  # 持久化链路（即使命中缓存也保留可观测性）
        return {
            "answer": cached["answer"], "intent": cached["intent"], "confidence": 1.0,  # 缓存命中置信度视为 1.0
            "citations": cached["citations"], "card": cached.get("card"),
            "need_human": False, "ticket_id": None,  # 缓存结果无需转人工或建单
            "trace_id": trace.trace_id, "total_tokens": 0, "entities": {},  # token 为 0（未调用 LLM）
            "need_clarify": False, "from_cache": True,  # 标记来自缓存
        }

    graph = build_graph(ctx, trace)  # 按本次请求的上下文动态构建 LangGraph 图

    # 初始化图状态，包含本次请求的所有输入信息
    init: AgentState = {
        "tenant_id": tenant_id, "user_id": user_id, "conversation_id": conversation_id,
        "is_internal": is_internal, "query": query,
    }
    try:
        final = await graph.ainvoke(init)  # 异步执行完整 LangGraph 编排，获取最终状态
    except Exception as e:  # noqa: BLE001  管线级兜底：规则回复 + 自动建单转人工
        # 整个图编排异常时的兜底处理：记录错误、自动建单、返回规则化回复
        trace.step("fallback", input_summary=query, status="error", error=f"{type(e).__name__}: {e}")
        tk = await ticket.create_from_context(
            query=query, intent="ticket", entities={}, ai_diagnosis="",
            evidence={"pipeline_error": f"{type(e).__name__}: {e}"}, ctx=ctx,  # 将异常信息作为证据写入工单
            user_id=user_id, conversation_id=conversation_id)
        await trace.persist()  # 持久化链路（含错误信息）
        return {
            "answer": f"{fallback.rule_reply(None)}（工单号 {tk.ticket_id}）",  # 规则回复 + 工单号，让用户有追踪依据
            "intent": None, "confidence": 0.0, "citations": [], "need_human": True,
            "ticket_id": tk.ticket_id, "trace_id": trace.trace_id,
            "total_tokens": trace.total_tokens, "entities": {}, "need_clarify": False,
        }

    # 建单友好提示统一在此拼接（最终状态稳定含 ticket_message，避免节点可见性时序问题）
    answer = final.get("final_answer", "")  # 取出经安全审查后的最终回复文本
    ticket_message = final.get("ticket_message")  # 取出工单节点生成的友好提示（若有）
    if ticket_message and ticket_message not in answer:
        answer = f"{answer}\n\n{ticket_message}"  # 将工单提示追加到回复末尾，避免重复拼接

    # 构建最终返回的结果字典
    result = {
        "answer": answer,  # 最终用户可见的回复文本
        "intent": final.get("intent"),  # 本次识别的意图
        "confidence": final.get("confidence"),  # 意图置信度
        "citations": final.get("rag_citations", []),  # 引用的文档列表
        "card": final.get("card"),  # 结构化卡片（前端可用于富文本展示）
        "need_human": final.get("need_human", False),  # 是否需要转人工
        "ticket_id": final.get("ticket_id"),  # 工单 ID（若已建单）
        "trace_id": trace.trace_id,  # 链路追踪 ID，用于日志排查
        "total_tokens": trace.total_tokens,  # 本次请求总 token 消耗
        "entities": final.get("entities", {}),  # 本次抽取的实体字典
        "need_clarify": final.get("need_clarify", False),  # 是否返回了追问（前端据此调整 UI）
        "from_cache": False,  # 本次未命中缓存
    }

    # 记忆：写入本轮对话
    await session.append_message(conversation_id, "user", query)  # 持久化用户消息到会话记忆
    await session.append_message(conversation_id, "assistant", result["answer"])  # 持久化 AI 回复到会话记忆
    # 持久化链路
    await trace.persist()  # 将完整链路数据写入持久化存储（如数据库/对象存储）
    # 成本统计
    await cost.record(tenant_id, conversation_id, "mixed", trace.total_tokens)  # 记录本次对话 token 成本，按租户统计
    # 语义缓存：仅缓存通用文档问答结果
    if (
        final.get("intent") == "doc_qa"  # 仅缓存文档问答类结果（此类问题最具重复性）
        and not final.get("need_clarify")  # 澄清追问不应缓存（问题未完整）
        and not final.get("need_human")  # 需转人工的结果不应缓存（每次情况不同）
        and result["answer"]  # 回复不为空才值得缓存
    ):
        await semantic_cache.put(tenant_id, query, result, query_emb)  # 将结果写入语义缓存，query_emb 为已计算好的向量

    return result  # 返回完整结果字典给调用方（如 API 路由层）
