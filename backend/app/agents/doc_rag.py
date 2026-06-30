# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Doc RAG Agent：检索增强问答。

流程：Query 改写 → 混合检索 → Rerank → 上下文压缩 → 带引用生成 → 无命中兜底。

本模块是 DevSupport-AI 多智能体系统中的文档检索增强（RAG）专职子智能体。
职责：
  1. Query 改写：结合对话历史，将含指代/省略的问题补全为独立完整的检索查询。
  2. 混合检索：同时执行向量检索（语义相似度）与关键词检索，取二者并集，扩大召回。
  3. Rerank 精排：对召回的候选文档进行交叉编码器重排序，提升精度。
  4. 无命中判定：综合向量余弦得分与 Rerank 分数双信号，判断是否真正命中相关文档。
  5. 上下文压缩：对精排结果裁剪、拼接，构建紧凑的上下文与引用列表。
  6. 带引用生成：将上下文交给 LLM，生成结构化 JSON 卡片回答，附带文档引用。
  7. 无命中兜底：未找到相关文档时返回标准拒答语，引导用户转人工。

被其他 Agent（如 api_diagnostic、billing）调用，也可直接接收用户的通用文档问答请求。
"""

# -----------------------------------------------------------------------
# 标准库导入
# -----------------------------------------------------------------------
# dataclass：定义轻量数据容器，避免手写样板代码
from dataclasses import dataclass, field

# -----------------------------------------------------------------------
# 项目内部模块导入
# -----------------------------------------------------------------------
from app.agents.util import normalize_card, parse_json, render_card  # 卡片规范化、JSON 解析、卡片渲染工具
from app.config import settings  # 全局配置，含 rag_vec_hit_threshold、rag_score_threshold 等阈值
from app.llm import client  # LLM 客户端，封装向大模型发请求的接口
from app.llm.router import model_for  # 模型路由函数，根据场景名称返回对应模型配置
from app.rag import compressor, reranker, retriever  # RAG 三大组件：上下文压缩器、重排序器、检索器

# 无命中兜底回复：当检索结果无相关文档时返回此消息，避免 LLM 编造错误答案
# 明确告知用户无法确定回答，并主动提供工单创建引导，提升用户体验
NO_HIT_MESSAGE = (
    "抱歉，我在现有文档中没有找到能确定回答这个问题的依据。"
    "为避免给你错误信息，建议转人工技术支持进一步确认，我可以帮你创建工单。"
)


@dataclass
class RagResult:
    """RAG 问答结果数据容器，由 answer() 函数填充并返回给调用方。

    Attributes:
        answer:      格式化后的回答文本（命中时为 LLM 生成的卡片，未命中时为兜底消息）。
        hit:         是否检索命中（True=找到相关文档并生成回答，False=无命中/兜底）。
        citations:   文档引用列表，每项含 doc_title/section/score 等字段（未命中时为空）。
        candidates:  混合检索返回的候选文档总数，供调试分析召回情况。
        top_score:   Rerank 后第一名的得分，用于评估检索质量。
        tokens:      本次生成消耗的 LLM token 数（仅计生成阶段，改写阶段不计入）。
        card:        LLM 输出的结构化 JSON 卡片原始数据（未命中时为 None）。
    """
    answer: str                                      # 回答文本（命中：卡片渲染结果；未命中：兜底消息）
    hit: bool                                        # 是否检索命中相关文档
    citations: list[dict] = field(default_factory=list)  # 文档引用列表（默认空列表）
    candidates: int = 0                              # 混合检索召回的候选文档总数
    top_score: float = 0.0                           # Rerank 第一名得分（0.0~1.0）
    tokens: int = 0                                  # 生成阶段 LLM token 消耗量
    card: dict | None = None                         # 结构化 JSON 卡片原始对象（未命中为 None）


async def _rewrite_query(query: str, history: list[dict] | None) -> str:
    """结合历史把指代/省略补全为独立查询；无历史则原样返回。

    例如：
      历史："如何申请 API Key？"
      当前问题："那如果过期了怎么办？"
      改写结果："API Key 过期后如何续期或重新申请？"

    改写目的是让检索查询不依赖对话上下文，确保向量检索与关键词检索语义完整。
    若无历史（单轮问答场景），跳过 LLM 调用直接返回原始查询，避免无意义开销。

    Args:
        query:   用户最新问题文本（可能含指代词或省略主语）。
        history: 对话历史列表，每项为 {"role": "user"/"assistant", "content": "..."}；
                 无历史时传 None 或空列表。

    Returns:
        改写后的完整独立查询字符串。若 LLM 返回为空，回退到原始 query。
    """
    if not history:
        return query  # 无历史则无需改写，直接返回原始问题节省 token

    # 取最近 4 条历史记录（2轮对话）拼接为文本，避免上下文过长
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
    # 构造改写 prompt：要求 LLM 只输出改写后的查询本身（无额外解释），便于直接作为检索输入
    prompt = [
        {"role": "system", "content": "你是检索查询改写器。根据对话历史，把用户最新问题改写为一句不依赖上下文的完整检索查询，只输出改写后的查询本身。"},
        {"role": "user", "content": f"对话历史：\n{hist_text}\n\n最新问题：{query}\n\n改写后的查询："},
    ]
    # 使用 intent 模型（通常为轻量快速模型）进行改写，temperature=0.0 确保输出确定性
    r = await client.chat(prompt, model=model_for("intent"), temperature=0.0)
    rewritten = r.content.strip()  # 去除首尾空白字符
    return rewritten or query  # LLM 若返回空字符串则回退到原始查询（防御性编程）


async def answer(
    query: str,
    *,
    history: list[dict] | None = None,
    error_code: str | None = None,
    top_n: int = 5,
) -> RagResult:
    """执行完整 RAG 流程，返回带引用的结构化问答结果。

    完整 RAG 流程（6个阶段）：
      1. Query 改写：补全指代/省略，生成独立检索查询。
      2. 混合检索：向量 + 关键词并行召回，扩大覆盖。
      3. Rerank 精排：交叉编码器对候选结果重排序，取 top_n 条。
      4. 无命中判定：向量余弦分 OR Rerank 分双信号判断是否真正命中。
      5. 上下文压缩：对精排结果裁剪、拼接上下文，构建引用列表。
      6. 带引用生成：LLM 基于压缩上下文生成 JSON 卡片回答。

    Args:
        query:      用户原始问题文本（或其他 Agent 构造的查询文本）。
        history:    对话历史列表（可选），用于 Query 改写中的指代消解。
        error_code: 错误码字符串（可选），传给检索器辅助过滤相关文档片段。
        top_n:      Rerank 后保留的最优候选文档数量，默认 5 条。

    Returns:
        RagResult 实例，命中时含生成回答、引用、得分等；未命中时返回兜底消息。
    """
    # 1. Query 改写：利用对话历史消解指代，生成可独立检索的查询字符串
    search_query = await _rewrite_query(query, history)

    # 2. 混合检索：同时执行向量检索（语义）与关键词检索（精确匹配），合并去重后返回候选列表
    # top_vec 为向量检索中第一名的余弦相似度得分，用于后续无命中判断
    candidates, top_vec = await retriever.hybrid_search(search_query, error_code=error_code)
    if not candidates:
        return RagResult(answer=NO_HIT_MESSAGE, hit=False)  # 无任何候选文档，直接返回兜底消息

    # 3. Rerank 精排：用交叉编码器对检索结果重排序，提升相关性排名，取 top_n 条最优结果
    reranked = await reranker.rerank_candidates(search_query, candidates, top_n=top_n)
    top_score = reranked[0]["rerank_score"] if reranked else 0.0  # 取第一名 Rerank 得分，无结果则为 0.0

    # 4. 无命中判定：rerank 绝对分对换述相关问题不稳，故用"向量余弦 OR rerank"双信号。
    #    覆盖到的问题向量余弦显著高于无关问题；rerank 高分也直接判命中。
    # 双信号设计：单一信号可能在某些问题类型上不稳定，两个信号互补提高召回准确性
    hit = bool(reranked) and (top_vec >= settings.rag_vec_hit_threshold or top_score >= settings.rag_score_threshold)
    if not hit:
        # 未命中：返回兜底消息并携带调试信息（候选数量和最高得分，便于排查阈值设置）
        return RagResult(
            answer=NO_HIT_MESSAGE, hit=False, candidates=len(candidates), top_score=top_score
        )

    # 5. 上下文压缩 + 引用：对精排结果截断/去冗余，构建紧凑上下文和结构化引用列表
    kept = compressor.compress(reranked)           # 压缩/筛选精排结果，去除冗余内容
    context, citations = compressor.build_context(kept)  # 拼接上下文文本并生成引用列表

    # 6. 带引用生成（结构化卡片）：LLM 基于压缩上下文生成 JSON 格式回答
    # System prompt：严格限定 LLM 只能基于参考资料回答，禁止编造，提升回答可信度
    sys_prompt = (
        "你是 API 平台的技术支持助手。只能基于【参考资料】回答，不得编造文档之外的接口能力。"
        "若资料不足以回答，conclusion 中明确说明不确定。涉及费用/合同以后台数据和正式合同为准。\n"
        "输出 JSON：{\"conclusion\":\"直接回答/结论\", \"steps\":[\"关键说明或可执行步骤\"]}。只输出 JSON。"
    )
    # User prompt：将压缩后的上下文和用户原始问题（非改写后的查询）拼接传给 LLM
    # 注意：生成阶段使用原始 query（更接近用户意图），检索阶段才用 search_query
    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query}"
    # 调用 LLM 生成回答（temperature=0.2 平衡语言流畅性与事实准确性）
    gen = await client.chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        model=model_for("rag_generate"),  # 路由到 RAG 生成场景对应的模型（通常为高质量模型）
        temperature=0.2,                  # 低温度确保引用文档内容的准确性，减少幻觉
    )
    card = normalize_card(parse_json(gen.content))  # 解析并规范化 LLM 输出的 JSON 卡片

    # 兜底处理：若 LLM 未能生成有效结论（JSON 解析失败），截取原始输出作为 conclusion
    if not card["conclusion"]:
        card["conclusion"] = gen.content.strip()[:300]  # 截取前 300 字符（比诊断场景稍长，适应详细解释）

    # 组装最终 RagResult 返回
    return RagResult(
        answer=render_card(card),       # 将卡片渲染为格式化文本展示给用户
        hit=True,                       # 标记命中，区别于兜底 NO_HIT_MESSAGE 场景
        citations=citations,            # 文档引用列表（来自 compressor.build_context）
        candidates=len(candidates),     # 混合检索召回的原始候选文档总数（调试用）
        top_score=top_score,            # Rerank 第一名得分（质量评估用）
        tokens=gen.total_tokens,        # 生成阶段 token 消耗（改写阶段 token 未计入）
        card=card,                      # 结构化 JSON 卡片原始对象（供上层进一步处理）
    )
