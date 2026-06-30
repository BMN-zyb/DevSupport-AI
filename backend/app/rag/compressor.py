# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""上下文压缩：在 token 预算内保留最相关片段，降低生成成本。

本模块实现了 RAG 系统的上下文压缩阶段（Context Compression Stage）：
重排（reranker）后已有 top_n 条高质量片段，但直接拼接到 prompt 中可能超出
token 预算，或包含与问题相关性仍然偏低的片段。本模块通过两个策略进一步精简：

策略（无需额外 LLM 调用，低成本）：
1. 丢弃 rerank 分数低于阈值的片段（弱相关噪声）。
2. 按相关性从高到低累加，直到达到字符预算上限。
"""

DEFAULT_BUDGET_CHARS = 1800  # 默认字符预算上限，约等于 ~600 个 token（按中文 3 字/token 估算），控制 prompt 长度
MIN_RERANK_SCORE = 0.05      # rerank 分数最低阈值；低于此值的片段视为弱相关噪声，直接丢弃


def compress(chunks: list[dict], budget_chars: int = DEFAULT_BUDGET_CHARS) -> list[dict]:
    """返回压缩后的片段列表（保序：按相关性）。

    压缩策略：
    1. 跳过 rerank_score < MIN_RERANK_SCORE 的片段（弱相关，对回答无益，反而可能干扰模型）；
    2. 按输入顺序（已按相关性降序）逐片段累加字符数，超过 budget_chars 时停止；
       例外：若已保留片段为空（第一个片段就超预算），仍保留该片段，确保至少有一条上下文。

    参数：
        chunks:       重排后的切片列表，按 rerank_score 降序排列
        budget_chars: 允许的最大总字符数，默认 DEFAULT_BUDGET_CHARS
    返回：
        压缩后的切片列表，长度 <= len(chunks)，总字符数 <= budget_chars（至少保留一条）
    """
    kept, used = [], 0  # kept: 已保留的片段列表；used: 已累积的字符数
    for c in chunks:  # 按相关性降序遍历每个片段
        if c.get("rerank_score", 1.0) < MIN_RERANK_SCORE:  # 无 rerank_score 字段时默认 1.0（通过阈值检查）
            continue  # 丢弃弱相关片段，降低生成时的噪声
        length = len(c["content"])  # 计算当前片段的字符长度
        if used + length > budget_chars and kept:  # 加入此片段会超出预算，且已有保留片段
            break  # 停止累加，保证总字符数在预算内
        kept.append(c)   # 将当前片段加入保留列表
        used += length   # 更新已累积字符数
    return kept  # 返回压缩后的片段列表


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """拼接上下文文本并返回引用清单。

    将压缩后的切片列表格式化为：
    - 带编号标签的文本块，拼接为 LLM prompt 的上下文部分；
    - 结构化引用清单，用于生成回答末尾的来源引用信息（供前端展示）。

    格式示例：
        [1] 《文档标题》- 章节标题
        切片内容文本

        [2] 《另一文档》- 另一章节
        切片内容文本

    参数：
        chunks: 压缩后的切片列表（来自 compress 函数的返回值）
    返回：
        tuple：
            - context_text: 拼接好的上下文字符串（各块以两个换行分隔）
            - citations:    引用清单列表，每项含 index/doc_title/section/version/score
    """
    blocks, citations = [], []  # blocks: 文本块列表；citations: 引用元信息列表
    for i, c in enumerate(chunks, 1):  # 从 1 开始编号，与引用标签 [1]、[2] 对应
        tag = f"[{i}] 《{c['doc_title']}》- {c['section']}"  # 构造带编号的来源标签
        blocks.append(f"{tag}\n{c['content']}")  # 将标签和内容拼接为一个文本块
        citations.append(
            {
                "index": i,                                      # 引用编号，对应正文中的 [i] 标记
                "doc_title": c["doc_title"],                     # 文档标题，供前端展示引用来源
                "section": c["section"],                         # 章节标题，精确定位到具体内容位置
                "version": c.get("version", "v1"),               # 文档版本，默认 v1（用于追踪知识库更新）
                "score": round(c.get("rerank_score", 0.0), 3),  # 保留 3 位小数的重排分数，方便调试与展示
            }
        )
    return "\n\n".join(blocks), citations  # 各文本块以双换行分隔，形成清晰的上下文段落
