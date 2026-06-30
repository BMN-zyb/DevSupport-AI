# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""混合检索：向量检索 + BM25 关键词检索 → RRF 融合。

本模块实现了 RAG 系统的召回阶段，采用混合检索策略提升召回质量：
1. 向量检索（Dense Retrieval）：使用 DashScope Embedding 将 query 向量化，
   通过 Milvus HNSW 索引执行近似最近邻检索，擅长语义相似匹配；
2. BM25 关键词检索（Sparse Retrieval）：基于 jieba 分词 + BM25Okapi 算法，
   对全量切片进行关键词频率匹配，擅长精确词汇命中；
3. RRF（Reciprocal Rank Fusion）融合：将两路检索的排名列表融合为统一的相关性分数，
   充分利用两种检索方式的互补优势。

错误码类问题支持对 error_code 标量字段精确过滤召回。
"""

import jieba  # 中文分词库，用于 BM25 预处理
from rank_bm25 import BM25Okapi  # BM25 经典变体，支持中文分词后的词袋模型检索

from app.llm import client  # LLM/Embedding 客户端，提供 embed_one 异步接口
from app.rag import store   # Milvus 向量存储模块，提供 all_chunks / search 接口

# 模块级全局 BM25 索引缓存，避免每次检索都重建（代价较高）
_bm25: BM25Okapi | None = None  # BM25 索引对象，None 表示尚未初始化
_corpus: list[dict] = []        # BM25 索引对应的语料库（与 Milvus 中的切片顺序一一对应）


def _tokenize(text: str) -> list[str]:
    """对文本进行中文分词，返回非空 token 列表。

    使用 jieba.lcut 精确模式分词，并转小写以消除大小写差异。
    过滤掉纯空白 token，避免噪声影响 BM25 计算。

    参数：
        text: 待分词的文本字符串
    返回：
        分词后的字符串列表（小写，已过滤空白）
    """
    return [t for t in jieba.lcut(text.lower()) if t.strip()]  # 分词 → 小写 → 过滤空白 token


def _ensure_bm25() -> None:
    """懒加载并缓存 BM25 索引（基于 Milvus 中全部切片）。

    采用懒加载策略：首次调用时从 Milvus 拉取全量切片并构建 BM25Okapi 索引，
    后续调用直接返回（使用缓存）。ingest 后需调用 reset_bm25() 使索引失效重建。
    """
    global _bm25, _corpus  # 声明操作的是模块级全局变量
    if _bm25 is not None:  # 索引已构建，直接返回，无需重建
        return
    _corpus = store.all_chunks()  # 从 Milvus 拉取全量切片，作为 BM25 语料库
    if _corpus:  # 语料库非空才构建索引（空库时 BM25Okapi 会报错）
        _bm25 = BM25Okapi([_tokenize(c["content"]) for c in _corpus])  # 对每个切片分词，构建 BM25 倒排索引


def reset_bm25() -> None:
    """ingest 后调用，强制重建 BM25。

    全量 ingest 更新了 Milvus 中的切片数据后，需调用此函数使已缓存的
    BM25 索引失效，下次检索时将重新从 Milvus 拉取最新语料并重建索引。
    """
    global _bm25, _corpus  # 声明操作模块级全局变量
    _bm25, _corpus = None, []  # 将索引和语料库重置为初始状态，触发下次调用时重建


def _rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion：输入多个有序 id 列表，输出融合分。

    RRF 公式：score(d) = Σ 1 / (k + rank(d))
    其中 k 是平滑超参数（默认 60），rank 从 0 开始计数。
    排名越靠前（rank 越小），贡献的分数越大。
    多路检索均命中同一文档，其分数累加，能有效提升融合后的排名。

    参数：
        rankings: 多个有序文档 ID（此处为 content 字符串）列表，每个列表代表一路检索结果
        k:        RRF 平滑超参数，默认 60（常见工程经验值）
    返回：
        dict，key 为文档 ID，value 为 RRF 融合分数（越大越相关）
    """
    scores: dict[str, float] = {}  # 存放每个文档的 RRF 融合分数
    for ranking in rankings:  # 遍历每一路检索结果列表
        for rank, doc_id in enumerate(ranking):  # 遍历该路结果，rank 从 0 开始
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)  # 累加该文档在此路的 RRF 贡献分
    return scores  # 返回所有文档的融合分字典


async def hybrid_search(
    query: str, top_k_each: int = 20, error_code: str | None = None
) -> tuple[list[dict], float]:
    """混合召回，返回 (候选片段, 最高向量余弦)。向量余弦用于更稳健的无命中判定。

    检索流程：
    1. 将 query 向量化，执行 Milvus 向量检索（支持 error_code 标量精确过滤）；
    2. 基于 BM25 对全量语料执行关键词检索；
    3. 以 content 为 key 合并两路结果，使用 RRF 融合排名；
    4. 返回按 RRF 分数降序排列的候选片段列表，及最高向量余弦分（用于置信度判断）。

    参数：
        query:       用户查询字符串
        top_k_each:  每路检索返回的最大候选数，默认 20
        error_code:  可选错误码，用于向量检索时的标量精确过滤
    返回：
        tuple：
            - candidates: 候选切片字典列表，每项含 rrf_score 及原始元数据字段
            - top_vec:    本次向量检索中最高的 COSINE 相似度分数（0~1）
    """
    _ensure_bm25()  # 确保 BM25 索引已构建（懒加载）

    # 1) 向量检索
    qvec = await client.embed_one(query)  # 异步调用 Embedding API，将 query 转为向量
    expr = f'error_code == "{error_code}"' if error_code else None  # 若有错误码，构造标量过滤表达式
    vec_hits = store.search(qvec, top_k=top_k_each, expr=expr)  # 带过滤的向量检索
    # 错误码精确过滤若无结果，退回全量向量检索
    if error_code and not vec_hits:  # 精确过滤无结果（该错误码无对应切片）
        vec_hits = store.search(qvec, top_k=top_k_each)  # 退回不带过滤的全量向量检索，保证有召回

    # 2) BM25 关键词检索
    bm25_hits: list[dict] = []  # 初始化 BM25 命中结果列表
    if _bm25 is not None and _corpus:  # 仅在 BM25 索引和语料库均已初始化时执行
        scores = _bm25.get_scores(_tokenize(query))  # 计算 query 分词后与每个切片的 BM25 相关性分数
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_each]  # 取分数最高的 top_k_each 个索引
        bm25_hits = [_corpus[i] for i in top_idx if scores[i] > 0]  # 过滤掉分数为 0 的无关切片

    # 用 content 作为去重/融合 key；vec_hits 在前，同片段优先保留向量检索的元信息
    by_content: dict[str, dict] = {}  # 以 content 为 key 的去重字典，保留第一次出现的元信息
    for h in vec_hits + bm25_hits:  # 合并两路结果，vec_hits 在前优先
        by_content.setdefault(h["content"], h)  # setdefault 只在 key 不存在时设值，实现去重且保留向量侧元信息

    vec_order = [h["content"] for h in vec_hits]    # 向量检索结果的有序 content 列表（RRF 输入）
    bm25_order = [h["content"] for h in bm25_hits]  # BM25 检索结果的有序 content 列表（RRF 输入）
    rrf = _rrf_fuse([vec_order, bm25_order])  # 对两路有序列表执行 RRF 融合，得到每个 content 的融合分

    candidates = []  # 存放最终融合排序的候选片段
    for content, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):  # 按 RRF 融合分降序排列
        item = dict(by_content[content])  # 复制对应切片的元信息字典（避免修改原始数据）
        item["rrf_score"] = score  # 将 RRF 融合分写入切片字典，供后续重排/压缩模块使用
        candidates.append(item)  # 加入候选列表

    top_vec = max((h["score"] for h in vec_hits), default=0.0)  # 取向量检索中最高相似度分；无结果时默认 0.0
    return candidates, top_vec  # 返回融合排序的候选片段列表和最高向量余弦分
