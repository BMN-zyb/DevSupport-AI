# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Milvus 向量存储：knowledge_chunk collection 的管理与检索。

本模块封装了与 Milvus 向量数据库的所有交互，包括：
- collection 的创建、删除与模式定义；
- 批量插入向量化切片数据；
- 向量相似度检索（ANN）；
- 全量切片读取（用于构建 BM25 索引）。

字段：
- pk           自增主键
- embedding    FLOAT_VECTOR(dim)，HNSW 索引，COSINE 度量
- content      原文片段
- doc_title    文档标题（引用 + 标量过滤）
- section      章节
- category     文档分类
- error_code   错误码（错误码类问题标量精确匹配）
- version      文档版本
"""

from functools import lru_cache  # 函数级 LRU 缓存装饰器，用于单例化 Milvus 客户端

from pymilvus import DataType, MilvusClient  # Milvus 官方 Python SDK：数据类型枚举与轻量级客户端

from app.config import settings  # 全局配置对象，提供 milvus_uri / milvus_collection / embedding_dim 等

VECTOR_DIM = settings.embedding_dim    # 向量维度，与 Embedding 模型输出维度一致（如 1536）
COLLECTION = settings.milvus_collection  # Milvus collection 名称，来自配置文件


@lru_cache  # 利用 LRU 缓存确保整个进程内只创建一个 MilvusClient 实例（单例模式）
def get_client() -> MilvusClient:
    """获取（或复用缓存的）Milvus 客户端实例。

    使用 lru_cache 装饰，保证多次调用只初始化一次连接，避免连接开销与资源浪费。

    返回：
        MilvusClient 实例，已连接到 settings.milvus_uri 指定的服务
    """
    return MilvusClient(uri=settings.milvus_uri)  # 按配置的 URI 创建客户端（支持本地文件或远程服务）


def ensure_collection(recreate: bool = False) -> None:
    """创建 collection（含 HNSW 索引）。recreate=True 时先删除重建。

    collection 模式设计：
    - auto_id=True：主键 pk 由 Milvus 自增生成，无需业务侧管理；
    - embedding 字段使用 HNSW 索引，COSINE 度量，适合文本语义相似度检索；
    - 其余字段均为 VARCHAR 标量，支持结合 filter 表达式做元数据过滤。

    参数：
        recreate: True 表示先删除已有 collection 再重建（全量入库前使用）；
                  False 表示若 collection 已存在则直接返回（幂等）
    """
    client = get_client()  # 获取 Milvus 客户端
    if recreate and client.has_collection(COLLECTION):  # recreate 模式且 collection 已存在
        client.drop_collection(COLLECTION)  # 先删除旧 collection，清空所有数据
    if client.has_collection(COLLECTION):  # collection 已存在（且未 recreate），直接返回
        return

    # 定义 collection schema
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)  # 启用自增 ID，禁用动态字段
    schema.add_field("pk", DataType.INT64, is_primary=True)  # 自增整型主键
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)  # 浮点向量字段，维度由配置决定
    schema.add_field("content", DataType.VARCHAR, max_length=4000)    # 原文片段，最长 4000 字符
    schema.add_field("doc_title", DataType.VARCHAR, max_length=256)   # 文档标题，用于检索结果展示与过滤
    schema.add_field("section", DataType.VARCHAR, max_length=256)     # 章节标题，用于精细来源定位
    schema.add_field("category", DataType.VARCHAR, max_length=64)     # 文档分类（如"接入"、"鉴权"）
    schema.add_field("error_code", DataType.VARCHAR, max_length=64)   # 错误码，支持标量精确过滤
    schema.add_field("version", DataType.VARCHAR, max_length=32)      # 文档版本（如 "v1"）

    # 配置 HNSW 向量索引参数
    index_params = client.prepare_index_params()  # 初始化索引参数容器
    index_params.add_index(
        field_name="embedding",       # 对 embedding 字段建索引
        index_type="HNSW",            # 使用 HNSW（Hierarchical Navigable Small World）图索引，兼顾速度与精度
        metric_type="COSINE",         # 余弦相似度度量，适合文本语义检索
        params={"M": 16, "efConstruction": 200},  # M: 每层最大邻居数；efConstruction: 建图时探索宽度
    )
    client.create_collection(
        collection_name=COLLECTION, schema=schema, index_params=index_params  # 以定义好的 schema 和索引参数创建 collection
    )


def drop_collection() -> None:
    """删除 Milvus collection（若存在）。

    用于手动清空向量数据库，通常在测试或数据重置场景下调用。
    """
    client = get_client()  # 获取 Milvus 客户端
    if client.has_collection(COLLECTION):  # 仅在 collection 存在时才执行删除，避免报错
        client.drop_collection(COLLECTION)  # 删除 collection 及其所有数据


def insert(rows: list[dict]) -> int:
    """批量插入切片。rows 每项含 embedding/content/doc_title/section/category/error_code/version。

    插入后立即执行 flush，确保数据持久化并对后续查询可见（Milvus 默认异步落盘）。

    参数：
        rows: 字典列表，每个字典对应一条切片记录，字段与 collection schema 一致
    返回：
        实际插入的记录数（int）
    """
    client = get_client()  # 获取 Milvus 客户端
    res = client.insert(collection_name=COLLECTION, data=rows)  # 批量写入向量及元数据
    client.flush(COLLECTION)  # 强制 flush，将缓冲区数据落盘，保证后续查询能命中
    return res.get("insert_count", len(rows))  # 返回插入数量，res 中无该字段时退回 rows 长度


def count() -> int:
    """统计 collection 中的切片总数。

    若 collection 不存在则返回 0；否则 load collection 后执行聚合查询。

    返回：
        int，当前 collection 中的记录数；collection 不存在时返回 0
    """
    client = get_client()  # 获取 Milvus 客户端
    if not client.has_collection(COLLECTION):  # collection 尚未创建，直接返回 0
        return 0
    client.load_collection(COLLECTION)  # 将 collection 加载到内存，使其可查询
    res = client.query(COLLECTION, filter="pk >= 0", output_fields=["count(*)"])  # 聚合查询总记录数
    if res and "count(*)" in res[0]:  # 确认查询结果包含聚合字段
        return res[0]["count(*)"]  # 返回聚合值
    return 0  # 查询结果异常时返回 0


def all_chunks(limit: int = 2000) -> list[dict]:
    """取出全部切片（用于构建 BM25 关键词索引）。

    retriever 模块在初始化 BM25Okapi 时调用此函数获取全量语料。
    limit 设为 2000 以防 collection 过大时内存溢出。

    参数：
        limit: 最多返回的记录数，默认 2000
    返回：
        字典列表，每项含 content/doc_title/section/category/error_code/version 字段
    """
    client = get_client()  # 获取 Milvus 客户端
    if not client.has_collection(COLLECTION):  # collection 不存在则返回空列表
        return []
    client.load_collection(COLLECTION)  # 将 collection 加载到内存
    return client.query(
        COLLECTION,
        filter="pk >= 0",  # 过滤条件：匹配所有记录（pk 自增，恒 >= 0）
        output_fields=["content", "doc_title", "section", "category", "error_code", "version"],  # 只取文本/元数据字段，不取 embedding（节省带宽）
        limit=limit,  # 限制最大返回条数
    )


def search(
    query_vector: list[float], top_k: int = 20, expr: str | None = None
) -> list[dict]:
    """向量检索，返回片段 + 相似度分数（COSINE，越大越相似）。

    基于 HNSW 索引执行近似最近邻（ANN）搜索，支持结合标量过滤表达式缩小候选范围。

    参数：
        query_vector: 查询文本的向量表示（维度需与 collection 一致）
        top_k:        返回最相似的 top_k 条结果，默认 20
        expr:         可选的标量过滤表达式，如 'error_code == "INVALID_PARAM"'
    返回：
        字典列表，每项包含：
            - score:      COSINE 相似度分数（0~1，越大越相似）
            - content:    切片文本
            - doc_title:  文档标题
            - section:    章节标题
            - category:   文档分类
            - error_code: 错误码（无则空字符串）
            - version:    文档版本
    """
    client = get_client()  # 获取 Milvus 客户端
    client.load_collection(COLLECTION)  # 确保 collection 已加载到内存，可执行向量检索
    results = client.search(
        collection_name=COLLECTION,
        data=[query_vector],  # 传入查询向量列表（支持批量，此处只查一条）
        limit=top_k,          # 返回最相似的 top_k 条
        filter=expr or "",    # 标量过滤表达式；None 转为空字符串表示不过滤
        output_fields=["content", "doc_title", "section", "category", "error_code", "version"],  # 返回所需元数据字段
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},  # ef: 查询时探索宽度，影响精度与速度的权衡
    )
    hits = []  # 存放格式化后的检索结果
    for hit in results[0]:  # results[0] 对应第一条（也是唯一一条）查询向量的结果列表
        entity = hit.get("entity", {})  # 获取命中记录的字段值字典
        hits.append(
            {
                "score": float(hit.get("distance", 0.0)),  # COSINE 距离即相似度分数，转为 Python float
                "content": entity.get("content", ""),      # 切片文本内容
                "doc_title": entity.get("doc_title", ""),  # 文档标题
                "section": entity.get("section", ""),      # 章节标题
                "category": entity.get("category", ""),    # 文档分类
                "error_code": entity.get("error_code", ""),  # 错误码（无则空字符串）
                "version": entity.get("version", ""),      # 文档版本
            }
        )
    return hits  # 返回所有命中结果，按相似度降序排列（Milvus 默认行为）
