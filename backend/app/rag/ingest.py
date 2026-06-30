# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""知识库 ingest：读取 markdown → 切片 → 向量化 → 写入 Milvus，并登记文档元信息。

本模块负责将 data/knowledge/ 目录下的 Markdown 文档全量入库：
1. 扫描所有 .md 文件，根据文件名前缀识别文档分类；
2. 按 ## 章节切片，超长章节再做滑动窗口二次切分，保留 overlap 保证语义完整；
3. 批量调用 DashScope Embedding API 将文本片段向量化；
4. 将向量及元信息写入 Milvus 向量数据库；
5. 在 MySQL 中登记或更新文档元信息（KnowledgeDocument 表）。
"""

import re  # 正则表达式库，用于解析错误码标题
from pathlib import Path  # 面向对象的文件系统路径操作

from app.db import SyncSessionLocal  # 同步 SQLAlchemy session，用于写 MySQL
from app.llm import client  # LLM/Embedding 客户端（封装了 DashScope 调用）
from app.models import KnowledgeDocument  # SQLAlchemy ORM 模型，对应 MySQL knowledge_documents 表
from app.rag import store  # Milvus 向量存储模块，提供 ensure_collection / insert 等接口

# 知识库原始资料位于项目根目录 data/knowledge（与后端代码分离，便于运营维护与用户查看）
# ingest.py: parents[0]=rag, [1]=app, [2]=backend, [3]=项目根
KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "data" / "knowledge"  # 知识库 Markdown 文件目录

# 文件名前缀到分类名称的映射表，用于自动识别文档所属业务分类
CATEGORY_MAP = {
    "01": "接入",      # 01-xxx.md 归属"接入"分类
    "02": "鉴权",      # 02-xxx.md 归属"鉴权"分类
    "03": "错误码",    # 03-xxx.md 归属"错误码"分类
    "04": "回调",      # 04-xxx.md 归属"回调"分类
    "05": "限流",      # 05-xxx.md 归属"限流"分类
    "06": "计费",      # 06-xxx.md 归属"计费"分类
    "07": "数据质量",  # 07-xxx.md 归属"数据质量"分类
    "08": "FAQ",       # 08-xxx.md 归属"FAQ"分类
}

# 用于从章节标题中提取错误码的正则：匹配以大写字母开头、由大写字母和下划线组成的词，后跟中文左括号
ERROR_CODE_RE = re.compile(r"^([A-Z][A-Z_]+)（")
MAX_CHARS = 600   # 单个切片的最大字符数，超过此长度才做二次切分
OVERLAP = 80      # 滑动窗口切分时相邻片段的重叠字符数，保证跨片段语义连续性


def _category(filename: str) -> str:
    """根据文件名前缀（如 "01"、"02"）查询 CATEGORY_MAP，返回对应分类名称。

    参数：
        filename: 文件名字符串，如 "01-接入指南.md"
    返回：
        分类名称字符串；若前缀不在映射表中则返回 "其它"
    """
    prefix = filename.split("-")[0]  # 取文件名第一个 "-" 之前的部分作为前缀
    return CATEGORY_MAP.get(prefix, "其它")  # 在映射表中查找，未匹配时默认返回"其它"


def _split_long(text: str) -> list[str]:
    """长段落按字符窗口切分，带 overlap。

    若文本长度未超过 MAX_CHARS，直接返回原文本；
    否则按滑动窗口切分，相邻窗口重叠 OVERLAP 个字符，避免语义在边界处断裂。

    参数：
        text: 待切分的文本字符串
    返回：
        切分后的字符串列表，每项长度 <= MAX_CHARS
    """
    if len(text) <= MAX_CHARS:  # 文本未超过单片最大字符数，无需切分
        return [text]
    chunks, start = [], 0  # chunks 存储结果，start 为当前窗口起始位置
    while start < len(text):  # 逐步向后滑动直到覆盖全文
        end = min(start + MAX_CHARS, len(text))  # 当前窗口终止位置，不超过文本末尾
        chunks.append(text[start:end])  # 截取当前窗口内容并加入结果列表
        if end == len(text):  # 已到达文本末尾，退出循环
            break
        start = end - OVERLAP  # 下一个窗口起始 = 当前窗口末尾回退 OVERLAP 个字符
    return chunks


def chunk_markdown(md: str, doc_title: str) -> list[dict]:
    """按 ## 章节切片，长章节再按窗口切分。

    解析 Markdown 文本结构：
    - H1 标题（# ）作为文档标题，已在调用方提取，此处跳过；
    - H2 标题（## ）作为章节分隔符，每个 H2 下的内容合并为一个 section；
    - 每个 section 内容若超过 MAX_CHARS，再调用 _split_long 二次切分；
    - 同时尝试从 section 标题中解析错误码（如 "INVALID_PARAM（..."）。

    参数：
        md:        完整的 Markdown 文本内容
        doc_title: 文档标题（从 H1 或文件名提取），作为无 H2 章节的默认 section title
    返回：
        list[dict]，每项包含：
            - section:    章节标题字符串
            - content:    该切片的正文文本
            - error_code: 若标题含错误码则为错误码字符串，否则为空字符串
    """
    lines = md.splitlines()  # 将 Markdown 按行拆分，逐行处理
    sections: list[tuple[str, list[str]]] = []  # 存放 (章节标题, 行列表) 的列表
    current_title, buf = doc_title, []  # 初始化当前章节标题为文档标题，行缓冲区为空
    for line in lines:  # 逐行扫描
        if line.startswith("## "):  # 遇到 H2 标题，表示新章节开始
            if buf:  # 若缓冲区非空，先保存上一章节
                sections.append((current_title, buf))
            current_title, buf = line[3:].strip(), [line]  # 更新章节标题，开始新缓冲区
        elif line.startswith("# "):
            continue  # H1 作为文档标题，已单独处理，此处跳过
        else:
            buf.append(line)  # 普通行追加到当前章节缓冲区
    if buf:  # 文件末尾最后一个章节的内容
        sections.append((current_title, buf))

    chunks = []  # 最终切片结果列表
    for section_title, body_lines in sections:  # 遍历每个章节
        body = "\n".join(body_lines).strip()  # 将行列表合并为文本，并去除首尾空白
        if not body:  # 跳过内容为空的章节
            continue
        err_match = ERROR_CODE_RE.match(section_title)  # 尝试从章节标题解析错误码
        error_code = err_match.group(1) if err_match else ""  # 有匹配则取捕获组，否则空字符串
        for piece in _split_long(body):  # 对章节正文做（可能的）二次切分
            chunks.append(
                {"section": section_title, "content": piece.strip(), "error_code": error_code}
            )  # 构建切片字典并追加到结果
    return chunks


async def ingest_all(recreate: bool = True) -> dict:
    """全量 ingest：重建 collection 并写入所有文档切片。返回统计。

    执行流程：
    1. 调用 store.ensure_collection 确保 Milvus collection 存在（recreate=True 时先删再建）；
    2. 扫描 KNOWLEDGE_DIR 下所有 .md 文件（按文件名排序）；
    3. 对每个文件：解析标题与分类 → 切片 → 批量向量化 → 写入 Milvus；
    4. 将文档元信息（标题、分类、路径、切片数）写入 MySQL knowledge_documents 表。

    参数：
        recreate: 是否先删除再重建 Milvus collection，默认 True（全量重建）
    返回：
        dict，包含 "documents"（处理文件数）和 "chunks"（总切片数）
    """
    store.ensure_collection(recreate=recreate)  # 初始化/重建 Milvus collection
    files = sorted(KNOWLEDGE_DIR.glob("*.md"))  # 按文件名字典序扫描所有 Markdown 文件
    total_chunks = 0  # 记录入库的总切片数
    doc_records = []  # 存放待写入 MySQL 的文档元信息列表

    for f in files:  # 逐文件处理
        md = f.read_text(encoding="utf-8")  # 读取 Markdown 文件内容（UTF-8 编码）
        first_line = md.splitlines()[0] if md else f.stem  # 取文件第一行（非空文件）或文件名作为候备
        doc_title = first_line[2:].strip() if first_line.startswith("# ") else f.stem  # 提取 H1 标题，否则用文件名
        category = _category(f.name)  # 根据文件名前缀获取文档分类
        chunks = chunk_markdown(md, doc_title)  # 将 Markdown 切分为结构化片段列表

        # 批量向量化（DashScope 单次上限保守取 10）
        contents = [c["content"] for c in chunks]  # 提取所有切片的文本内容
        embeddings = []  # 存放所有切片的向量
        for i in range(0, len(contents), 10):  # 每次最多处理 10 条，避免超出 API 单批限制
            embeddings.extend(await client.embed(contents[i : i + 10]))  # 异步批量请求 Embedding，累积结果

        rows = []  # 准备写入 Milvus 的行列表
        for c, emb in zip(chunks, embeddings):  # 将切片元信息与对应向量配对
            rows.append(
                {
                    "embedding": emb,          # 文本向量，维度由 settings.embedding_dim 决定
                    "content": c["content"],   # 原始文本片段
                    "doc_title": doc_title,    # 所属文档标题
                    "section": c["section"],   # 所属章节标题
                    "category": category,      # 文档分类（如"接入"、"鉴权"）
                    "error_code": c["error_code"],  # 错误码（无则空字符串）
                    "version": "v1",           # 文档版本，当前固定为 v1
                }
            )
        store.insert(rows)  # 批量写入 Milvus collection
        total_chunks += len(rows)  # 累加切片总数
        doc_records.append(
            {
                "id": f"doc_{f.stem.split('-')[0]}",  # 文档 ID 由文件名前缀构成，如 "doc_01"
                "title": doc_title,                   # 文档标题
                "category": category,                 # 文档分类
                "source_path": str(f.relative_to(KNOWLEDGE_DIR.parent.parent)),  # 相对于 data/ 的路径
                "chunk_count": len(rows),             # 该文档的切片数量
            }
        )
        print(f"[ingest] {f.name} -> {len(rows)} chunks (category={category})")  # 打印进度日志

    # 登记文档元信息到 MySQL
    with SyncSessionLocal() as s:  # 使用同步 session 操作 MySQL，with 块结束自动关闭
        for rec in doc_records:  # 逐条处理文档元信息
            existing = s.get(KnowledgeDocument, rec["id"])  # 按主键查询是否已存在记录
            if existing:  # 若已存在，更新可变字段（幂等操作，支持重复 ingest）
                existing.chunk_count = rec["chunk_count"]    # 更新切片数量
                existing.title = rec["title"]                # 更新文档标题（可能已改名）
                existing.category = rec["category"]          # 更新分类
                existing.source_path = rec["source_path"]    # 更新来源路径
            else:  # 若不存在，插入新记录
                s.add(KnowledgeDocument(status="published", version="v1", **rec))  # 初始状态为已发布
        s.commit()  # 提交事务，持久化所有变更

    return {"documents": len(files), "chunks": total_chunks}  # 返回入库统计
