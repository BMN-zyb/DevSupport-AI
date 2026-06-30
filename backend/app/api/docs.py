# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""文档中心接口：向已登录用户提供知识库原始资料的浏览能力。

资料来源为项目根目录 data/knowledge/*.md（与 RAG 检索同一份源）。

本模块在 DevSupport-AI 项目中充当"文档中心"路由层，提供如下能力：
  - GET /api/docs          ：列出知识库目录下所有 Markdown 文档的元信息列表
  - GET /api/docs/{doc_id} ：读取指定文档的元信息与全文内容

所有接口均要求用户已登录（通过 get_current_user 依赖验证 JWT），保证知识库内容不对外公开。
知识库文件命名约定：<前缀ID>-<描述>.md，首行为 Markdown 一级标题（# 标题文字）。
"""

# ── FastAPI 核心依赖 ──────────────────────────────────────────────────────────
from fastapi import APIRouter, Depends, HTTPException  # APIRouter 分组路由；Depends 注入依赖；HTTPException 抛出 HTTP 错误

# ── 项目内部模块 ──────────────────────────────────────────────────────────────
from app.deps import CurrentUser, get_current_user  # CurrentUser：解码 JWT 后的当前用户数据类；get_current_user：验证并提取当前用户的依赖（接口鉴权使用）
from app.rag.ingest import CATEGORY_MAP, KNOWLEDGE_DIR  # KNOWLEDGE_DIR：知识库文件所在 Path 对象；CATEGORY_MAP：文件名前缀到分类名称的映射字典

# ── 路由器声明 ────────────────────────────────────────────────────────────────
# prefix 使本模块所有路由自动带 /api/docs 前缀；tags 用于 OpenAPI 文档界面中分组展示
router = APIRouter(prefix="/api/docs", tags=["docs"])


def _meta(path):
    """从单个 Markdown 文件路径中提取文档元信息（私有辅助函数）。

    命名规则：文件名格式为 "<前缀ID>-<描述>.md"，首行为一级 Markdown 标题。
    此函数被 list_docs 和 get_doc 复用，避免重复解析逻辑。

    参数：
        path (pathlib.Path): 知识库中某个 .md 文件的绝对路径对象。

    返回：
        dict: 包含以下字段的元信息字典：
            - id       (str): 文件名前缀（连字符前的部分），如 "api-timeout" -> "api"
            - title    (str): 文档标题，从首行 Markdown 标题提取；无标题则用文件 stem 代替
            - category (str): 分类名称，由 CATEGORY_MAP 根据前缀 ID 映射；未匹配则为"其它"
            - filename (str): 原始文件名（含扩展名），供前端展示或下载时使用
    """
    stem = path.stem  # 取不含扩展名的文件名，如 "api-timeout-kb" -> "api-timeout-kb"
    prefix = stem.split("-")[0]  # 按第一个连字符切割，取前缀作为文档 ID，如 "api"
    # 若文件非空则读取内容并取第一行；若文件为空则用文件 stem 作为标题占位符
    first = path.read_text(encoding="utf-8").splitlines()[0] if path.stat().st_size else stem
    # 若首行是 Markdown 一级标题（以 "# " 开头），则去掉前两字符并去空白得到纯文本标题
    title = first[2:].strip() if first.startswith("# ") else stem
    return {
        "id": prefix,  # 文档唯一标识符（文件名前缀），用于 GET /api/docs/{doc_id} 路由参数
        "title": title,  # 文档可读标题，来自 Markdown 首行一级标题
        "category": CATEGORY_MAP.get(prefix, "其它"),  # 按前缀在分类映射表中查找分类；找不到时归入"其它"
        "filename": path.name,  # 完整文件名（含 .md 后缀），供前端展示或追溯文件来源
    }


@router.get("")
async def list_docs(user: CurrentUser = Depends(get_current_user)) -> dict:
    """列出知识库目录下所有 Markdown 文档的元信息。

    接口需要已登录用户，通过 get_current_user 依赖自动校验 JWT Token。
    结果按文件名字典序排列，便于前端稳定渲染列表顺序。

    参数：
        user (CurrentUser): 由依赖注入的当前登录用户，仅用于鉴权，不参与业务逻辑。

    返回：
        dict: {"documents": [<meta>, ...]}，每个 meta 由 _meta() 生成，包含 id/title/category/filename。
    """
    # 使用 pathlib.glob 匹配 KNOWLEDGE_DIR 下所有 .md 文件，sorted 按文件名字典序排序保证列表稳定
    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    # 对每个文件路径调用 _meta 提取元信息，组成列表返回
    return {"documents": [_meta(f) for f in files]}


@router.get("/{doc_id}")
async def get_doc(doc_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """获取指定 doc_id 对应文档的元信息与全文内容。

    接口需要已登录用户，通过 get_current_user 依赖自动校验 JWT Token。
    doc_id 对应文件名中的前缀部分（连字符前），通过 glob 模式 "<doc_id>-*.md" 匹配文件。

    参数：
        doc_id (str)        : 文档前缀 ID，如 "api"（对应文件名形如 api-timeout.md）。
        user   (CurrentUser): 由依赖注入的当前登录用户，仅用于鉴权，不参与业务逻辑。

    返回：
        dict: 文档元信息（id/title/category/filename）加上 content（文档全文 Markdown 字符串）。

    异常：
        HTTPException(404): 在 KNOWLEDGE_DIR 中找不到匹配 "<doc_id>-*.md" 的文件时抛出。
    """
    # 用 glob 模式 "<doc_id>-*.md" 匹配该前缀对应的文档文件（list 化以便多次访问）
    matches = list(KNOWLEDGE_DIR.glob(f"{doc_id}-*.md"))
    if not matches:
        # 无匹配文件时以 404 响应，提示前端文档不存在
        raise HTTPException(404, "文档不存在")
    path = matches[0]  # 若存在多个匹配文件，取第一个（按 glob 默认排序）
    meta = _meta(path)  # 提取文档元信息（id/title/category/filename）
    meta["content"] = path.read_text(encoding="utf-8")  # 读取文件全文内容（UTF-8 编码），附加到元信息字典中返回给前端
    return meta
