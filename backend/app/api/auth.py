# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""认证接口：登录、当前用户。

本模块是 DevSupport-AI 项目的认证路由层，负责处理用户身份验证相关的 HTTP 请求：
  - POST /api/auth/login  ：接收用户名/密码，验证通过后签发 JWT Access Token
  - GET  /api/auth/me     ：返回已登录用户的当前身份信息（需携带有效 Token）

设计要点：
  1. 采用异步数据库查询，避免阻塞 FastAPI 事件循环。
  2. 登录失败时对"用户不存在"与"密码错误"返回相同的 401 响应，
     防止攻击者通过响应差异枚举有效账号。
  3. JWT 中内嵌 tenant_id 和 role，后续业务接口凭此实现多租户隔离与权限管控。
"""

# ── FastAPI 核心依赖 ──────────────────────────────────────────────────────────
from fastapi import APIRouter, Depends, HTTPException, status  # APIRouter 用于分组注册路由；Depends 注入依赖；HTTPException 抛出 HTTP 错误；status 提供标准状态码常量

# ── SQLAlchemy 查询工具 ───────────────────────────────────────────────────────
from sqlalchemy import select  # 构造 SELECT 语句的声明式 API
from sqlalchemy.ext.asyncio import AsyncSession  # 异步数据库会话，配合 await 使用

# ── 项目内部模块 ──────────────────────────────────────────────────────────────
from app.db import get_db  # 获取异步数据库会话的依赖函数（FastAPI Depends 使用）
from app.deps import CurrentUser, get_current_user  # CurrentUser：解码 JWT 后的当前用户数据类；get_current_user：验证并提取当前用户的依赖
from app.models import Tenant, User  # ORM 模型：User 表示用户记录，Tenant 表示租户记录
from app.schemas.auth import LoginRequest, TokenResponse, UserInfo  # Pydantic 请求/响应 Schema：LoginRequest 接收登录表单，TokenResponse 返回 token 与用户信息，UserInfo 表示用户身份数据
from app.security import create_access_token, verify_password  # 安全工具：create_access_token 签发 JWT，verify_password 对比明文密码与哈希

# ── 路由器声明 ────────────────────────────────────────────────────────────────
# prefix 使本模块所有路由自动拥有 /api/auth 前缀；tags 用于 OpenAPI 文档分组展示
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    """用户登录接口，验证凭据后返回 JWT Token 与用户信息。

    参数：
        body (LoginRequest): 请求体，包含 username（用户名）和 password（明文密码）。
        db   (AsyncSession): 由 FastAPI 依赖注入的异步数据库会话。

    返回：
        TokenResponse: 包含 access_token（JWT 字符串）和 user（UserInfo 对象）。

    异常：
        HTTPException(401): 用户名不存在或密码不匹配时统一返回，不区分两种错误以防账号枚举。
    """
    # 用 SQLAlchemy 异步 execute 在 User 表中按用户名查找，scalar_one_or_none 返回单行或 None
    user = (
        await db.execute(select(User).where(User.username == body.username))
    ).scalar_one_or_none()
    # 用户不存在与密码错误统一返回同一提示，避免泄露账号是否存在
    if user is None or not verify_password(body.password, user.password_hash):
        # verify_password 内部使用 bcrypt 对比明文与哈希，任一条件不满足即拒绝
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")

    # 用户认证通过后，查询该用户所属的租户记录以填充 tenant_name 字段
    tenant = (
        await db.execute(select(Tenant).where(Tenant.id == user.tenant_id))
    ).scalar_one_or_none()
    # JWT 内嵌 tenant_id/role，后续请求据此做租户隔离与鉴权
    # create_access_token 将 user_id/tenant_id/role 编码进 JWT payload，并设置过期时间
    token = create_access_token(user_id=user.id, tenant_id=user.tenant_id, role=user.role)
    # 构造并返回响应体：access_token 供客户端后续请求携带，user 提供前端展示所需的用户信息
    return TokenResponse(
        access_token=token,
        user=UserInfo(
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,  # 用于前端展示的友好名称（可为空）
            role=user.role,  # 角色信息（如 internal/customer），影响后续接口的鉴权逻辑
            tenant_id=user.tenant_id,  # 租户 ID，用于多租户数据隔离
            tenant_name=tenant.name if tenant else None,  # 若租户记录存在则取名称，否则为 None
        ),
    )


@router.get("/me", response_model=UserInfo)
async def me(
    user: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> UserInfo:
    """获取当前已登录用户的身份信息（需在 Authorization Header 中携带有效 JWT）。

    参数：
        user (CurrentUser): 由 get_current_user 依赖解析 JWT 后注入，包含 user_id/tenant_id/role 等字段。
        db   (AsyncSession): 由 FastAPI 依赖注入的异步数据库会话，用于查询租户名称。

    返回：
        UserInfo: 当前用户的完整身份信息，含 tenant_name（从数据库实时读取）。
    """
    # 根据 JWT 中解析出的 tenant_id 查询租户记录，以获取可读的 tenant_name
    tenant = (
        await db.execute(select(Tenant).where(Tenant.id == user.tenant_id))
    ).scalar_one_or_none()
    # 将 CurrentUser 数据类的字段与数据库查询的租户名称组合成 UserInfo 响应对象返回
    return UserInfo(
        user_id=user.user_id,
        username=user.username,
        display_name=user.display_name,  # 用户展示名，由注册时填写
        role=user.role,  # 角色（如 internal 表示内部运营人员，customer 表示普通客户）
        tenant_id=user.tenant_id,  # 当前用户所属租户的唯一标识
        tenant_name=tenant.name if tenant else None,  # 租户名称，若租户被删除则为 None
    )
