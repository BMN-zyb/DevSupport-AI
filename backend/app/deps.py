# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""依赖注入：当前用户、角色校验、租户隔离。

本模块为 FastAPI 提供统一的认证与授权依赖函数，供所有需要鉴权的路由通过
Depends() 注入使用。核心逻辑：
  1. 从请求头 Authorization: Bearer <token> 中提取 JWT；
  2. 解码 JWT 获取用户 ID，再从数据库查询完整用户信息；
  3. 封装为 CurrentUser 数据类，供路由函数直接使用；
  4. 提供基于角色的访问控制（RBAC）函数和租户隔离断言。
"""

# 标准库：dataclass 装饰器，用于简洁地声明数据类（无需手写 __init__）
from dataclasses import dataclass

# FastAPI：Depends 用于声明依赖注入；HTTPException 用于抛出 HTTP 错误响应；status 提供 HTTP 状态码常量
from fastapi import Depends, HTTPException, status
# HTTPBearer：从请求头提取 Bearer Token 的安全方案；HTTPAuthorizationCredentials：封装提取到的凭证信息
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# SQLAlchemy：select 构建查询语句
from sqlalchemy import select
# 异步会话类型注解
from sqlalchemy.ext.asyncio import AsyncSession

# 内部模块：get_db 提供数据库会话依赖
from app.db import get_db
# ORM 模型：User 对应 user 表
from app.models import User
# 安全工具：decode_access_token 解码并验证 JWT
from app.security import decode_access_token

# 创建 HTTP Bearer Token 提取器；auto_error=False 表示未提供 Token 时不自动抛错，
# 而是返回 None，由后续代码决定如何处理（便于区分"未登录"和"Token 格式错误"）
bearer = HTTPBearer(auto_error=False)

# 客户侧角色：只能访问本租户数据
CUSTOMER_ROLES = {"customer_dev", "customer_admin"}  # 客户开发者和客户管理员，受租户隔离约束
# 内部角色：可进工作台
INTERNAL_ROLES = {"support", "admin"}  # 技术支持人员和系统管理员，可跨租户操作


@dataclass
class CurrentUser:
    """当前已认证用户的上下文信息。

    由 get_current_user() 依赖函数填充，通过 FastAPI Depends 注入到路由函数。
    使用 dataclass 而非 Pydantic Model，是因为此对象仅在请求处理链路内部流转，
    不需要序列化/反序列化能力。

    Attributes:
        user_id: 用户唯一标识（对应 user.id）
        username: 用户登录名（唯一）
        display_name: 用户展示名称
        role: 用户角色字符串，可选值：customer_dev / customer_admin / support / admin
        tenant_id: 用户所属租户 ID，用于租户隔离判断
    """
    user_id: str         # 用户唯一 ID，来源于 JWT sub 字段
    username: str        # 用户名，用于展示和审计日志
    display_name: str    # 显示名称，用于界面展示
    role: str            # 角色，决定访问权限范围
    tenant_id: str       # 所属租户 ID，客户侧角色只能访问同租户资源

    @property
    def is_internal(self) -> bool:
        """判断当前用户是否为内部角色（support 或 admin）。

        内部角色可访问工作台、跨租户查询等高权限接口。

        Returns:
            bool: True 表示内部角色，False 表示客户侧角色
        """
        return self.role in INTERNAL_ROLES  # 检查角色是否属于内部角色集合


async def get_current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(bearer),  # 从请求头提取 Bearer Token，可能为 None
    db: AsyncSession = Depends(get_db),                           # 注入数据库异步会话
) -> CurrentUser:
    """FastAPI 依赖：解析 JWT 并查询数据库，返回当前用户对象。

    认证流程：
      1. 检查请求头中是否存在 Bearer Token；
      2. 解码 JWT，验证签名和有效期；
      3. 从 JWT payload 取出 user_id（sub 字段）；
      4. 查询数据库确认用户存在；
      5. 构建并返回 CurrentUser 数据类实例。

    Args:
        cred: HTTP Bearer 凭证对象，包含原始 token 字符串；未提供时为 None
        db: 异步数据库会话，由 get_db 依赖注入

    Returns:
        CurrentUser: 包含用户 ID、用户名、角色、租户 ID 等信息的数据类实例

    Raises:
        HTTPException(401): 未提供凭证、凭证无效/过期、或用户不存在
    """
    if cred is None:  # 请求头中没有携带 Authorization: Bearer ... 时
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未提供凭证")  # 直接拒绝，返回 401

    payload = decode_access_token(cred.credentials)  # 解码 JWT，验证签名和有效期；失败返回 None
    if not payload:  # JWT 无效（签名错误、已过期等）
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "凭证无效或已过期")  # 返回 401

    user_id = payload.get("sub")  # 从 JWT payload 中取出用户 ID（标准字段 sub = subject）

    # 查询数据库：用 user_id 查找对应的 User 记录，避免仅凭 JWT 中的信息作判断（防止已注销用户复用 Token）
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:  # 用户在数据库中不存在（可能已被删除或 JWT 数据被篡改）
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")  # 返回 401

    # 构建并返回 CurrentUser，只暴露路由函数需要的字段，不传递敏感信息如 password_hash
    return CurrentUser(
        user_id=user.id,                  # 用户唯一 ID
        username=user.username,            # 登录用户名
        display_name=user.display_name,    # 展示名称
        role=user.role,                    # 角色（决定权限）
        tenant_id=user.tenant_id,          # 租户 ID（用于数据隔离）
    )


def require_internal(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """要求内部角色（工作台等）。

    作为 FastAPI 路由的依赖函数使用，自动先执行 get_current_user 完成认证，
    再检查角色，仅允许 support 或 admin 角色通过。

    Args:
        user: 当前已认证用户，由 get_current_user 依赖提供

    Returns:
        CurrentUser: 角色验证通过的当前用户对象

    Raises:
        HTTPException(403): 当前用户不是内部角色时抛出，拒绝访问
    """
    if not user.is_internal:  # 客户侧角色（customer_dev/customer_admin）不允许访问内部接口
        raise HTTPException(status.HTTP_403_FORBIDDEN, "需要技术支持或管理员权限")  # 返回 403 禁止访问
    return user  # 校验通过，将用户对象传递给路由函数


def assert_tenant_access(user: CurrentUser, target_tenant_id: str) -> None:
    """租户隔离：客户侧角色只能访问本租户数据；内部角色可跨租户。

    在需要跨租户操作时（如管理员查询特定租户数据），内部角色不受限制；
    客户侧角色只能访问与自身 tenant_id 相同的数据，防止数据泄漏。

    Args:
        user: 当前已认证用户
        target_tenant_id: 请求访问的目标租户 ID

    Raises:
        HTTPException(403): 客户侧角色尝试访问其他租户数据时抛出
    """
    if user.is_internal:  # 内部角色（support/admin）无租户限制，直接放行
        return
    if user.tenant_id != target_tenant_id:  # 客户侧角色的租户 ID 与目标租户不匹配
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权访问其它租户数据")  # 返回 403，租户隔离拦截
