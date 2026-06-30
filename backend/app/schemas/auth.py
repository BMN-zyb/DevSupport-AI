# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""认证相关 DTO（Data Transfer Object，数据传输对象）。

本模块定义了用户认证流程中涉及的请求和响应数据结构，使用 Pydantic BaseModel
提供自动的数据验证、序列化和文档生成能力。包含以下三个 schema：
  - LoginRequest：登录请求体，包含用户名和密码。
  - UserInfo：用户信息数据对象，用于在 Token 中携带用户上下文。
  - TokenResponse：登录成功后的响应体，包含 JWT Token 和用户信息。
"""

# 第三方库：pydantic 提供基于类型注解的数据验证框架，FastAPI 用它处理请求/响应的数据校验
from pydantic import BaseModel


class LoginRequest(BaseModel):
    """用户登录请求体 DTO。

    客户端提交用户名和密码时使用此结构，FastAPI 会自动将请求 JSON 反序列化并校验。
    字段类型为 str，pydantic 会确保值非空且为字符串类型。
    """
    username: str  # 用户登录名（账号），不允许为 None
    password: str  # 用户密码（明文传输，应配合 HTTPS 使用），不允许为 None


class UserInfo(BaseModel):
    """用户信息数据对象 DTO。

    存储登录用户的身份和租户上下文信息，通常嵌入在 Token 响应中返回给客户端，
    也可从 JWT Payload 中解码还原，供 API 处理器识别当前用户。
    """
    user_id: str          # 用户唯一标识符，如数据库主键或 UUID
    username: str         # 用户登录名，与 LoginRequest.username 对应
    display_name: str     # 用户显示名（昵称/真实姓名），用于前端展示
    role: str             # 用户角色，如 "admin"、"agent"、"user"，用于权限控制
    tenant_id: str        # 所属租户 ID，多租户系统中用于数据隔离和缓存命名空间划分
    tenant_name: str | None = None  # 租户名称（可选），用于前端展示；None 表示未设置


class TokenResponse(BaseModel):
    """登录成功后的响应体 DTO。

    包含颁发的 JWT Access Token 和当前用户信息，客户端收到后应将 access_token
    存储在本地（如 localStorage），并在后续请求中通过 Authorization: Bearer <token> 携带。
    """
    access_token: str          # JWT Access Token 字符串，用于后续 API 请求的身份验证
    token_type: str = "bearer"  # Token 类型，固定为 "bearer"，符合 OAuth2 规范
    user: UserInfo             # 当前登录用户的详细信息，避免前端再次发起获取用户信息的请求
