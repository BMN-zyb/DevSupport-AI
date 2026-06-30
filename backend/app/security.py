# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""认证安全工具：密码哈希 + JWT 签发/校验。

本模块提供所有与身份认证相关的底层安全函数，职责包括：
  1. 密码安全存储：使用 bcrypt 算法对明文密码加盐哈希，防止数据库泄露后密码被逆推；
  2. JWT 签发：生成携带用户身份信息和过期时间的访问令牌，供客户端后续请求携带；
  3. JWT 校验：验证 Token 签名合法性和有效期，解码并返回 payload 供上层使用。

安全注意事项：
  - bcrypt 有效输入上限为 72 字节，超出部分会被忽略，因此先截断再处理；
  - jwt_secret 在生产环境必须替换为高熵随机字符串；
  - 解码失败（JWTError）统一返回 None，不向调用方暴露具体错误原因（防止信息泄漏）。
"""

# 标准库：datetime 用于计算 Token 过期时间；timedelta 表示时间间隔；timezone 用于获取 UTC 时区
from datetime import datetime, timedelta, timezone

# bcrypt：业界标准的密码哈希库，提供加盐哈希和验证功能，计算成本高（抗暴力破解）
import bcrypt
# python-jose：JWT 编解码库；JWTError 是所有 JWT 相关异常的基类；jwt 提供 encode/decode 函数
from jose import JWTError, jwt

# 导入全局配置，获取 JWT 密钥、算法和过期分钟数
from app.config import settings


def _to_bytes(password: str) -> bytes:
    """将密码字符串转换为 UTF-8 字节串，并截断到 72 字节。

    bcrypt 算法设计上只处理前 72 字节（72 字节之后的内容会被忽略），
    为了确保行为可预期（防止两个仅在第 73 字节之后不同的密码被判定为相同），
    在传入 bcrypt 前主动截断。

    Args:
        password: 用户输入的明文密码字符串

    Returns:
        bytes: UTF-8 编码后截断至最多 72 字节的字节串
    """
    # bcrypt 上限 72 字节，超出截断
    return password.encode("utf-8")[:72]  # 先编码为 UTF-8 字节，再切片取前 72 字节


def hash_password(password: str) -> str:
    """对明文密码进行 bcrypt 加盐哈希，返回可存入数据库的哈希字符串。

    每次调用都会生成不同的随机盐值（gensalt()），
    因此相同的密码每次哈希结果不同，防止彩虹表攻击。

    Args:
        password: 用户注册或修改密码时提供的明文密码

    Returns:
        str: bcrypt 哈希字符串（包含算法版本、盐值和哈希值），格式如 $2b$12$...
    """
    return bcrypt.hashpw(_to_bytes(password), bcrypt.gensalt()).decode("utf-8")
    # bcrypt.gensalt() 生成随机盐值；hashpw 执行加盐哈希；结果解码为 str 便于存入 VARCHAR 列


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码与已存储的 bcrypt 哈希是否匹配。

    bcrypt.checkpw 内部会从哈希字符串中自动提取盐值，
    对明文密码执行相同参数的哈希，再进行常数时间比较（防时序攻击）。

    Args:
        plain: 用户登录时提供的明文密码
        hashed: 数据库中存储的 bcrypt 哈希字符串

    Returns:
        bool: True 表示密码匹配，False 表示不匹配或发生异常
    """
    try:
        return bcrypt.checkpw(_to_bytes(plain), hashed.encode("utf-8"))
        # 将明文密码转为字节，将哈希字符串重新编码为字节，交给 bcrypt 验证
    except (ValueError, TypeError):  # hashed 格式非法（如空字符串、非 bcrypt 格式）时捕获异常
        return False  # 任何异常均视为验证失败，避免向上层暴露内部错误


def create_access_token(*, user_id: str, tenant_id: str, role: str) -> str:
    """签发 JWT 访问令牌。

    令牌 payload 包含用户 ID（sub）、租户 ID、角色和过期时间（exp）。
    上层（deps.py）解码后可直接获取这些信息，无需再次查询数据库（热路径优化）。
    注意：角色和租户 ID 存于 Token 中意味着修改数据库中的角色/租户后，
    旧 Token 在过期前仍携带旧值，需结合业务场景决定是否需要主动吊销。

    Args:
        user_id: 用户唯一 ID，对应数据库 user.id，存入 JWT sub 字段（标准）
        tenant_id: 用户所属租户 ID，存入 payload 供路由快速做租户隔离判断
        role: 用户角色字符串，存入 payload 供路由快速做 RBAC 判断

    Returns:
        str: 签名后的 JWT 字符串，格式为 header.payload.signature（Base64URL 编码）
    """
    # 计算过期时间：当前 UTC 时间加上配置中的分钟数，使用 timezone.utc 确保时区正确
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)

    # 构建 JWT payload 字典
    payload = {
        "sub": user_id,           # subject（标准 JWT 字段）：标识 Token 对应的用户
        "tenant_id": tenant_id,   # 自定义字段：租户 ID，便于 deps.py 快速提取
        "role": role,             # 自定义字段：用户角色，便于 deps.py 快速做权限判断
        "exp": expire,            # expiration（标准 JWT 字段）：过期时间戳，jose 库自动验证
    }
    # 使用配置中的密钥和算法对 payload 进行 HMAC 签名，生成 JWT 字符串
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict | None:
    """解码并验证 JWT 访问令牌，返回 payload 字典。

    jose 库在 decode 时会自动：
      1. 验证签名（防止 Token 被篡改）；
      2. 验证 exp 字段（防止使用过期 Token）。
    任何验证失败均抛出 JWTError，统一捕获后返回 None。

    Args:
        token: 从请求头 Authorization: Bearer <token> 中提取的原始 JWT 字符串

    Returns:
        dict | None: 验证通过时返回解码后的 payload 字典（含 sub、tenant_id、role 等字段），
                     验证失败（签名错误、过期等）时返回 None
    """
    try:
        # 解码 JWT：验证签名和有效期，返回 payload 字典
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        # algorithms 传入列表，jose 要求明确指定允许的算法，防止"none"算法攻击
    except JWTError:  # 签名无效、Token 过期、格式错误等均属于 JWTError 的子类
        return None   # 统一返回 None，由调用方（deps.py）决定如何处理
