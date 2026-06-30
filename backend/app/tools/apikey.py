# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""API Key 状态查询工具（真实查询 MySQL）。

本模块向 AI Agent 提供一个 API Key 状态查询工具：
  - query_apikey_status：根据 api_key_id 或 app_id 查询该应用下所有 API Key 的
    状态（有效/过期/禁用）与过期时间，主要用于诊断 401 鉴权失败的原因——
    帮助开发者快速判断是 Key 过期、Key 被禁用还是其他鉴权问题。

工具在模块末尾通过 register() 注册到全局 REGISTRY，应用启动时
load_tools() 会 import 本模块以触发注册。
"""

# ---- 标准库导入 ----
from datetime import datetime  # 用于生成演示基准时间（固定日期），判断 Key 是否过期

# ---- 第三方库导入（SQLAlchemy ORM） ----
from sqlalchemy import select  # 构造 SELECT 查询语句

# ---- 项目内部模块导入 ----
from app.db import AsyncSessionLocal  # 异步数据库会话工厂
from app.models import ApiKey         # ORM 模型：API Key 表
from app.tools.registry import ToolContext, ToolSpec, register  # 工具注册相关


async def query_apikey_status(args: dict, ctx: ToolContext) -> dict:
    """查询 API Key 状态。支持按 api_key_id 或 app_id 查询。

    从数据库查询符合条件的 API Key 列表，经过租户隔离后，
    返回每个 Key 的脱敏值、状态、过期时间及是否已过期标志。
    "是否过期"基于固定演示基准时间判断，保证测试数据行为可复现。

    Args:
        args: 参数字典，支持以下字段（至少提供一个）：
            - api_key_id (str, 可选): API Key 的数据库 ID，精确定位单个 Key。
            - app_id (str, 可选): 应用 ID，查询该应用下所有 Key（可能多个）。
        ctx:  工具执行上下文，提供 tenant_id 与 is_internal 标记。

    Returns:
        found=True 时返回 Key 列表（含脱敏 Key、状态、过期信息）；
        found=False 时返回失败原因。
    """
    api_key_id = args.get("api_key_id")  # 从参数中提取 Key ID（可为 None）
    app_id = args.get("app_id")          # 从参数中提取应用 ID（可为 None）
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        stmt = select(ApiKey)  # 构造基础查询语句，默认查所有 ApiKey
        if api_key_id:
            # 优先按 api_key_id 精确查询（比 app_id 更精确）
            stmt = stmt.where(ApiKey.id == api_key_id)
        elif app_id:
            # 无 api_key_id 时按 app_id 查该应用下所有 Key
            stmt = stmt.where(ApiKey.app_id == app_id)
        else:
            # 两个参数都未提供，无法构造有效查询，直接返回错误
            return {"found": False, "reason": "需提供 api_key_id 或 app_id"}
        keys = (await s.execute(stmt)).scalars().all()  # 执行查询，返回 Key 列表
        # 租户隔离：客户侧只能看到本租户的 Key
        # 内部调用（is_internal=True）可查看所有租户的 Key（如运营排查）
        keys = [k for k in keys if ctx.is_internal or k.tenant_id == ctx.tenant_id]
        if not keys:
            # 过滤后没有可见 Key（不存在或无权访问）
            return {"found": False, "reason": "未找到对应 API Key 或无权访问"}
        now = datetime(2026, 6, 15)  # 固定为演示数据基准时间，保证过期判定可复现
        return {
            "found": True,  # 标记成功找到记录
            "keys": [
                {
                    "api_key_masked": k.key_masked,                         # 脱敏后的 Key（如 sk-***...xxx）
                    "status": k.status,                                     # Key 状态：active/disabled 等
                    "expire_at": k.expire_at.isoformat() if k.expire_at else None,  # 过期时间（ISO格式）；永不过期则为 None
                    "expired": bool(k.expire_at and k.expire_at < now),    # 是否已过期：有过期时间且早于基准时间
                }
                for k in keys  # 遍历所有过滤后的 Key 记录
            ],
        }


# ---- 将 query_apikey_status 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="query_apikey_status",  # 工具名称，LLM function calling 使用此名
    description="查询应用 API Key 的状态（有效/过期/禁用）与过期时间。用于诊断 401 鉴权失败。",
    parameters={
        "type": "object",
        "properties": {
            "app_id": {"type": "string", "description": "应用 ID，如 app_acme"},
            "api_key_id": {"type": "string", "description": "API Key ID（可选）"},
        },
        # app_id 和 api_key_id 均为可选，但运行时至少需要提供一个（由函数内部校验）
    },
    func=query_apikey_status,  # 绑定上面定义的异步查询函数
    category="apikey",          # 工具分类：API Key 管理
))
