# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""调用日志查询工具（真实查询 MySQL）。

本模块向 AI Agent 提供两个日志查询工具：
  - query_call_log：根据 request_id 精确查询单次 API 调用的详细日志，
    用于排查具体某次请求的报错原因（如 401/429/5xx）。
  - query_recent_call_stats：统计近 N 分钟某接口的调用量与各状态码分布，
    主要用于限流（429）诊断——判断是否因请求频次过高触发了 QPS 限制。

两个工具均在模块末尾通过 register() 注册到全局 REGISTRY，应用启动时
load_tools() 会 import 本模块以触发注册。
"""

# ---- 标准库导入 ----
from datetime import timedelta  # 用于计算时间窗口：anchor 时间 - N 分钟

# ---- 第三方库导入（SQLAlchemy ORM） ----
from sqlalchemy import func, select  # func 提供聚合函数（max/count），select 构造查询

# ---- 项目内部模块导入 ----
from app.db import AsyncSessionLocal  # 异步数据库会话工厂
from app.models import ApiCallLog, ApiKey  # ORM 模型：API 调用日志表、API Key 表
from app.tools.registry import ToolContext, ToolSpec, register  # 工具注册相关


async def query_call_log(args: dict, ctx: ToolContext) -> dict:
    """按 request_id 查询单次调用日志。

    根据请求唯一标识（request_id）从数据库精确查找该次 API 调用的完整记录，
    包括 HTTP 状态码、错误码、耗时等，帮助 AI 诊断具体某次请求的失败原因。
    客户侧调用时会进行租户隔离，只能查询自己租户的日志。

    Args:
        args: 包含 "request_id" 字段的参数字典。
        ctx:  工具执行上下文，提供 tenant_id 与 is_internal 标记。

    Returns:
        found=True 时返回调用日志详情；found=False 时返回失败原因。
    """
    request_id = args.get("request_id")  # 从参数中提取请求 ID
    if not request_id:
        # 必填参数缺失，提前返回错误，避免空查询
        return {"found": False, "reason": "缺少 request_id"}
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        log = (
            await s.execute(select(ApiCallLog).where(ApiCallLog.request_id == request_id))
        ).scalar_one_or_none()  # 按 request_id 精确查找，不存在则返回 None
        if log is None:
            # 数据库中无此 request_id 的记录
            return {"found": False, "reason": "未找到该 request_id 的日志"}
        # 租户隔离：客户侧只能查本租户
        if not ctx.is_internal and log.tenant_id != ctx.tenant_id:
            # 非内部调用且日志归属租户与当前租户不符，拒绝返回（防止越权查看他人数据）
            return {"found": False, "reason": "无权访问其它租户的调用日志"}
        key_masked = None  # 初始化 API Key 脱敏字段，默认为空
        if log.api_key_id:
            # 如果日志关联了 API Key，进一步查询 Key 的脱敏展示值
            key = (
                await s.execute(select(ApiKey).where(ApiKey.id == log.api_key_id))
            ).scalar_one_or_none()  # 查询对应的 ApiKey 记录
            key_masked = key.key_masked if key else None  # 取脱敏 Key，Key 不存在则为 None
        return {
            "found": True,                                      # 标记找到记录
            "request_id": log.request_id,                      # 请求唯一 ID
            "app_id": log.app_id,                              # 发起调用的应用 ID
            "endpoint": log.endpoint,                          # 被调用的接口路径
            "http_status": log.http_status,                    # HTTP 响应状态码（如 200/401/429）
            "error_code": log.error_code,                      # 业务错误码（如 AUTH_FAILED）
            "latency_ms": log.latency_ms,                      # 接口响应耗时（毫秒）
            "api_key_masked": key_masked,                      # 脱敏后的 API Key（如 sk-****）
            "called_at": log.created_at.isoformat(),           # 调用时间（ISO 8601 格式）
        }


async def query_recent_call_stats(args: dict, ctx: ToolContext) -> dict:
    """统计近 N 分钟某接口的调用情况（用于 429 限流诊断）。

    以该租户（可选指定接口）最近一次 429 限流事件的时间为"锚点"，
    向前统计 N 分钟内的调用量及各状态码分布。这种锚定方式能确保
    统计窗口围绕真实限流事件，而不受系统当前时间或背景测试日志干扰。

    如果没有 429 事件，则以该租户（接口）最新一条日志时间为锚点。

    Args:
        args: 包含可选 "endpoint"（接口路径）和可选 "minutes"（窗口分钟数，默认 240）。
        ctx:  工具执行上下文，提供 tenant_id。

    Returns:
        包含总调用量、各状态码计数、限流次数与限流比例的统计字典。
    """
    endpoint = args.get("endpoint")              # 可选：指定统计的接口路径
    minutes = int(args.get("minutes", 240))      # 时间窗口（分钟），默认 240 分钟（4 小时）
    async with AsyncSessionLocal() as s:         # 创建异步数据库会话
        # 锚点：优先定位该租户(+接口)最近一次 429 限流事件的时间；无 429 则用最近一条日志。
        # 这样能围绕"限流事件"统计窗口，不依赖系统当前时间，也不被随机背景日志漂移。
        base_conds = [ApiCallLog.tenant_id == ctx.tenant_id]  # 基础过滤：限定当前租户
        if endpoint:
            base_conds.append(ApiCallLog.endpoint == endpoint)  # 如指定接口则进一步过滤
        anchor = (
            await s.execute(
                select(func.max(ApiCallLog.created_at)).where(
                    *base_conds, ApiCallLog.http_status == 429  # 查最近一次 429 的时间
                )
            )
        ).scalar()  # 取聚合结果（单个时间值）
        if anchor is None:
            # 没有 429 记录，退而求其次：用最近一条日志时间作为锚点
            anchor = (
                await s.execute(select(func.max(ApiCallLog.created_at)).where(*base_conds))
            ).scalar()
        if anchor is None:
            # 连任何日志都没有，说明该租户/接口尚无历史数据，直接返回空统计
            return {"total": 0, "by_status": {}}
        since = anchor - timedelta(minutes=minutes)  # 计算统计窗口的起始时间
        # 构造最终统计查询的过滤条件：租户 + 时间范围（可选 + 接口）
        conds = [ApiCallLog.tenant_id == ctx.tenant_id, ApiCallLog.created_at >= since]
        if endpoint:
            conds.append(ApiCallLog.endpoint == endpoint)  # 限制只统计指定接口
        rows = (
            await s.execute(
                select(ApiCallLog.http_status, func.count())  # 查询各状态码及其出现次数
                .where(*conds)
                .group_by(ApiCallLog.http_status)             # 按状态码分组聚合
            )
        ).all()  # 返回 [(状态码, 次数), ...] 列表
        by_status = {int(st): int(cnt) for st, cnt in rows}  # 转为 {状态码: 次数} 字典
        total = sum(by_status.values())   # 计算总调用次数
        n429 = by_status.get(429, 0)      # 获取 429 限流次数，无则为 0
        return {
            "endpoint": endpoint,                                    # 统计的接口路径（可为 None）
            "window_minutes": minutes,                               # 统计时间窗口（分钟）
            "total": total,                                          # 窗口内总调用次数
            "by_status": by_status,                                  # 各状态码分布
            "rate_limited_count": n429,                              # 被限流（429）的次数
            "rate_limited_ratio": round(n429 / total, 3) if total else 0.0,  # 限流占比（保留3位小数）
        }


# ---- 将 query_call_log 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="query_call_log",                  # 工具名称，LLM function calling 使用此名
    description="根据 request_id 查询某次 API 调用的状态码、错误码、耗时、调用时间等日志详情。",
    parameters={
        "type": "object",
        "properties": {"request_id": {"type": "string", "description": "调用请求 ID，如 req_20260615_8842"}},
        "required": ["request_id"],  # request_id 为必填参数
    },
    func=query_call_log,   # 绑定上面定义的异步函数
    category="logs",       # 工具分类：日志查询
))

# ---- 将 query_recent_call_stats 工具注册到全局 REGISTRY ----
register(ToolSpec(
    name="query_recent_call_stats",          # 工具名称
    description="统计某接口近 N 分钟的调用量与各状态码分布，用于判断是否触发限流(429)。",
    parameters={
        "type": "object",
        "properties": {
            "endpoint": {"type": "string", "description": "接口路径，如 /v1/risk/score"},
            "minutes": {"type": "integer", "description": "时间窗口（分钟），默认 60"},
        },
        # endpoint 和 minutes 均为可选参数，无 required 字段
    },
    func=query_recent_call_stats,  # 绑定上面定义的统计函数
    category="logs",               # 工具分类：日志查询
))
