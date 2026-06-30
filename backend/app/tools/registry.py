# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""工具调用中心：注册、schema、超时、重试、脱敏日志、高风险隔离。

- 每个工具声明 input schema、超时、重试、是否高风险。
- execute() 真实执行并写 tool_call_log（参数与结果脱敏）。
- 高风险工具不暴露给 AI（openai_tools 默认排除），只能人工/后台执行。

本模块是整个工具层的核心调度中心，其他工具模块（apikey、billing_tools、
logs、ticket_tools）通过调用 register() 将自己注册到全局 REGISTRY 字典中，
AI Agent 在需要调用外部能力时，统一通过本模块的 execute() 入口分发执行。
"""

# ---- 标准库导入 ----
import asyncio  # 提供异步等待（wait_for）与事件循环支持
import json     # 用于将参数/结果序列化为 JSON 字符串，便于脱敏日志记录
import time     # 用于测量工具调用的耗时（perf_counter 高精度计时）
from collections.abc import Awaitable, Callable  # 类型提示：异步可调用类型
from dataclasses import dataclass, field         # 数据类装饰器，简化配置类定义

# ---- 项目内部模块导入 ----
from app.config import settings          # 全局配置，提供 tool_timeout_seconds 等参数
from app.db import AsyncSessionLocal     # 异步数据库会话工厂，用于写入工具调用日志
from app.guardrail import desensitize   # 脱敏模块，对日志中的敏感字段（如 Key、手机号）做掩码
from app.models import ToolCallLog      # ORM 模型：工具调用日志表


@dataclass
class ToolContext:
    """工具执行上下文，随每次调用传入，用于租户隔离和权限判断。

    Attributes:
        tenant_id:    当前调用所属租户 ID，所有工具必须据此做数据隔离。
        trace_id:     链路追踪 ID，写入日志以便跨服务排查调用链。
        is_internal:  是否为内部调用（后台/运营侧），True 时可绕过租户隔离和高风险拦截。
    """
    tenant_id: str           # 租户标识，数据隔离的核心字段
    trace_id: str = ""       # 链路 ID，默认为空字符串（非必须）
    is_internal: bool = False  # 内部标记，默认 False 即客户侧普通调用


@dataclass
class ToolSpec:
    """工具规格描述，定义一个可注册工具的完整元信息。

    Attributes:
        name:        工具唯一标识名称，同时作为 function calling 的函数名。
        description: 工具功能描述，LLM 据此判断何时调用。
        parameters:  JSON Schema 格式的参数定义，供 LLM 解析与验证入参。
        func:        实际执行逻辑的异步函数，签名为 (args, ctx) -> Awaitable[dict]。
        timeout:     单次调用的超时秒数，超时后触发重试机制。
        retries:     超时或异常时的重试次数（默认 1 次，即最多执行 2 次）。
        high_risk:   高风险标记；为 True 时不暴露给 AI，只能通过内部后台调用。
        category:    工具分类，便于按类型过滤与展示（如 billing、logs、ticket）。
    """
    name: str                  # 工具名称，需全局唯一
    description: str           # 供 LLM 理解的自然语言描述
    parameters: dict           # JSON schema，定义工具的输入参数结构
    func: Callable[[dict, ToolContext], Awaitable[dict]]  # 异步执行函数
    timeout: float = settings.tool_timeout_seconds  # 默认超时从全局配置读取
    retries: int = 1           # 默认重试 1 次（共尝试 2 次）
    high_risk: bool = False    # 默认非高风险，允许 AI 调用
    category: str = "general"  # 默认分类为通用


# 全局工具注册表：key 为工具名称，value 为对应的 ToolSpec 实例
# 所有工具模块在 import 时通过 register() 往此字典写入自己
REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    """将工具规格注册到全局 REGISTRY，并返回该规格（便于链式调用）。

    Args:
        spec: 待注册的工具规格实例。

    Returns:
        传入的 spec 本身（方便赋值给变量后复用）。
    """
    REGISTRY[spec.name] = spec  # 以工具名为 key 存入全局字典，同名会覆盖
    return spec  # 返回 spec 支持链式或赋值使用


def openai_tools(include_high_risk: bool = False) -> list[dict]:
    """返回可供 LLM function calling 的工具 schema（默认排除高风险）。

    将 REGISTRY 中的工具转换为 OpenAI function calling 所需的 JSON 结构列表，
    默认过滤掉 high_risk=True 的工具，防止 AI 直接触发危险操作。

    Args:
        include_high_risk: 为 True 时也包含高风险工具（仅供内部调试使用）。

    Returns:
        符合 OpenAI function calling 格式的工具描述列表。
    """
    tools = []  # 用于收集符合条件的工具 schema
    for spec in REGISTRY.values():  # 遍历注册表中所有工具
        if spec.high_risk and not include_high_risk:
            continue  # 默认模式下跳过高风险工具，不将其暴露给 AI
        tools.append(
            {
                "type": "function",  # OpenAI function calling 的固定类型字段
                "function": {
                    "name": spec.name,             # 函数名，LLM 调用时使用
                    "description": spec.description,  # 描述，帮助 LLM 决策
                    "parameters": spec.parameters,    # 参数 schema，LLM 据此生成入参
                },
            }
        )
    return tools  # 返回可直接传给 ChatCompletion API 的工具列表


async def _log_call(ctx, name, args, result, status, duration_ms, error):
    """将一次工具调用的执行记录写入数据库日志表（内部函数，不对外暴露）。

    参数和结果在写入前均经过脱敏处理，防止 API Key、手机号等敏感信息落库。

    Args:
        ctx:         工具上下文，提供 tenant_id 和 trace_id。
        name:        被调用的工具名称。
        args:        调用时传入的原始参数字典。
        result:      工具返回的结果字典。
        status:      执行状态，"ok" 表示成功，"error" 表示失败。
        duration_ms: 本次调用耗时（毫秒）。
        error:       错误信息字符串，无错误时为 None。
    """
    async with AsyncSessionLocal() as s:  # 创建异步数据库会话
        s.add(
            ToolCallLog(
                trace_id=ctx.trace_id,        # 写入链路 ID，便于跨服务追踪
                tenant_id=ctx.tenant_id,      # 写入租户 ID，支持多租户日志查询
                tool_name=name,               # 记录工具名称
                # 对参数 JSON 做脱敏，截断到 1000 字符防止日志过大
                args_summary=desensitize.desensitize_text(json.dumps(args, ensure_ascii=False))[:1000],
                # 对结果 JSON 做脱敏，截断到 1500 字符，default=str 处理不可序列化对象
                result_summary=desensitize.desensitize_text(json.dumps(result, ensure_ascii=False, default=str))[:1500],
                status=status,                # 执行状态
                duration_ms=duration_ms,      # 耗时（毫秒）
                error_message=error,          # 错误消息（成功时为 None）
            )
        )
        await s.commit()  # 提交事务，将日志持久化到数据库


async def execute(name: str, args: dict, ctx: ToolContext) -> dict:
    """执行工具：高风险拦截 + 超时 + 重试 + 脱敏日志。

    这是所有工具调用的统一入口，负责：
    1. 查找工具是否存在；
    2. 高风险工具非内部调用时直接拦截返回错误；
    3. 在超时限制内执行工具函数，失败则按重试次数重试；
    4. 无论成功或失败，均写入工具调用日志。

    Args:
        name: 工具名称，对应 REGISTRY 中的 key。
        args: 工具调用的入参字典，由 LLM 或调用方构造。
        ctx:  执行上下文，包含租户信息和内部标记。

    Returns:
        成功时返回 {"ok": True, "data": <工具返回值>}；
        失败时返回 {"ok": False, "error": <错误描述>}。
    """
    spec = REGISTRY.get(name)  # 从注册表中查找工具规格
    if spec is None:
        return {"ok": False, "error": f"未知工具: {name}"}  # 工具不存在，直接返回错误

    # 高风险工具不允许 AI 直接执行（仅内部显式调用且需标记）
    if spec.high_risk and not ctx.is_internal:
        # 拦截高风险操作，保护系统安全，防止 AI 误触发资金/密钥相关操作
        return {"ok": False, "error": f"工具 {name} 为高风险操作，需人工处理，AI 不可直接执行"}

    start = time.perf_counter()  # 记录开始时间，用于计算总耗时（包括所有重试）
    last_err = None              # 保存最后一次失败的错误信息
    for attempt in range(spec.retries + 1):  # 共执行 retries+1 次（1次正常+N次重试）
        try:
            # 在超时限制内等待工具异步函数执行完成
            data = await asyncio.wait_for(spec.func(args, ctx), timeout=spec.timeout)
            duration = int((time.perf_counter() - start) * 1000)  # 计算耗时（毫秒）
            result = {"ok": True, "data": data}  # 构造成功响应
            await _log_call(ctx, name, args, data, "ok", duration, None)  # 写入成功日志
            return result  # 成功则立即返回，不再重试
        except asyncio.TimeoutError:
            # 工具执行超时，记录错误信息后进入下一次重试
            last_err = f"工具调用超时(>{spec.timeout}s)"
        except Exception as e:  # noqa: BLE001
            # 捕获所有其他异常（网络错误、数据库异常等），继续重试
            last_err = f"{type(e).__name__}: {e}"
    # 所有重试均失败，计算总耗时并写入失败日志
    duration = int((time.perf_counter() - start) * 1000)
    await _log_call(ctx, name, args, {}, "error", duration, last_err)  # 写入失败日志
    return {"ok": False, "error": last_err}  # 返回最后一次错误信息


def load_tools() -> None:
    """导入各工具模块以触发注册。

    Python 的模块级代码（如模块末尾的 register(ToolSpec(...)) 调用）
    只在模块被 import 时执行一次。本函数通过显式 import 确保所有工具
    模块都被加载，从而完成各自工具的注册，使 REGISTRY 就绪。

    应在应用启动时（如 lifespan 钩子中）调用一次。
    """
    from app.tools import apikey, billing_tools, logs, ticket_tools  # noqa: F401
    # 上面的 import 会触发各模块末尾的 register() 调用，将工具写入 REGISTRY
