# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""链路观测接口：查询 AgentTrace 与工具调用日志（供前端链路可视化）。

本模块是 DevSupport-AI 项目的链路追踪（Tracing）路由层，所有接口仅对内部人员开放，提供如下能力：
  - GET /api/traces/{trace_id} ：查询指定 trace 的完整执行步骤与工具调用记录（链路详情）
  - GET /api/traces            ：按多种维度（会话/租户/request_id/ticket_id）列出最近的 trace 列表

核心数据模型说明：
  - AgentTrace   ：记录 AI Agent 每一个执行步骤（step），含 agent 名称、状态、耗时、Token 用量、输入输出摘要等
  - ToolCallLog  ：记录 Agent 在某个 step 中调用的工具详情（工具名、参数摘要、结果摘要、耗时等）
  - 一个 trace_id 对应一次完整的 Agent 处理链路，包含多个 AgentTrace step 和零到多个 ToolCallLog

设计要点：
  1. 所有接口均通过 require_internal 依赖进行权限校验，防止链路数据对外泄露。
  2. list_traces 接口支持 request_id 反查（通过 ToolCallLog.args_summary 模糊匹配）和 ticket_id 反查（通过工单关联会话）。
  3. list_traces 的去重逻辑：先按数量 limit*10 超量查询以应对重复 trace_id，再手动去重取前 limit 条。
"""

# ── FastAPI 核心依赖 ──────────────────────────────────────────────────────────
from fastapi import APIRouter, Depends, HTTPException  # APIRouter 分组路由；Depends 注入依赖；HTTPException 抛出 HTTP 错误

# ── SQLAlchemy 查询工具 ───────────────────────────────────────────────────────
from sqlalchemy import select  # 构造 SELECT 语句的声明式 API
from sqlalchemy.ext.asyncio import AsyncSession  # 异步数据库会话，配合 await 使用

# ── 项目内部模块 ──────────────────────────────────────────────────────────────
from app.db import get_db  # 获取异步数据库会话的依赖函数
from app.deps import CurrentUser, require_internal  # CurrentUser：当前用户数据类；require_internal：仅允许内部角色的鉴权依赖
from app.models import AgentTrace, Ticket, ToolCallLog  # ORM 模型：AgentTrace Agent 执行步骤；Ticket 工单（用于 ticket_id 反查）；ToolCallLog 工具调用日志

# ── 路由器声明 ────────────────────────────────────────────────────────────────
# prefix="/api/traces" 使本模块所有路由挂载在 /api/traces 下；tags 用于 OpenAPI 文档分组展示
router = APIRouter(prefix="/api/traces", tags=["observability"])


@router.get("/{trace_id}")
async def get_trace(
    trace_id: str,
    user: CurrentUser = Depends(require_internal),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """获取指定 trace_id 的完整链路详情，包含所有执行步骤与工具调用记录。

    仅内部角色（role=internal）可访问，通过 require_internal 依赖自动鉴权。

    参数：
        trace_id (str)      : Agent 执行链路的唯一标识符。
        user     (CurrentUser): 由 require_internal 依赖注入，用于权限校验。
        db       (AsyncSession): 由 get_db 依赖注入的异步数据库会话。

    返回：
        dict: 链路详情，结构如下：
            - trace_id         : 链路标识
            - conversation_id  : 该 trace 所属的对话会话 ID
            - tenant_id        : 该 trace 所属的租户 ID
            - total_duration_ms: 所有 step 耗时之和（毫秒），表示链路总耗时
            - total_tokens     : 所有 step Token 用量之和
            - steps            : Agent 执行步骤列表（按 step_order 升序）
            - tool_calls       : 工具调用日志列表（按 id 升序）

    异常：
        HTTPException(404): 指定 trace_id 没有对应的 AgentTrace 记录时抛出。
    """
    # 查询该 trace_id 下的所有 Agent 执行步骤，按 step_order 升序排列（还原执行顺序）
    steps = (
        await db.execute(
            select(AgentTrace).where(AgentTrace.trace_id == trace_id).order_by(AgentTrace.step_order)
        )
    ).scalars().all()
    if not steps:
        # 无任何步骤记录表示该 trace_id 不存在，返回 404
        raise HTTPException(404, "未找到该 trace")
    # 查询该 trace_id 下的所有工具调用日志，按 id 升序排列（还原调用顺序）
    tools = (
        await db.execute(
            select(ToolCallLog).where(ToolCallLog.trace_id == trace_id).order_by(ToolCallLog.id)
        )
    ).scalars().all()
    # 组装完整链路详情：汇总指标 + 步骤列表 + 工具调用列表
    return {
        "trace_id": trace_id,  # 链路唯一标识
        "conversation_id": steps[0].conversation_id,  # 取第一个 step 的 conversation_id（同一 trace 下所有 step 归属同一会话）
        "tenant_id": steps[0].tenant_id,  # 取第一个 step 的 tenant_id（同一 trace 下所有 step 归属同一租户）
        "total_duration_ms": sum(s.duration_ms for s in steps),  # 累加所有 step 的耗时，得到整条链路的总执行时间（毫秒）
        "total_tokens": sum(s.token_usage for s in steps),  # 累加所有 step 的 Token 用量，用于费用核算
        "steps": [
            {
                "step_order": s.step_order,  # 步骤序号（从 0 或 1 开始，表示执行先后顺序）
                "agent_name": s.agent_name,  # 执行该步骤的 Agent 名称（如 "IntentAgent"、"RAGAgent"）
                "status": s.status,  # 步骤执行状态（success/failed/skipped 等）
                "duration_ms": s.duration_ms,  # 该步骤耗时（毫秒）
                "token_usage": s.token_usage,  # 该步骤消耗的 Token 数量
                "input_summary": s.input_summary,  # 该步骤的输入内容摘要（避免返回过长的完整内容）
                "output_summary": s.output_summary,  # 该步骤的输出内容摘要
                "hit_docs": s.hit_docs,  # RAG 检索命中的文档列表（仅 RAG 相关步骤有值）
                "error_message": s.error_message,  # 步骤执行失败时的错误信息（成功时为 None）
            }
            for s in steps  # 遍历所有步骤，构造步骤字典列表
        ],
        "tool_calls": [
            {
                "tool_name": t.tool_name,  # 被调用的工具名称（如 "search_knowledge"、"query_ticket"）
                "status": t.status,  # 工具调用结果状态（success/failed）
                "duration_ms": t.duration_ms,  # 工具调用耗时（毫秒）
                "args_summary": t.args_summary,  # 工具调用参数摘要（敏感信息已脱敏）
                "result_summary": t.result_summary,  # 工具调用返回结果摘要
                "error_message": t.error_message,  # 调用失败时的错误信息（成功时为 None）
            }
            for t in tools  # 遍历所有工具调用日志，构造工具调用字典列表
        ],
    }


@router.get("")
async def list_traces(
    conversation_id: str | None = None,  # 按会话 ID 过滤，查询某次对话产生的所有 trace
    tenant_id: str | None = None,  # 按租户 ID 过滤，查询某租户的所有 trace
    request_id: str | None = None,  # 按 API 请求 ID 过滤，通过工具调用参数摘要反查关联 trace
    ticket_id: str | None = None,  # 按工单 ID 过滤，通过工单关联的会话 ID 反查 trace
    limit: int = 20,  # 返回结果数量上限，默认 20 条
    user: CurrentUser = Depends(require_internal),  # 鉴权：仅内部角色可访问
    db: AsyncSession = Depends(get_db),  # 数据库会话注入
) -> dict:
    """按会话/租户/request_id/ticket_id 列出最近的 trace（去重 trace_id）。

    支持多种过滤维度，过滤条件可组合使用（多个条件为 AND 关系）。
    特殊过滤逻辑：
      - request_id 过滤：通过 ToolCallLog.args_summary LIKE 模糊匹配，反查包含该 request_id 的 trace
      - ticket_id  过滤：通过工单记录找到关联 conversation_id，再查询该会话的 trace

    去重策略：先超量查询（limit*10 条）避免因重复 trace_id 导致实际结果不足 limit 条，
    再通过 seen 集合手动去重，确保最终返回恰好 limit 条唯一 trace。

    参数：
        conversation_id (str | None): 可选，按对话会话 ID 过滤。
        tenant_id       (str | None): 可选，按租户 ID 过滤。
        request_id      (str | None): 可选，按 API 请求 ID 过滤（模糊匹配工具参数摘要）。
        ticket_id       (str | None): 可选，按工单 ID 过滤（反查关联会话）。
        limit           (int)       : 返回结果数量上限，默认 20。
        user            (CurrentUser): 由 require_internal 依赖注入，用于权限校验。
        db              (AsyncSession): 由 get_db 依赖注入的异步数据库会话。

    返回：
        dict: {"traces": [{"trace_id": ..., "conversation_id": ..., "tenant_id": ..., "created_at": ...}, ...]}
    """
    # 构造基础查询：按 AgentTrace.id 降序排列，优先返回最近产生的 trace 记录
    stmt = select(AgentTrace).order_by(AgentTrace.id.desc())
    if conversation_id:
        # 按会话 ID 过滤：只返回该会话产生的 Agent 执行记录
        stmt = stmt.where(AgentTrace.conversation_id == conversation_id)
    if tenant_id:
        # 按租户 ID 过滤：只返回该租户下的 Agent 执行记录，支持运营人员按租户排查问题
        stmt = stmt.where(AgentTrace.tenant_id == tenant_id)
    if request_id:
        # 经工具调用日志反查：哪些 trace 调用过含该 request_id 的工具
        # 通过 ToolCallLog.args_summary 的 LIKE 模糊匹配，找到与该 request_id 相关的所有 trace_id
        tids = (
            await db.execute(
                select(ToolCallLog.trace_id).where(ToolCallLog.args_summary.like(f"%{request_id}%"))
            )
        ).scalars().all()
        # 若找不到匹配的 trace_id，使用 "__none__" 哨兵值确保 WHERE IN 子句返回空结果集（而非全量扫描）
        stmt = stmt.where(AgentTrace.trace_id.in_(tids or ["__none__"]))
    if ticket_id:
        # 经工单关联会话反查
        # 先查询工单记录，获取其关联的 conversation_id
        t = (await db.execute(select(Ticket).where(Ticket.ticket_id == ticket_id))).scalar_one_or_none()
        # 若工单存在则按其 conversation_id 过滤；若工单不存在则使用哨兵值确保返回空结果集
        stmt = stmt.where(AgentTrace.conversation_id == (t.conversation_id if t else "__none__"))
    # 超量查询 limit*10 条，为后续手动去重 trace_id 留出足够的数据空间
    rows = (await db.execute(stmt.limit(limit * 10))).scalars().all()
    # 手动去重：通过 seen 集合记录已出现的 trace_id，避免同一 trace 的多个 step 被重复计入
    seen, traces = set(), []
    for r in rows:
        if r.trace_id in seen:
            # 该 trace_id 已经被收录，跳过当前行（同一 trace 的后续 step 记录）
            continue
        seen.add(r.trace_id)  # 将此 trace_id 标记为已处理
        # 将该 trace 的概要信息（不含详细 step）加入结果列表
        traces.append({"trace_id": r.trace_id, "conversation_id": r.conversation_id,
                       "tenant_id": r.tenant_id, "created_at": r.created_at.isoformat()})  # isoformat() 将 datetime 转为 ISO 8601 字符串
        if len(traces) >= limit:
            # 已收集到足够数量的唯一 trace，提前退出循环以节省计算资源
            break
    return {"traces": traces}  # 返回去重后的 trace 列表
