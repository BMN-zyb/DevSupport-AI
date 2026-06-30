# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""评估与运营指标接口（内部角色）。

本模块是 DevSupport-AI 项目的评估与运营监控路由层，所有接口仅对内部人员（role=internal）开放：
  - GET  /api/metrics   ：返回对话数量统计、AI 解决率、意图分布、工单统计及 Token 费用摘要等运营指标
  - POST /api/eval/run  ：触发标准评估集的自动化测试，返回各项评测指标与失败用例（Badcase）

设计要点：
  1. 使用 require_internal 依赖进行权限校验，非内部用户访问将直接被拒绝。
  2. /metrics 接口聚合多张数据库表的统计信息，全部使用 COUNT/GROUP BY 聚合查询，避免全量加载。
  3. /eval/run 使用延迟导入（函数内 import），避免启动时加载耗时的评估模块影响服务启动速度。
"""

# ── FastAPI 核心依赖 ──────────────────────────────────────────────────────────
from fastapi import APIRouter, Depends  # APIRouter 用于注册路由分组；Depends 用于注入鉴权与数据库依赖

# ── SQLAlchemy 查询与聚合工具 ─────────────────────────────────────────────────
from sqlalchemy import func, select  # func 提供 SQL 聚合函数（如 COUNT）；select 构造 SELECT 语句
from sqlalchemy.ext.asyncio import AsyncSession  # 异步数据库会话，配合 await 使用

# ── 项目内部模块 ──────────────────────────────────────────────────────────────
from app.db import get_db  # 获取异步数据库会话的依赖函数
from app.deps import CurrentUser, require_internal  # CurrentUser：当前用户数据类；require_internal：仅允许内部角色的鉴权依赖
from app.models import Conversation, Ticket  # ORM 模型：Conversation 表示对话会话，Ticket 表示工单记录
from app.observability import cost  # 可观测性模块中的 cost 对象，提供按租户汇总 Token 费用的方法

# ── 路由器声明 ────────────────────────────────────────────────────────────────
# prefix="/api" 使路由挂载在 /api 下；tags=["eval"] 用于 OpenAPI 文档界面分组展示
router = APIRouter(prefix="/api", tags=["eval"])


@router.get("/metrics")
async def metrics(
    user: CurrentUser = Depends(require_internal), db: AsyncSession = Depends(get_db)
) -> dict:
    """返回系统运营指标汇总，包含对话统计、AI 解决率、意图分布、工单状态与 Token 费用。

    该接口为内部运营监控接口，仅内部角色（role=internal）可访问，
    通过 require_internal 依赖自动鉴权，非内部用户将收到 403 错误。

    参数：
        user (CurrentUser): 由 require_internal 依赖注入，用于权限校验。
        db   (AsyncSession): 由 get_db 依赖注入的异步数据库会话，用于执行聚合查询。

    返回：
        dict: 包含以下结构的指标字典：
            - conversations     : 对话总数、AI 解决数、转人工数、AI 解决率
            - intent_distribution: 各意图标签对应的对话数量分布
            - tickets           : 工单按状态分布 + 按优先级分布
            - token_cost_by_tenant: 各租户的 Token 用量与费用摘要
    """
    # 查询 Conversation 表的总记录数，作为对话总量基准值；若结果为 None（表为空）则默认为 0
    total_conv = await db.scalar(select(func.count()).select_from(Conversation)) or 0
    # 查询被 AI 成功解决的对话数量（resolved_by_ai 字段为 True）
    resolved = await db.scalar(
        select(func.count()).select_from(Conversation).where(Conversation.resolved_by_ai.is_(True))
    ) or 0
    # 查询已转人工处理的对话数量（transferred_to_human 字段为 True）
    transferred = await db.scalar(
        select(func.count()).select_from(Conversation).where(Conversation.transferred_to_human.is_(True))
    ) or 0

    # 按 latest_intent 字段分组统计各意图的对话数量，用于了解用户问题类型分布
    intent_rows = (
        await db.execute(
            select(Conversation.latest_intent, func.count()).group_by(Conversation.latest_intent)
        )
    ).all()
    # 按工单 status 字段分组统计各状态（new/in_progress/closed 等）的工单数量
    status_rows = (
        await db.execute(select(Ticket.status, func.count()).group_by(Ticket.status))
    ).all()
    # 按工单 priority 字段分组统计各优先级（P1/P2/P3 等）的工单数量
    priority_rows = (
        await db.execute(select(Ticket.priority, func.count()).group_by(Ticket.priority))
    ).all()

    # 将所有统计结果组装成嵌套字典返回
    return {
        "conversations": {
            "total": int(total_conv),  # 对话总数，转为 int 确保 JSON 序列化安全
            "resolved_by_ai": int(resolved),  # AI 自动解决的对话数
            "transferred_to_human": int(transferred),  # 转给人工处理的对话数
            # AI 解决率 = AI解决数/总对话数，保留3位小数；无对话时返回 0.0 避免除零错误
            "ai_resolution_rate": round(resolved / total_conv, 3) if total_conv else 0.0,
        },
        # 将意图分组查询结果转为 {意图标签: 数量} 字典；意图为 None 时用 "unknown" 替代
        "intent_distribution": {(k or "unknown"): int(v) for k, v in intent_rows},
        "tickets": {
            # 工单状态分布：{状态: 数量}；状态为 None 时用 "unknown" 替代
            "by_status": {(k or "unknown"): int(v) for k, v in status_rows},
            # 工单优先级分布：{优先级: 数量}；优先级为 None 时用 "unknown" 替代
            "by_priority": {(k or "unknown"): int(v) for k, v in priority_rows},
        },
        # 调用 cost 模块的 summary_by_tenant 异步方法，按租户汇总 Token 用量与估算费用
        "token_cost_by_tenant": await cost.summary_by_tenant(),
    }


@router.post("/eval/run")
async def run_eval(user: CurrentUser = Depends(require_internal)) -> dict:
    """运行标准评估集，返回各项指标与 Badcase（耗时较长）。

    该接口触发完整的自动化评测流程，会对预定义的测试问题集逐一发起 AI 推理并评分，
    因此耗时可能较长（视测试集大小而定），建议异步调用或设置足够长的超时时间。

    仅内部角色可调用，通过 require_internal 依赖自动鉴权。

    参数：
        user (CurrentUser): 由 require_internal 依赖注入，用于权限校验。

    返回：
        dict: 评测结果，具体结构由 eval.run_eval.evaluate() 函数决定，通常包含：
              - 各项评测指标得分（准确率、召回率等）
              - Badcase 列表（回答不符合预期的测试用例详情）
    """
    # 使用函数内延迟导入，避免在服务启动时加载耗时的评估模块（如加载模型、读取测试集等）
    from eval.run_eval import evaluate  # 导入评估入口函数，该函数内部会执行完整的评测流程

    return await evaluate()  # 异步执行评测并返回结果字典
