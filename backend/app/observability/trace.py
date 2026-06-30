# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""AgentTrace 链路采集与持久化。

每次请求生成 trace_id，逐节点记录耗时/token/命中文档/状态，最终落库 agent_trace。

本模块职责：
  - TraceCollector：一次请求的链路收集器，在每个编排节点调用 step() 记录执行情况，
    请求结束后调用 persist() 将所有步骤批量写入数据库。
  - timer：轻量上下文管理器，通过 with 语句精确测量代码块的执行耗时（毫秒）。
  - 链路数据供可观测性看板使用，可还原完整 Agent 编排调用链。
"""

# 标准库：time 用于高精度计时；uuid 用于生成全局唯一 trace_id
import time
import uuid

# 项目内数据库模块：获取异步数据库会话工厂
from app.db import AsyncSessionLocal
# 项目内 ORM 模型：AgentTrace 对应数据库中存储链路追踪记录的表
from app.models import AgentTrace


class TraceCollector:
    """单次请求的链路追踪收集器。

    用法示例：
        collector = TraceCollector(tenant_id="t1", conversation_id="c1")
        with timer() as t:
            result = await some_agent.run(...)
        collector.step("SomeAgent", output_summary=result, duration_ms=t.ms)
        await collector.persist(message_id="m1")

    属性：
        trace_id:        本次请求的全局唯一追踪 ID，格式为 'trace_' + 16位十六进制。
        tenant_id:       租户 ID，用于多租户数据隔离。
        conversation_id: 所属会话 ID（可为 None，如后台任务场景）。
        message_id:      所属消息 ID，可在 persist 时传入覆盖。
        steps:           已收集的步骤列表，每个元素为一个节点的执行快照字典。
        _order:          内部计数器，记录当前已添加的步骤序号，用于还原执行顺序。
    """

    def __init__(self, tenant_id: str, conversation_id: str | None = None):
        """初始化链路收集器，自动生成唯一 trace_id。

        参数：
            tenant_id:       所属租户 ID。
            conversation_id: 所属会话 ID，可选（部分场景无会话概念）。
        """
        # 生成唯一 trace_id：前缀 'trace_' + UUID4 的前 16 位十六进制，兼顾可读性与唯一性
        self.trace_id = "trace_" + uuid.uuid4().hex[:16]
        self.tenant_id = tenant_id        # 租户 ID，落库时用于多租户隔离查询
        self.conversation_id = conversation_id  # 会话 ID，关联对话维度的链路查询
        self.message_id: str | None = None  # 消息 ID，默认为 None，persist 时可传入覆盖
        self.steps: list[dict] = []        # 存储所有已记录步骤的列表，顺序即执行顺序
        self._order = 0                    # 步骤序号计数器，每次 step() 调用时自增

    def step(
        self,
        agent_name: str,
        *,
        input_summary: str = "",
        output_summary: str = "",
        status: str = "ok",
        duration_ms: int = 0,
        token_usage: int = 0,
        hit_docs: list | None = None,
        error: str | None = None,
    ) -> None:
        """记录一个编排节点的执行情况，step_order 自增以还原链路顺序。

        参数：
            agent_name:     节点名称，如 'IntentAgent'、'DocRAGAgent'。
            input_summary:  节点输入的摘要（截断至 500 字符），避免存储过大内容。
            output_summary: 节点输出的摘要（截断至 500 字符）。
            status:         执行状态，通常为 'ok' 或 'error'。
            duration_ms:    节点执行耗时（毫秒），配合 timer 上下文管理器使用。
            token_usage:    本节点消耗的 token 数（仅 LLM 节点有值）。
            hit_docs:       RAG 命中的文档列表（文档 ID 或摘要），非 RAG 节点传 None。
            error:          异常信息字符串（status='error' 时填写），便于事后排查。

        返回：
            无（None）。
        """
        self._order += 1  # 步骤序号自增，确保多节点可按 step_order 还原执行顺序
        self.steps.append(  # 将本节点快照追加到步骤列表
            {
                "agent_name": agent_name,           # 节点名称，标识是哪个 Agent
                "step_order": self._order,           # 执行顺序编号，从 1 开始单调递增
                "input_summary": input_summary[:500],   # 截断输入摘要，防止超长内容占用存储
                "output_summary": output_summary[:500], # 截断输出摘要，同上
                "status": status,                    # 执行结果状态：'ok' 或 'error'
                "duration_ms": duration_ms,          # 耗时（毫秒），用于性能分析
                "token_usage": token_usage,          # token 消耗，用于成本分配
                "hit_docs": hit_docs or [],          # 命中文档列表，None 转为空列表保证类型一致
                "error_message": error,              # 异常信息，正常时为 None
            }
        )

    @property
    def total_tokens(self) -> int:
        """计算本次请求所有编排节点的 token 消耗总和。

        返回：
            整数，所有步骤的 token_usage 之和；无步骤时返回 0。

        用途：
            请求结束时汇总给 cost.record() 写入成本统计表。
        """
        # 遍历所有步骤，累加每个节点的 token_usage 字段
        return sum(s["token_usage"] for s in self.steps)

    async def persist(self, message_id: str | None = None) -> None:
        """将所有已收集的步骤批量写入数据库 agent_trace 表。

        参数：
            message_id: 覆盖实例属性 self.message_id 的消息 ID（可选）；
                        若传入则优先使用，否则退回 self.message_id。

        返回：
            无（None）。

        设计说明：
            在单个事务中批量插入所有步骤，保证同一次请求的链路数据原子落库，
            避免部分步骤写入失败导致链路断层。
        """
        async with AsyncSessionLocal() as s:  # 获取异步数据库会话，with 块结束自动关闭
            for step in self.steps:  # 遍历所有已记录步骤，逐条构造 ORM 对象
                s.add(
                    AgentTrace(  # 构造 ORM 对象，对应 agent_trace 表的一行
                        trace_id=self.trace_id,         # 链路 ID，将同一请求的所有步骤串联
                        conversation_id=self.conversation_id,  # 会话 ID
                        # message_id 优先使用传入参数，其次使用实例属性，实现灵活覆盖
                        message_id=message_id or self.message_id,
                        tenant_id=self.tenant_id,       # 租户 ID
                        **step,  # 展开步骤字典，将 agent_name/step_order/... 等字段直接传入
                    )
                )
            await s.commit()  # 提交事务，批量持久化所有步骤记录


class timer:
    """上下文管理器：测量耗时（毫秒）。

    使用示例：
        with timer() as t:
            await some_heavy_operation()
        print(t.ms)  # 输出耗时毫秒数

    属性：
        ms: with 块退出后可读取的耗时（毫秒，整型）。
    """

    def __enter__(self):
        """进入 with 块时记录起始时间戳（高精度）。

        返回：
            self，允许 `with timer() as t` 形式使用。
        """
        # perf_counter() 使用系统高精度时钟，比 time.time() 更适合测量短时间间隔
        self._t = time.perf_counter()
        return self  # 返回自身，使 `as t` 后可通过 t.ms 访问结果

    def __exit__(self, *exc):
        """退出 with 块时计算耗时并存储到 self.ms。

        参数：
            *exc: 异常信息三元组（exc_type, exc_val, traceback），不处理异常直接透传。

        返回：
            False，表示不抑制异常，with 块内的异常会正常向上传播。
        """
        # 计算从 __enter__ 到 __exit__ 的时间差，乘以 1000 转换为毫秒，取整存储
        self.ms = int((time.perf_counter() - self._t) * 1000)
        return False  # 返回 False 确保 with 块内异常不被吞掉，保持正常异常传播行为
