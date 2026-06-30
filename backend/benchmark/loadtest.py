# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""并发压测：测 P50/P95/吞吐，并基于 AgentTrace 做阶段耗时分解。

对比「冷启动(无缓存)」与「缓存预热」两轮，量化语义缓存的优化效果。
用法（backend 目录下）：python -m benchmark.loadtest

职责说明：
    本模块是 DevSupport-AI 系统的并发压测工具，负责：
    1. 以并发方式向 supervisor Agent 发送标准问题集，模拟真实并发负载。
    2. 统计延迟分布（P50/P95/avg/max）和吞吐量（req/s）。
    3. 从 AgentTrace 表查询各 Agent 阶段的平均耗时，定位性能瓶颈。
    4. 先跑冷启动（清空 Redis 缓存），再跑缓存预热，对比 P95 延迟降幅，
       量化语义缓存（semcache）对性能的优化效果。
"""

import asyncio    # 标准库：异步事件循环，用于并发发送请求
import statistics  # 标准库：统计计算（均值等），用于延迟分析
import time        # 标准库：高精度计时，用于测量请求延迟和总耗时

from sqlalchemy import select  # SQLAlchemy 查询构造工具，用于查询 AgentTrace 表

from app.agents import supervisor           # AI Agent 调度器，压测的核心调用目标
from app.cache.redis_client import get_redis  # Redis 客户端工厂，用于清空语义缓存
from app.db import AsyncSessionLocal, async_engine  # 异步数据库 Session 工厂和引擎
from app.models import AgentTrace           # AgentTrace ORM 模型，记录每次 Agent 调用的阶段耗时

# 标准压测问题（以可缓存的文档问答为主，混合诊断/账单）
# 覆盖签名、Webhook、限流、接入流程、账单、错误码等典型场景
QUESTIONS = [
    ("签名算法怎么生成？", "t_acme", "u_acme_dev"),
    ("Webhook 回调收不到怎么排查", "t_acme", "u_acme_dev"),
    ("429 限流了怎么办", "t_acme", "u_acme_dev"),
    ("接入 API 的流程是什么", "t_acme", "u_acme_dev"),
    ("账单费用是怎么计算的", "t_acme", "u_acme_dev"),
    ("SIGN_INVALID 是什么原因", "t_acme", "u_acme_dev"),
]

CONCURRENCY = 4   # 并发度：同时在途请求数，使用 asyncio.Semaphore 控制
REPEAT = 2        # 每个问题重复次数，放大并发量（总请求数 = len(QUESTIONS) * REPEAT）


def _percentile(data: list[float], p: float) -> float:
    """计算列表数据的第 p 百分位数（最近排名法）。

    参数：
        data: 浮点数列表，通常为延迟样本（单位秒）。
        p:    百分位数，取值 0~100，如 50 表示中位数，95 表示 P95。
    返回：
        float: 第 p 百分位对应的值；列表为空时返回 0.0。
    """
    if not data:  # 空列表直接返回 0，避免除零错误
        return 0.0
    s = sorted(data)  # 对延迟数据升序排列，百分位计算的前提
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))  # 计算百分位对应的索引（边界截断）
    return s[k]  # 返回该索引处的延迟值


async def _run_batch(label: str) -> tuple[list[float], list[str]]:
    """并发执行一批压测请求，统计延迟并打印汇总指标。

    参数：
        label: 批次标签，如 "冷启动-无缓存" 或 "缓存预热"，用于区分打印输出。
    返回：
        tuple[list[float], list[str]]:
            latencies: 每条请求的端到端延迟（秒）列表。
            trace_ids: 每条请求对应的 AgentTrace ID 列表，供阶段耗时分析使用。
    """
    sem = asyncio.Semaphore(CONCURRENCY)  # 并发控制信号量，限制同时在途请求不超过 CONCURRENCY 个
    latencies: list[float] = []   # 收集每条请求的延迟（秒）
    trace_ids: list[str] = []     # 收集每条请求的 trace_id，用于后续查询 AgentTrace
    queries = [q for q in QUESTIONS for _ in range(REPEAT)]  # 将每个问题重复 REPEAT 次，生成完整请求列表

    async def one(idx: int, q, tenant, user):
        """单条请求的异步执行函数，通过信号量控制并发。

        参数：
            idx:    请求序号，用于生成唯一 conversation_id。
            q:      问题文本。
            tenant: 租户 ID。
            user:   用户 ID。
        返回：
            bool: 是否命中语义缓存（from_cache 字段）。
        """
        async with sem:  # 申请信号量，超过并发上限时自动等待，确保并发度受控
            t0 = time.perf_counter()  # 记录请求开始时间（高精度计时）
            r = await supervisor.run(query=q, tenant_id=tenant, user_id=user,
                                     conversation_id=f"bench_{label}_{idx}")  # 调用 supervisor Agent
            dt = time.perf_counter() - t0  # 计算本次请求的端到端延迟（秒）
            latencies.append(dt)  # 将延迟追加到列表
            trace_ids.append(r["trace_id"])  # 记录 trace_id，供阶段耗时分析
            return r.get("from_cache", False)  # 返回是否命中缓存

    t0 = time.perf_counter()  # 记录整批请求的开始时间
    flags = await asyncio.gather(*[one(i, q, t, u) for i, (q, t, u) in enumerate(queries)])  # 并发执行所有请求
    wall = time.perf_counter() - t0  # 计算整批请求的总耗时（秒）
    cache_hits = sum(1 for f in flags if f)  # 统计本批次中命中语义缓存的请求数
    print(f"\n[{label}] 请求数={len(queries)} 并发={CONCURRENCY} 总耗时={wall:.2f}s "
          f"吞吐={len(queries)/wall:.2f} req/s 缓存命中={cache_hits}")  # 打印吞吐和缓存命中概况
    print(f"  延迟 P50={_percentile(latencies,50):.2f}s P95={_percentile(latencies,95):.2f}s "
          f"avg={statistics.mean(latencies):.2f}s max={max(latencies):.2f}s")  # 打印延迟分位数指标
    return latencies, trace_ids  # 返回延迟列表和 trace_id 列表


async def _stage_breakdown(trace_ids: list[str]) -> None:
    """查询 AgentTrace 表，按 Agent 名称聚合并打印各阶段平均耗时。

    参数：
        trace_ids: 需要分析的 trace_id 列表（来自一批压测请求）。
    说明：
        通过对 AgentTrace 按 agent_name 分组，计算每个 Agent 阶段的平均耗时（ms），
        按耗时从高到低排序输出，帮助识别延迟瓶颈所在的 Agent 阶段。
    """
    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(select(AgentTrace).where(AgentTrace.trace_id.in_(trace_ids)))
            # 查询所有属于本批次的 AgentTrace 记录
        ).scalars().all()  # 将查询结果转换为 ORM 对象列表
    agg: dict[str, list[int]] = {}  # 按 agent_name 聚合耗时数据：agent_name -> [duration_ms, ...]
    for r in rows:
        agg.setdefault(r.agent_name, []).append(r.duration_ms)  # 将每条 trace 的耗时追加到对应 Agent 的列表
    print("  阶段平均耗时(ms):")
    for name, ds in sorted(agg.items(), key=lambda x: -statistics.mean(x[1])):  # 按平均耗时降序排列，耗时最高的先输出
        print(f"    {name:18s} avg={statistics.mean(ds):7.0f}  count={len(ds)}")  # 打印 Agent 名、平均耗时、调用次数


async def main() -> None:
    """主异步函数：执行冷启动和缓存预热两轮压测，并输出优化对比结果。

    执行流程：
        1. 清空 t_acme 租户的语义缓存（semcache），模拟冷启动状态。
        2. 运行第一轮压测（冷启动-无缓存），记录延迟并做阶段耗时分解。
        3. 运行第二轮压测（缓存预热），此时 doc_qa 类问题应命中语义缓存。
        4. 对比两轮 P95 延迟，计算缓存优化带来的降幅百分比。
        5. 释放数据库连接池。
    """
    r = get_redis()  # 获取 Redis 异步客户端
    await r.delete("semcache:t_acme")  # 清空 Acme 租户的语义缓存，确保第一轮为真正的冷启动

    # 冷启动（缓存空）
    cold_lat, cold_traces = await _run_batch("冷启动-无缓存")  # 第一轮：无缓存，测量真实推理延迟
    await _stage_breakdown(cold_traces)  # 分析冷启动阶段耗时，定位各 Agent 的延迟贡献

    # 缓存预热后再跑（doc_qa 命中缓存）
    warm_lat, _ = await _run_batch("缓存预热")  # 第二轮：相同问题命中语义缓存，延迟应大幅降低

    p95_cold, p95_warm = _percentile(cold_lat, 95), _percentile(warm_lat, 95)  # 分别计算两轮 P95 延迟
    print(f"\n==== 优化对比 ====\n  P95: 冷启动 {p95_cold:.2f}s -> 预热 {p95_warm:.2f}s "
          f"(降低 {(1-p95_warm/p95_cold)*100:.0f}%)")  # 打印 P95 对比及相对降幅
    await async_engine.dispose()  # 关闭异步数据库连接池，避免进程挂起


if __name__ == "__main__":
    asyncio.run(main())  # 直接运行脚本时启动事件循环并执行主函数
