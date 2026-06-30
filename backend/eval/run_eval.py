# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""评估集运行器：真实跑多 Agent 链路并计算各项指标 + Badcase 归因。

指标：意图准确率、实体抽取准确率、引用率、转人工判定准确率、澄清判定准确率、脱敏准确率。
用法（backend 目录下）：python -m eval.run_eval

职责说明：
    本模块是 DevSupport-AI 系统的离线评测工具，负责：
    1. 从 dataset.jsonl 加载评测案例，逐条调用 supervisor Agent 链路（真实推理）。
    2. 将模型输出与标注的 expected_intent、expected_entities、expect_citations 等字段对比。
    3. 汇总意图准确率、实体抽取准确率、引用率、转人工判定准确率、澄清准确率、脱敏准确率。
    4. 对意图或转人工判定错误的案例输出 Badcase 归因列表，方便定位问题。
    5. 从 security_set.jsonl 加载脱敏测试集，验证敏感信息是否被正确脱敏。
    评估前会清空 Redis 语义缓存和路由缓存，确保每次评测走真实推理链路，结果可信。
"""

import asyncio  # 标准库：异步事件循环，用于驱动异步评估流程
import json     # 标准库：JSON 解析，用于读取 JSONL 格式的评测数据集
from pathlib import Path  # 标准库：路径操作，用于定位评测数据文件

from app.agents import supervisor         # AI Agent 调度器（supervisor），是系统的核心推理入口
from app.cache.redis_client import get_redis  # Redis 客户端工厂，用于在评估前清空缓存
from app.guardrail import desensitize     # 脱敏模块，包含 desensitize_text 和 detect 函数

EVAL_DIR = Path(__file__).resolve().parent  # 评测数据目录：与本脚本同级，即 backend/eval/


def _load(name: str) -> list[dict]:
    """从 EVAL_DIR 下加载指定 JSONL 文件，返回解析后的字典列表。

    参数：
        name: 文件名，如 "dataset.jsonl" 或 "security_set.jsonl"。
    返回：
        list[dict]: 每行 JSON 解析后的字典列表（自动跳过空行）。
    """
    return [json.loads(l) for l in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    # 逐行读取文件，跳过空行后逐行解析 JSON，收集为列表


async def evaluate() -> dict:
    """异步执行完整评估流程，返回各项指标和 Badcase 列表。

    返回：
        dict，包含以下字段：
            total_cases (int)             : 评测案例总数
            intent_accuracy (float)       : 意图识别准确率（0~1）
            entity_accuracy (float|None)  : 实体抽取准确率（无实体标注时为 None）
            citation_rate (float|None)    : 需要引用的案例中实际产生引用的比例（无需引用时为 None）
            human_transfer_accuracy (float): 转人工判定准确率
            clarify_accuracy (float)      : 澄清判定准确率
            desensitization_accuracy (float): 脱敏准确率
            badcases (list[dict])         : 意图或转人工判定错误的案例归因列表
    """
    cases = _load("dataset.jsonl")  # 加载主评测数据集，包含问题、预期意图、预期实体等标注信息
    # 清理语义缓存 + 路由缓存，保证每次评估走真实链路
    r = get_redis()  # 获取 Redis 异步客户端
    for t in {c["tenant_id"] for c in cases}:  # 遍历所有涉及的租户 ID（去重）
        await r.delete(f"semcache:{t}")  # 删除该租户的语义缓存，防止缓存命中掩盖真实推理结果
    async for k in r.scan_iter("routecache:*"):  # 遍历所有路由缓存 key（通配符扫描）
        await r.delete(k)  # 删除每条路由缓存，确保路由决策走真实逻辑

    # 初始化各项指标计数器
    intent_ok = entity_total = entity_ok = 0  # 意图正确数；实体总数；实体正确数
    cite_need = cite_ok = human_ok = clarify_ok = 0  # 需引用数；引用正确数；转人工正确数；澄清正确数
    badcases = []  # 收集意图或转人工判定错误的案例，用于 Badcase 归因分析

    for i, c in enumerate(cases):  # 逐案例评测，i 用于生成唯一 conversation_id
        res = await supervisor.run(
            query=c["query"], tenant_id=c["tenant_id"], user_id=c["user_id"],
            conversation_id=f"eval_{c['id']}_{i}", is_internal=False,  # 以外部用户身份运行，模拟真实场景
        )
        # 意图
        intent_correct = res.get("intent") == c["expected_intent"]  # 判断模型意图是否与标注一致
        intent_ok += intent_correct  # 正确则累加计数（True 等于 1）
        # 实体
        for k, v in (c.get("expected_entities") or {}).items():  # 遍历标注的预期实体键值对
            entity_total += 1  # 实体总数 +1
            if res.get("entities", {}).get(k) == v:  # 判断模型抽取的实体值是否与标注一致
                entity_ok += 1  # 实体正确数 +1
        # 引用
        if c.get("expect_citations"):  # 该案例标注为需要引用知识库文档
            cite_need += 1  # 需引用案例数 +1
            if res.get("citations"):  # 模型响应中包含 citations 字段（非空）
                cite_ok += 1  # 引用命中数 +1
        # 转人工
        human_correct = bool(res.get("need_human")) == bool(c.get("expect_human"))  # 转人工标志与标注一致则为 True
        human_ok += human_correct  # 转人工判定正确数 +1
        # 澄清
        clarify_correct = bool(res.get("need_clarify")) == bool(c.get("expect_clarify", False))  # 澄清标志与标注一致
        clarify_ok += clarify_correct  # 澄清判定正确数 +1

        if not (intent_correct and human_correct):  # 意图或转人工任一错误，则记录为 Badcase
            badcases.append({
                "id": c["id"], "query": c["query"],
                "expected_intent": c["expected_intent"], "got_intent": res.get("intent"),  # 标注意图 vs 模型意图
                "expect_human": bool(c.get("expect_human")), "got_human": bool(res.get("need_human")),  # 标注转人工 vs 模型判断
                "attribution": "意图识别" if not intent_correct else "转人工判定",  # 归因：哪个维度出错
            })

    n = len(cases)  # 案例总数，用于计算准确率分母
    # 脱敏评估
    sec = _load("security_set.jsonl")  # 加载脱敏测试集（包含含敏感信息的文本和期望脱敏类型标注）
    sec_ok = 0  # 脱敏正确案例数
    for s in sec:
        clean = desensitize.desensitize_text(s["text"])  # 对测试文本执行脱敏处理，得到脱敏后文本
        if not s["types"]:  # 该文本不含敏感信息（无需脱敏）
            ok = clean == s["text"]  # 脱敏后文本应与原文完全一致（无误脱敏）
        else:
            ok = len(desensitize.detect(s["text"])) > 0 and len(desensitize.detect(clean)) == 0
            # 原文应检测到敏感信息（detect 非空），脱敏后文本不应再有敏感信息（detect 为空）
        sec_ok += ok  # 脱敏正确数累加

    # 汇总并返回所有指标
    return {
        "total_cases": n,                                                                         # 总案例数
        "intent_accuracy": round(intent_ok / n, 3),                                              # 意图准确率
        "entity_accuracy": round(entity_ok / entity_total, 3) if entity_total else None,         # 实体准确率（无实体标注时为 None）
        "citation_rate": round(cite_ok / cite_need, 3) if cite_need else None,                   # 引用率（无需引用案例时为 None）
        "human_transfer_accuracy": round(human_ok / n, 3),                                       # 转人工判定准确率
        "clarify_accuracy": round(clarify_ok / n, 3),                                            # 澄清判定准确率
        "desensitization_accuracy": round(sec_ok / len(sec), 3),                                 # 脱敏准确率
        "badcases": badcases,                                                                     # Badcase 归因列表
    }


def main() -> None:
    """主函数：驱动评估流程，打印评估报告，并释放数据库连接。

    执行流程：
        1. 在 asyncio 事件循环中运行 evaluate()。
        2. 打印各项指标（跳过 badcases 字段单独处理）。
        3. 逐条打印 Badcase，显示意图预期/实际值和归因标签。
        4. 释放 async_engine 连接池（避免进程挂起）。
    """
    from app.db import async_engine  # 在函数内延迟导入，避免模块加载时初始化数据库连接

    async def _run():
        """内部异步包装函数，执行评估并打印结果。"""
        m = await evaluate()  # 执行完整评估，获取指标字典
        print("==== 评估报告 ====")
        for k, v in m.items():
            if k != "badcases":  # 非 badcases 字段直接打印键值
                print(f"  {k}: {v}")
        print(f"  badcases ({len(m['badcases'])}):")  # 打印 Badcase 数量
        for b in m["badcases"]:
            print(f"    - {b['id']} [{b['attribution']}] {b['query'][:30]} "
                  f"intent {b['expected_intent']}→{b['got_intent']}")  # 显示案例ID、归因、问题摘要、意图对比
        await async_engine.dispose()  # 关闭异步数据库连接池，确保进程正常退出

    asyncio.run(_run())  # 启动事件循环并运行内部异步函数


if __name__ == "__main__":
    main()  # 直接运行脚本时执行主函数
