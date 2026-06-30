# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""种子数据：真实写入 MySQL（幂等，可重复执行）。

包含：
- 固定"剧本数据"：用于 4 个黄金场景可复现演示（如 401 的 req_20260615_8842）。
- 背景数据：近 30 天随机调用日志，让数据更真实。

用法（backend 目录下）：python -m scripts.seed_data

职责说明：
    本脚本是 DevSupport-AI 系统的种子数据初始化工具。执行后会清空并重建：
    套餐（Plan）、租户（Tenant）、用户（User）、应用（App）、API 密钥（ApiKey）、
    接口（ApiEndpoint）、错误码（ErrorCode）、调用日志（ApiCallLog）、
    用量记录（UsageRecord）、账单（Invoice）。
    所有写入操作幂等，可多次执行而不产生重复数据。
"""

import random  # 用于生成随机背景日志数据
from datetime import datetime, timedelta  # 用于生成各时间段的时间戳

from sqlalchemy import delete  # SQLAlchemy 的 DELETE 语句构造工具，用于清空表数据

# 导入同步数据库 Session 工厂，用于写入 MySQL
from app.db import SyncSessionLocal
# 导入所有 ORM 模型，对应 MySQL 中的各张表
from app.models import (
    ApiCallLog,    # API 调用日志模型
    ApiEndpoint,   # API 接口端点模型
    ApiKey,        # API 密钥模型
    App,           # 应用模型
    ErrorCode,     # 错误码手册模型
    Invoice,       # 账单模型
    Plan,          # 套餐模型
    Tenant,        # 租户模型
    UsageRecord,   # 用量记录模型
    User,          # 用户模型
)
from app.security import hash_password  # 密码哈希工具，用于对明文密码进行加密存储

random.seed(20260615)  # 固定随机种子，确保每次运行生成的背景数据一致，可复现
NOW = datetime(2026, 6, 15, 18, 0, 0)  # 以此时间点为"当前时间"基准，生成近 30 天数据

# ---------------- 套餐 ----------------
# 平台提供的四档套餐，差异在于 QPS 上限、月调用配额和单价
PLANS = [
    dict(id="plan_free", name="免费版", qps_limit=5, monthly_quota=10000,
         price_per_call=0.010, overage_price_per_call=0.020),       # 免费版：5 QPS，1 万次/月
    dict(id="plan_basic", name="基础版", qps_limit=20, monthly_quota=100000,
         price_per_call=0.008, overage_price_per_call=0.015),       # 基础版：20 QPS，10 万次/月
    dict(id="plan_pro", name="专业版", qps_limit=50, monthly_quota=500000,
         price_per_call=0.006, overage_price_per_call=0.012),       # 专业版：50 QPS，50 万次/月
    dict(id="plan_enterprise", name="企业版", qps_limit=200, monthly_quota=2000000,
         price_per_call=0.004, overage_price_per_call=0.008),       # 企业版：200 QPS，200 万次/月
]

# ---------------- 租户 ----------------
# 模拟 3 家外部客户租户 + 1 个内部平台租户（无绑定套餐）
TENANTS = [
    dict(id="t_acme", name="Acme 数据科技", plan_id="plan_pro"),              # Acme：专业版
    dict(id="t_globex", name="Globex 金融", plan_id="plan_enterprise"),       # Globex：企业版
    dict(id="t_initech", name="Initech 物流", plan_id="plan_basic"),          # Initech：基础版
    dict(id="t_platform", name="DevSupport 平台（内部）", plan_id=None),      # 内部平台：不绑定套餐
]

# ---------------- 用户（密码统一 password123） ----------------
PWD = "password123"  # 所有演示用户的统一明文密码，入库时会被哈希处理
USERS = [
    # Acme 租户：一名开发者 + 一名管理员
    dict(id="u_acme_dev", tenant_id="t_acme", username="dev_acme", role="customer_dev", display_name="Acme 开发者"),
    dict(id="u_acme_admin", tenant_id="t_acme", username="admin_acme", role="customer_admin", display_name="Acme 管理员"),
    # Globex 租户：一名开发者 + 一名管理员
    dict(id="u_globex_dev", tenant_id="t_globex", username="dev_globex", role="customer_dev", display_name="Globex 开发者"),
    dict(id="u_globex_admin", tenant_id="t_globex", username="admin_globex", role="customer_admin", display_name="Globex 管理员"),
    # Initech 租户：仅一名开发者
    dict(id="u_initech_dev", tenant_id="t_initech", username="dev_initech", role="customer_dev", display_name="Initech 开发者"),
    # 平台内部：两名技术支持 + 一名系统管理员
    dict(id="u_support1", tenant_id="t_platform", username="support1", role="support", display_name="技术支持-小王"),
    dict(id="u_support2", tenant_id="t_platform", username="support2", role="support", display_name="技术支持-小李"),
    dict(id="u_admin", tenant_id="t_platform", username="admin", role="admin", display_name="系统管理员"),
]

# ---------------- 应用与密钥 ----------------
# 每个租户下有一个生产应用
APPS = [
    dict(id="app_acme", tenant_id="t_acme", name="Acme 生产应用"),
    dict(id="app_globex", tenant_id="t_globex", name="Globex 风控应用"),
    dict(id="app_initech", tenant_id="t_initech", name="Initech 物流应用"),
]
API_KEYS = [
    # Acme 一把已过期的 key（401 场景）+ 一把有效 key
    # 过期 key 用于演示 401 鉴权失败（AUTH_KEY_EXPIRED）场景
    dict(id="key_acme_expired", app_id="app_acme", tenant_id="t_acme",
         key_masked="ak_live_****8a2f", status="EXPIRED", expire_at=datetime(2026, 6, 10)),  # 已于 6/10 过期
    dict(id="key_acme_active", app_id="app_acme", tenant_id="t_acme",
         key_masked="ak_live_****3c7d", status="ACTIVE", expire_at=datetime(2027, 1, 1)),    # 有效至 2027 年
    dict(id="key_globex_active", app_id="app_globex", tenant_id="t_globex",
         key_masked="ak_live_****b19e", status="ACTIVE", expire_at=datetime(2027, 1, 1)),    # Globex 有效密钥
    dict(id="key_initech_active", app_id="app_initech", tenant_id="t_initech",
         key_masked="ak_live_****6d40", status="ACTIVE", expire_at=datetime(2027, 1, 1)),    # Initech 有效密钥
]

# ---------------- 接口 ----------------
# 平台对外开放的 4 个 API 接口，覆盖实名认证、数据查询、风控评分等产品
ENDPOINTS = [
    dict(id="ep_idcard", product="实名认证", path="/v1/idcard/verify", name="身份证实名核验"),
    dict(id="ep_company", product="数据查询", path="/v1/company/query", name="企业信息查询"),
    dict(id="ep_risk", product="风控评分", path="/v1/risk/score", name="风控评分"),
    dict(id="ep_bankcard", product="实名认证", path="/v1/bankcard/verify", name="银行卡核验"),
]
ENDPOINT_PATHS = [e["path"] for e in ENDPOINTS]  # 仅提取路径字符串，便于随机抽取

# ---------------- 错误码手册（与知识库《错误码手册》一致） ----------------
# 错误码定义与知识库保持同步，AI 在回答错误码相关问题时可引用这份数据
ERROR_CODES = [
    dict(code="AUTH_KEY_EXPIRED", name="API Key 已过期", http_status=401,
         cause="请求使用的 API Key 已超过有效期。",
         fix_steps="1. 登录控制台查看 Key 状态与过期时间；2. 重新生成 API Key；3. 更新服务端配置后重试。"),
    dict(code="AUTH_KEY_INVALID", name="API Key 无效", http_status=401,
         cause="API Key 不存在、被禁用或格式错误。",
         fix_steps="1. 核对 Key 是否复制完整；2. 确认 Key 未被禁用；3. 使用控制台有效 Key 重试。"),
    dict(code="SIGN_INVALID", name="签名错误", http_status=401,
         cause="请求签名与服务端计算不一致，常见于参数排序、时间戳或密钥错误。",
         fix_steps="1. 按文档对参数字典序排序拼接；2. 确认使用正确 Secret；3. 校验时间戳在 5 分钟内；4. 重新计算签名。"),
    dict(code="PERMISSION_DENIED", name="权限不足", http_status=403,
         cause="当前应用无该接口或资源的访问权限。",
         fix_steps="1. 确认应用已开通该 API 产品；2. 联系管理员授权；3. 重试。"),
    dict(code="PRODUCT_NOT_ENABLED", name="接口未开通", http_status=403,
         cause="该 API 产品尚未为应用开通。",
         fix_steps="1. 控制台开通对应 API 产品；2. 等待生效后重试。"),
    dict(code="IP_NOT_ALLOWED", name="IP 不在白名单", http_status=403,
         cause="请求来源 IP 不在应用配置的白名单内。",
         fix_steps="1. 控制台将服务器出口 IP 加入白名单；2. 等待生效后重试。"),
    dict(code="NOT_FOUND", name="接口不存在", http_status=404,
         cause="请求路径错误或接口已下线。",
         fix_steps="1. 核对接口路径与版本；2. 参考最新文档。"),
    dict(code="REQUEST_TIMEOUT", name="请求超时", http_status=408,
         cause="请求处理超时，可能为网络或上游慢。",
         fix_steps="1. 增大客户端超时时间；2. 重试；3. 持续出现请提工单。"),
    dict(code="RATE_LIMIT_EXCEEDED", name="QPS 超限", http_status=429,
         cause="单位时间请求数超过套餐 QPS 限制。",
         fix_steps="1. 客户端做本地限速；2. 采用指数退避重试；3. 错峰调用；4. 如需更高并发请升级套餐。"),
    dict(code="QUOTA_EXCEEDED", name="调用量超额", http_status=429,
         cause="本月调用量已超出套餐配额。",
         fix_steps="1. 查看本月用量；2. 升级套餐或购买加量包；3. 次月自动恢复。"),
    dict(code="INTERNAL_ERROR", name="服务端错误", http_status=500,
         cause="服务端内部异常。",
         fix_steps="1. 稍后重试；2. 记录 request_id；3. 持续出现请提工单。"),
    dict(code="PARAM_MISSING", name="参数缺失", http_status=400,
         cause="缺少必填参数。",
         fix_steps="1. 对照文档补齐必填参数；2. 重试。"),
    dict(code="PARAM_INVALID", name="参数格式错误", http_status=400,
         cause="参数类型或格式不符合要求。",
         fix_steps="1. 核对参数类型与格式；2. 修正后重试。"),
    dict(code="BALANCE_NOT_ENOUGH", name="余额不足", http_status=402,
         cause="账户余额不足以完成本次计费调用。",
         fix_steps="1. 控制台充值；2. 充值后重试。"),
    dict(code="DATA_EMPTY", name="查询结果为空", http_status=200,
         cause="请求成功但未命中数据，可能为参数不匹配或数据未覆盖。",
         fix_steps="1. 核对查询参数；2. 确认数据覆盖范围；3. 如确认应有数据请提数据质量工单。"),
    dict(code="DATA_INCONSISTENT", name="数据不一致", http_status=200,
         cause="返回数据与预期来源存在差异，可能为更新延迟。",
         fix_steps="1. 查看数据更新时间；2. 提供对比样例提数据质量工单。"),
    dict(code="WEBHOOK_DELIVERY_FAILED", name="回调投递失败", http_status=200,
         cause="平台已发送回调但客户地址返回非 2xx 或不可达。",
         fix_steps="1. 确认回调地址可公网访问；2. 返回 200 表示已接收；3. 平台会按策略重试。"),
    dict(code="WEBHOOK_SIGN_INVALID", name="回调验签失败", http_status=200,
         cause="客户侧对回调验签失败。",
         fix_steps="1. 使用回调密钥按文档验签；2. 注意原始 body 不要被改写。"),
]


def _clear(session):
    """清空所有业务表的数据，按外键依赖顺序（子表优先）删除。

    参数：
        session: SQLAlchemy 同步 Session 对象。
    说明：
        此函数确保重新运行种子脚本时不会产生主键冲突，实现幂等效果。
        删除顺序遵循外键依赖：先删子表（ApiCallLog/ApiKey/...），后删父表（Tenant/Plan）。
    """
    for model in (ApiCallLog, ApiKey, App, ApiEndpoint, ErrorCode, Invoice,
                  UsageRecord, User, Tenant, Plan):  # 按外键依赖逆序遍历，子表优先删除
        session.execute(delete(model))  # 执行 DELETE 语句清空该表


def _gen_background_logs() -> list[dict]:
    """近 30 天背景调用日志：多数 200，少量各类错误。

    返回：
        list[dict]: 调用日志字典列表，每条记录包含 request_id、tenant_id、
                    app_id、api_key_id、endpoint、http_status、error_code、
                    latency_ms、client_ip、created_at 等字段。
    说明：
        生成的背景日志用于充实数据库，使数据分布更真实，
        不与"黄金剧本"日志冲突，request_id 格式为 req_YYYYMMDD_NNNN。
    """
    rows = []
    tenants_apps = [("t_acme", "app_acme", "key_acme_active"),
                    ("t_globex", "app_globex", "key_globex_active"),
                    ("t_initech", "app_initech", "key_initech_active")]  # 三个租户-应用-密钥组合
    err_pool = ["PARAM_INVALID", "PARAM_MISSING", "RATE_LIMIT_EXCEEDED",
                "INTERNAL_ERROR", "SIGN_INVALID", "PERMISSION_DENIED"]  # 背景日志中可能出现的错误码池
    seq = 0  # 全局序列号，用于生成唯一 request_id 后缀
    for day in range(30):  # 遍历过去 30 天
        ts_day = NOW - timedelta(days=day)  # 计算当天的基准日期
        for _ in range(random.randint(35, 50)):  # 每天随机生成 35~50 条日志
            seq += 1  # 递增序列号，保证 request_id 唯一
            tenant, app, key = random.choice(tenants_apps)  # 随机选择一个租户-应用-密钥组合
            endpoint = random.choice(ENDPOINT_PATHS)  # 随机选择一个 API 接口路径
            ts = ts_day.replace(hour=random.randint(0, 23), minute=random.randint(0, 59),
                                second=random.randint(0, 59))  # 在当天随机时间点创建时间戳
            if random.random() < 0.85:  # 85% 概率为成功请求（模拟真实高成功率）
                status, err = 200, None  # 成功状态：HTTP 200，无错误码
            else:
                err = random.choice(err_pool)  # 从错误池中随机选取一个错误码
                status = next(e["http_status"] for e in ERROR_CODES if e["code"] == err)  # 根据错误码查找对应 HTTP 状态码
            rid = f"req_{ts.strftime('%Y%m%d')}_{seq:04d}"  # 生成格式化 request_id，如 req_20260615_0001
            rows.append(dict(request_id=rid, tenant_id=tenant, app_id=app, api_key_id=key,
                             endpoint=endpoint, http_status=status, error_code=err,
                             latency_ms=random.randint(40, 800),  # 随机模拟 40~800ms 的响应延迟
                             client_ip=f"203.0.113.{random.randint(2, 250)}", created_at=ts))  # 使用 TEST-NET IP 段
    return rows


def _gen_scripted_logs() -> list[dict]:
    """固定剧本日志：4 个黄金场景可复现。

    返回：
        list[dict]: 预设的场景日志列表，request_id 固定，便于演示时精准查询。
    说明：
        黄金场景包括：
        ① 401 鉴权失败（Acme 使用过期 Key 调用身份证核验）
        ② 429 限流（Globex 风控接口下午突发大量请求）
        ③ 数据质量（Acme 查询企业信息成功但客户认为数据不一致）
        附加：签名错误样例（Initech 银行卡核验）
    """
    rows = []
    # 场景①：401 鉴权失败（Key 过期）
    rows.append(dict(request_id="req_20260615_8842", tenant_id="t_acme", app_id="app_acme",
                     api_key_id="key_acme_expired", endpoint="/v1/idcard/verify",
                     http_status=401, error_code="AUTH_KEY_EXPIRED", latency_ms=35,
                     client_ip="203.0.113.10", created_at=datetime(2026, 6, 15, 14, 22, 0)))  # 固定时间点确保可复现
    # 场景②：429 限流（Globex 风控接口下午突发大量 429）
    for i in range(18):  # 生成 18 条 429 日志，模拟集中爆发的限流场景
        ts = datetime(2026, 6, 15, 15, random.randint(0, 30), random.randint(0, 59))  # 15:00~15:30 之间随机
        rows.append(dict(request_id=f"req_20260615_90{i:02d}", tenant_id="t_globex",
                         app_id="app_globex", api_key_id="key_globex_active",
                         endpoint="/v1/risk/score", http_status=429,
                         error_code="RATE_LIMIT_EXCEEDED", latency_ms=12,  # 限流时延迟极低（直接拒绝）
                         client_ip="198.51.100.7", created_at=ts))
    # 场景⑤：数据质量（请求成功但客户认为数据不一致）
    rows.append(dict(request_id="req_20260614_5521", tenant_id="t_acme", app_id="app_acme",
                     api_key_id="key_acme_active", endpoint="/v1/company/query",
                     http_status=200, error_code=None, latency_ms=220,  # 请求本身成功，HTTP 200
                     client_ip="203.0.113.10", created_at=datetime(2026, 6, 14, 10, 5, 0)))
    # 附加：签名错误样例
    rows.append(dict(request_id="req_20260613_3302", tenant_id="t_initech", app_id="app_initech",
                     api_key_id="key_initech_active", endpoint="/v1/bankcard/verify",
                     http_status=401, error_code="SIGN_INVALID", latency_ms=28,
                     client_ip="192.0.2.55", created_at=datetime(2026, 6, 13, 9, 12, 0)))
    return rows


def _gen_usage_and_invoices():
    """用量与账单：t_acme 6 月环比大涨且产生超额（账单解释场景）。

    返回：
        tuple[list[dict], list[dict]]: 用量记录列表 和 账单列表。
    说明：
        Acme 6 月调用量 56 万，超出专业版 50 万配额，产生超额计费，
        用于演示账单费用解释场景（AI 能计算并解释为何账单金额高于预期）。
    """
    plan_by_tenant = {t["id"]: t["plan_id"] for t in TENANTS}  # 构建 租户ID -> 套餐ID 映射
    plan_map = {p["id"]: p for p in PLANS}  # 构建 套餐ID -> 套餐详情 映射
    # (tenant, month, calls) 三元组：各租户各月的调用量
    usage_plan = [
        ("t_acme", "2026-04", 210000),
        ("t_acme", "2026-05", 215000),
        ("t_acme", "2026-06", 560000),   # 环比大涨，超过专业版 50万配额，会产生超额费用
        ("t_globex", "2026-05", 1500000),
        ("t_globex", "2026-06", 1620000),
        ("t_initech", "2026-05", 60000),
        ("t_initech", "2026-06", 72000),
    ]
    usage_rows, invoice_rows = [], []  # 分别收集用量记录和账单记录
    for tenant, month, calls in usage_plan:
        plan = plan_map[plan_by_tenant[tenant]]  # 获取该租户的套餐详情
        quota = plan["monthly_quota"]  # 套餐包含的月度配额（次）
        overage = max(0, calls - quota)  # 超出配额的调用次数（不足 0 时取 0）
        base_calls = min(calls, quota)   # 套餐内的调用次数（不超过配额上限）
        base_fee = round(base_calls * plan["price_per_call"], 2)  # 套餐内费用（保留 2 位小数）
        overage_fee = round(overage * plan["overage_price_per_call"], 2)  # 超额费用
        total = round(base_fee + overage_fee, 2)  # 当月总账单金额
        usage_rows.append(dict(tenant_id=tenant, month=month, call_count=calls,
                               overage_count=overage))  # 记录用量：总量 + 超额量
        invoice_rows.append(dict(id=f"inv_{tenant}_{month.replace('-', '')}", tenant_id=tenant,
                                 month=month,
                                 items=dict(plan=plan["name"], base_calls=base_calls,
                                            base_fee=base_fee, overage_calls=overage,
                                            overage_fee=overage_fee, total=total),  # 账单明细 JSON
                                 amount=total, status="ISSUED"))  # 账单状态：已出具
    return usage_rows, invoice_rows


def main() -> None:
    """主函数：清空数据库并按外键依赖顺序写入所有种子数据。

    执行流程：
        1. 清空全部业务表（_clear）
        2. 按 Plan -> Tenant -> User -> App -> ApiKey -> ApiEndpoint -> ErrorCode 顺序写入基础数据
        3. 写入调用日志（黄金剧本 + 背景日志）
        4. 写入用量记录和账单
        5. 提交事务并打印统计摘要
    """
    with SyncSessionLocal() as s:
        _clear(s)  # 先清空所有业务表，保证幂等性
        s.flush()  # 将 DELETE 操作刷入数据库（但不提交），确保后续 INSERT 不冲突
        # 按外键依赖顺序逐组 flush
        s.add_all([Plan(**p) for p in PLANS]); s.flush()        # 先写套餐（无外键依赖）
        s.add_all([Tenant(**t) for t in TENANTS]); s.flush()    # 租户依赖套餐
        s.add_all([User(password_hash=hash_password(PWD), **u) for u in USERS]); s.flush()  # 用户依赖租户，密码哈希后存储
        s.add_all([App(**a) for a in APPS]); s.flush()          # 应用依赖租户
        s.add_all([ApiKey(**k) for k in API_KEYS]); s.flush()   # 密钥依赖应用和租户
        s.add_all([ApiEndpoint(**e) for e in ENDPOINTS]); s.flush()  # 接口端点（独立，无外键）
        s.add_all([ErrorCode(**e) for e in ERROR_CODES]); s.flush()  # 错误码（独立，无外键）

        logs = _gen_scripted_logs() + _gen_background_logs()  # 合并黄金剧本日志和背景日志
        s.add_all([ApiCallLog(**row) for row in logs]); s.flush()  # 写入所有调用日志

        usage_rows, invoice_rows = _gen_usage_and_invoices()  # 生成用量记录和账单数据
        s.add_all([UsageRecord(**u) for u in usage_rows])   # 写入用量记录
        s.add_all([Invoice(**inv) for inv in invoice_rows])  # 写入账单

        s.commit()  # 提交事务，所有数据持久化到 MySQL

        print(f"[seed] 套餐 {len(PLANS)}、租户 {len(TENANTS)}、用户 {len(USERS)}、"
              f"应用 {len(APPS)}、密钥 {len(API_KEYS)}、接口 {len(ENDPOINTS)}、"
              f"错误码 {len(ERROR_CODES)}")
        print(f"[seed] 调用日志 {len(logs)}（含剧本 {len(_gen_scripted_logs())}）、"
              f"用量 {len(usage_rows)}、账单 {len(invoice_rows)}")
        print("[seed] 完成。统一密码: password123")


if __name__ == "__main__":
    main()  # 直接运行脚本时执行主函数
