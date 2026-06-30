# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""ORM 模型：对应 PRD 数据模型（§10 / §13）。

所有业务数据真实存储于 MySQL；知识库切片向量存于 Milvus（见 app/rag/store.py）。

本模块定义了 DevSupport AI 系统的全部数据库表结构，使用 SQLAlchemy 2.x
的 Mapped / mapped_column 声明式风格定义 ORM 模型。每个类对应 MySQL 中的一张表。

表分组概览：
  - 租户/用户/应用/密钥：多租户 SaaS 基础数据
  - 接口/调用日志/错误码：API 平台核心业务数据
  - 套餐/用量/账单：计费相关数据
  - 知识库文档：RAG 知识库元数据（向量切片在 Milvus）
  - 会话/消息：用户与 AI 的对话记录
  - 可观测：Agent 链路追踪和工具调用日志
  - 工单/反馈/审计：工单系统和操作审计
  - Token 用量：LLM 调用成本统计
"""

# 标准库：datetime 用于字段类型注解和默认值函数的返回类型
from datetime import datetime

# SQLAlchemy 列类型和约束：
# JSON - 存储 dict/list 类型数据，MySQL 中对应 JSON 列；
# BigInteger - 大整数，用于自增主键（避免 int 溢出）；
# DateTime - 日期时间类型；ForeignKey - 外键约束；
# Integer - 普通整数；String - 变长字符串；Text - 长文本
from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
# MySQL 方言特有的 DATETIME 类型，支持通过 fsp 参数指定小数秒精度（0-6 位）
from sqlalchemy.dialects.mysql import DATETIME as MySQLDATETIME

# 微秒精度时间戳：避免同一秒内多条消息排序不稳定
# fsp=6 表示保留 6 位小数秒（微秒级），是 MySQL DATETIME 支持的最高精度
DateTime6 = MySQLDATETIME(fsp=6)

# Mapped：Python 类型注解，声明列字段的 Python 类型；mapped_column：列定义函数（2.x 新 API）
from sqlalchemy.orm import Mapped, mapped_column

# 导入 ORM 基类，所有模型均继承此类，以便 SQLAlchemy 将其注册到元数据
from app.db import Base


def _now() -> datetime:
    """返回当前 UTC 时间，用作时间戳字段的默认值工厂函数。

    使用函数而非 datetime.utcnow 直接作为默认值，是因为 SQLAlchemy 的
    default= 参数接受可调用对象时会在每次 INSERT 时重新调用，
    从而保证每条记录拥有正确的插入时间，而非模块加载时的固定时间。

    Returns:
        datetime: 当前 UTC 时间（无时区信息）
    """
    return datetime.utcnow()  # 返回 UTC 当前时间，不带时区信息（数据库统一存储 UTC）


# ============ 租户 / 用户 / 应用 / 密钥 ============

class Tenant(Base):
    """租户模型，对应 tenant 表。

    DevSupport AI 是多租户 SaaS 系统，每个接入方（公司/组织）为一个租户。
    租户是数据隔离的最高维度，所有业务数据（用户、应用、会话等）均属于某个租户。
    """
    __tablename__ = "tenant"                                                              # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 租户唯一 ID，业务层生成（如 UUID 或自定义编码），主键
    name: Mapped[str] = mapped_column(String(128))                                        # 租户名称，如公司/组织名称
    plan_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("plan.id"))       # 关联套餐 ID（外键），None 表示未绑定套餐
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 租户创建时间，自动填充当前 UTC 时间


class User(Base):
    """用户模型，对应 user 表。

    用户归属于某个租户，通过角色（role）区分权限级别。
    角色体系：
      - customer_dev：客户侧开发者，只读访问本租户数据
      - customer_admin：客户侧管理员，可管理本租户用户和配置
      - support：内部技术支持人员，可访问工作台和跨租户数据
      - admin：系统管理员，最高权限
    """
    __tablename__ = "user"                                                                 # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 用户唯一 ID，主键
    tenant_id: Mapped[str] = mapped_column(String(64), ForeignKey("tenant.id"), index=True)  # 所属租户 ID（外键，建立索引加速租户维度查询）
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)            # 登录用户名，全局唯一（唯一约束 + 索引加速登录查询）
    password_hash: Mapped[str] = mapped_column(String(255))                               # bcrypt 哈希后的密码（含盐值），长度 60 字节，留 255 余量
    role: Mapped[str] = mapped_column(String(32))  # customer_dev/customer_admin/support/admin  # 用户角色字符串，决定访问权限范围
    display_name: Mapped[str] = mapped_column(String(64))                                 # 展示名称（中文名/昵称），用于界面展示和审计日志
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 用户创建时间


class App(Base):
    """应用模型，对应 app 表。

    租户可创建多个"应用"（如不同产品线的 API 接入），
    每个应用可生成独立的 API Key，实现细粒度的调用追踪和权限控制。
    """
    __tablename__ = "app"                                                                  # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 应用唯一 ID，主键
    tenant_id: Mapped[str] = mapped_column(String(64), ForeignKey("tenant.id"), index=True)  # 所属租户 ID（外键，索引加速租户维度查询）
    name: Mapped[str] = mapped_column(String(128))                                        # 应用名称，如"商城后端"、"iOS 客户端"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 应用创建时间


class ApiKey(Base):
    """API 密钥模型，对应 api_key 表。

    每个应用可生成多个 API Key，用于调用方身份认证。
    出于安全考虑，数据库只存储脱敏后的密钥（key_masked），
    完整密钥仅在生成时一次性展示给用户，系统不保留明文。
    """
    __tablename__ = "api_key"                                                              # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 密钥记录唯一 ID，主键
    app_id: Mapped[str] = mapped_column(String(64), ForeignKey("app.id"), index=True)    # 所属应用 ID（外键，索引加速应用维度查询）
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 冗余存储所属租户 ID（避免多表 JOIN，加速租户维度过滤）
    key_masked: Mapped[str] = mapped_column(String(64))  # 仅存脱敏值，如 ak_****8a2f    # 脱敏后的密钥展示值，便于用户识别（不存明文，不可反推）
    status: Mapped[str] = mapped_column(String(16))  # ACTIVE / EXPIRED / DISABLED        # 密钥状态：ACTIVE-有效，EXPIRED-已过期，DISABLED-已禁用
    expire_at: Mapped[datetime | None] = mapped_column(DateTime)                          # 密钥过期时间，None 表示永不过期
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 密钥创建时间


# ============ 接口 / 调用日志 / 错误码 ============

class ApiEndpoint(Base):
    """API 端点定义模型，对应 api_endpoint 表。

    存储 API 开放平台提供的所有接口的元信息（路径、名称、所属产品），
    供知识库检索和错误诊断时引用，也用于调用日志的结构化关联。
    """
    __tablename__ = "api_endpoint"                                                         # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 端点唯一 ID，主键
    product: Mapped[str] = mapped_column(String(64))                                      # 所属产品线名称，如"人脸识别"、"语音合成"
    path: Mapped[str] = mapped_column(String(128), index=True)                            # API 请求路径，如 /v2/face/detect（索引加速路径查询）
    name: Mapped[str] = mapped_column(String(128))                                        # 接口人类可读名称，如"人脸检测"


class ApiCallLog(Base):
    """API 调用日志模型，对应 api_call_log 表。

    记录每次 API 调用的关键信息，是故障诊断、用量统计、错误分析的核心数据来源。
    AI Agent 在诊断用户问题时会查询此表，通过 request_id 定位具体失败请求。
    """
    __tablename__ = "api_call_log"                                                         # 对应数据库表名
    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)                 # 请求唯一 ID，主键（由 API 网关生成，用于全链路追踪）
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 发起调用的租户 ID（索引加速租户维度聚合查询）
    app_id: Mapped[str] = mapped_column(String(64), index=True)                           # 发起调用的应用 ID（索引加速应用维度查询）
    api_key_id: Mapped[str | None] = mapped_column(String(64))                            # 使用的 API Key ID，None 表示匿名调用
    endpoint: Mapped[str] = mapped_column(String(128), index=True)                        # 调用的接口路径（索引加速按接口聚合统计）
    http_status: Mapped[int] = mapped_column(Integer, index=True)                         # HTTP 响应状态码（如 200/400/429/500，索引加速按状态码过滤）
    error_code: Mapped[str | None] = mapped_column(String(64), index=True)                # 业务错误码（如 RATE_LIMIT_EXCEEDED），None 表示调用成功
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)                           # 端到端响应延迟（毫秒），用于性能分析
    client_ip: Mapped[str | None] = mapped_column(String(64))                             # 客户端 IP 地址，用于安全审计和地域分析
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True, default=_now)      # 调用发生时间（索引加速时间范围查询）


class ErrorCode(Base):
    """错误码知识库模型，对应 error_code 表。

    存储 API 平台所有错误码的详细说明，是 AI Agent 进行错误诊断的核心知识来源。
    Agent 根据调用日志中的 error_code 查询此表，获取原因分析和修复建议。
    """
    __tablename__ = "error_code"                                                           # 对应数据库表名
    code: Mapped[str] = mapped_column(String(64), primary_key=True)                       # 错误码字符串，主键（如 RATE_LIMIT_EXCEEDED）
    name: Mapped[str] = mapped_column(String(128))                                        # 错误码人类可读名称
    http_status: Mapped[int] = mapped_column(Integer)                                     # 对应的 HTTP 状态码（如 429、400、500）
    cause: Mapped[str] = mapped_column(Text)                                              # 错误原因详细说明（长文本）
    fix_steps: Mapped[str] = mapped_column(Text)                                          # 修复步骤和建议（长文本，直接回复用户）


# ============ 套餐 / 用量 / 账单 ============

class Plan(Base):
    """套餐模型，对应 plan 表。

    定义 SaaS 平台的订阅套餐，包含 QPS 限制、月度配额和计费规则。
    租户选择套餐后，系统按套餐规则进行限流和计费。
    """
    __tablename__ = "plan"                                                                 # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 套餐唯一 ID，主键
    name: Mapped[str] = mapped_column(String(64))                                         # 套餐名称，如"基础版"、"专业版"、"企业版"
    qps_limit: Mapped[int] = mapped_column(Integer)                                       # 每秒最大请求数（QPS 上限），超出触发限流
    monthly_quota: Mapped[int] = mapped_column(Integer)                                   # 每月包含的免费调用次数配额
    price_per_call: Mapped[float] = mapped_column(default=0.0)                            # 套餐内每次调用单价（元/次），通常为 0（包月）
    overage_price_per_call: Mapped[float] = mapped_column(default=0.0)                    # 超出月度配额后的阶梯单价（元/次）


class UsageRecord(Base):
    """月度用量记录模型，对应 usage_record 表。

    按租户和月份聚合记录 API 调用次数，用于统计超出月度配额的超量调用，
    作为账单生成的依据。
    """
    __tablename__ = "usage_record"                                                         # 对应数据库表名
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)      # 自增主键（BigInteger 防止高频写入溢出）
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 租户 ID（索引加速租户维度查询）
    month: Mapped[str] = mapped_column(String(7), index=True)  # YYYY-MM                  # 账期月份，格式 YYYY-MM（如 2024-01），索引加速月度查询
    call_count: Mapped[int] = mapped_column(Integer, default=0)                           # 当月累计调用总次数
    overage_count: Mapped[int] = mapped_column(Integer, default=0)                        # 当月超出月度配额的调用次数（用于计算超量费用）


class Invoice(Base):
    """账单模型，对应 invoice 表。

    每月月底为每个租户生成一份账单，记录费用构成明细和总金额。
    账单状态流转：PENDING（待确认）→ ISSUED（已出账）→ PAID（已支付）。
    """
    __tablename__ = "invoice"                                                              # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 账单唯一 ID，主键
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 所属租户 ID（索引加速租户账单查询）
    month: Mapped[str] = mapped_column(String(7), index=True)                             # 账期月份，格式 YYYY-MM（索引加速按月查询）
    items: Mapped[dict] = mapped_column(JSON)  # 费用构成                                  # JSON 格式费用构成明细，如 {"base_fee": 100, "overage_fee": 50}
    amount: Mapped[float] = mapped_column(default=0.0)                                    # 账单总金额（元），items 中各项之和
    status: Mapped[str] = mapped_column(String(16))  # ISSUED / PAID / PENDING             # 账单状态：PENDING-待确认，ISSUED-已出账，PAID-已支付


# ============ 知识库文档（切片向量在 Milvus） ============

class KnowledgeDocument(Base):
    """知识库文档元数据模型，对应 knowledge_document 表。

    存储知识库中每篇文档的元信息（标题、分类、版本、状态等）。
    文档内容被切片后向量化存入 Milvus（见 app/rag/store.py），
    MySQL 中只存元数据，通过 id 关联 Milvus 中的切片向量。
    """
    __tablename__ = "knowledge_document"                                                   # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 文档唯一 ID，主键（同时作为 Milvus 中切片的文档 ID 外键）
    title: Mapped[str] = mapped_column(String(128))                                       # 文档标题，展示给用户
    category: Mapped[str] = mapped_column(String(64))                                     # 文档分类，如"API 文档"、"FAQ"、"故障排查指南"
    version: Mapped[str] = mapped_column(String(32), default="v1")                        # 文档版本号，支持多版本管理（默认 v1）
    status: Mapped[str] = mapped_column(String(16), default="published")                  # 文档状态：published-已发布，draft-草稿，archived-已归档
    source_path: Mapped[str] = mapped_column(String(255))                                 # 文档源文件路径（如 Markdown/PDF 文件的存储路径）
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)                          # 文档被切分成的切片总数，与 Milvus 中切片记录数对应
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 文档最后更新时间（切片更新时同步刷新）


# ============ 会话 / 消息 ============

class Conversation(Base):
    """会话模型，对应 conversation 表。

    一次用户与 AI Agent 的完整交互过程为一个会话（Conversation），
    会话内包含多条消息（Message）。会话记录了意图识别状态、实体信息、
    是否被 AI 解决、是否转人工等关键状态。
    """
    __tablename__ = "conversation"                                                         # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 会话唯一 ID，主键
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 所属租户 ID（索引加速租户维度查询）
    user_id: Mapped[str] = mapped_column(String(64), index=True)                          # 发起会话的用户 ID（索引加速用户历史会话查询）
    channel: Mapped[str] = mapped_column(String(32), default="web")                       # 接入渠道：web-网页，app-移动端，api-API 直接调用等
    status: Mapped[str] = mapped_column(String(16), default="active")  # active/closed    # 会话状态：active-进行中，closed-已结束
    latest_intent: Mapped[str | None] = mapped_column(String(64))                         # 当前识别到的最新意图（如 "query_error"、"check_usage"）
    collected_entities: Mapped[dict] = mapped_column(JSON, default=dict)                  # 已收集的实体信息（JSON 格式，如 {"request_id": "xxx", "error_code": "yyy"}）
    resolved_by_ai: Mapped[bool] = mapped_column(default=False)                           # 是否被 AI 成功解决（True=AI 解答满足用户，无需人工介入）
    transferred_to_human: Mapped[bool] = mapped_column(default=False)                     # 是否已转接人工客服（True=已转人工处理）
    satisfaction: Mapped[str | None] = mapped_column(String(16))  # resolved/unresolved/null  # 用户满意度反馈：resolved-已解决，unresolved-未解决，None-未反馈
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 会话创建时间
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)   # 会话最后更新时间（每次新消息或状态变更时自动刷新）


class Message(Base):
    """消息模型，对应 message 表。

    会话中的每一条消息（用户发送或 AI 回复）为一条记录。
    通过 role 字段区分消息来源（user/assistant/system），
    meta 字段以 JSON 格式存储诊断卡片、引用来源、trace_id 等附加信息。
    """
    __tablename__ = "message"                                                              # 对应数据库表名
    id: Mapped[str] = mapped_column(String(64), primary_key=True)                         # 消息唯一 ID，主键
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversation.id"), index=True)  # 所属会话 ID（外键 + 索引，加速会话消息列表查询）
    role: Mapped[str] = mapped_column(String(16))  # user / assistant / system             # 消息角色：user-用户，assistant-AI 助手，system-系统消息
    content: Mapped[str] = mapped_column(Text)                                            # 消息正文内容（长文本，支持 Markdown 格式）
    meta: Mapped[dict] = mapped_column(JSON, default=dict)  # 诊断卡片/引用/trace_id 等    # 附加元数据：存储 AI 回复的诊断卡片、知识库引用、关联 trace_id 等结构化信息
    created_at: Mapped[datetime] = mapped_column(DateTime6, default=_now)                  # 消息创建时间（微秒精度，保证同一秒内多条消息的排序稳定）


# ============ 可观测：Agent 链路 / 工具调用 ============

class AgentTrace(Base):
    """Agent 执行链路追踪模型，对应 agent_trace 表。

    记录每次 AI Agent 处理请求时的执行步骤，实现 Agent 推理过程的可观测性。
    每个处理步骤（意图识别、RAG 检索、工具调用、回复生成等）各记录一条 AgentTrace，
    通过 trace_id 将同一次请求的所有步骤串联起来，便于问题排查和性能分析。
    """
    __tablename__ = "agent_trace"                                                          # 对应数据库表名
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)      # 自增主键（BigInteger 应对高并发写入）
    trace_id: Mapped[str] = mapped_column(String(64), index=True)                         # 链路追踪 ID（同一次请求的所有步骤共享，索引加速按链路查询）
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True)           # 关联的会话 ID（索引加速按会话查询，None 表示非会话触发的 Agent 调用）
    message_id: Mapped[str | None] = mapped_column(String(64))                            # 触发此 Agent 执行的消息 ID
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 所属租户 ID（索引加速租户维度查询）
    agent_name: Mapped[str] = mapped_column(String(64))                                   # 执行步骤的 Agent 或处理器名称（如 "intent_agent"、"rag_agent"）
    step_order: Mapped[int] = mapped_column(Integer)                                      # 步骤执行顺序编号（从 1 开始递增，用于还原执行顺序）
    input_summary: Mapped[str] = mapped_column(Text, default="")                          # 步骤输入内容摘要（避免存储完整内容导致字段过大）
    output_summary: Mapped[str] = mapped_column(Text, default="")                         # 步骤输出内容摘要
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok/error/skip        # 步骤执行状态：ok-成功，error-失败，skip-被跳过
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)                          # 步骤执行耗时（毫秒），用于性能分析
    token_usage: Mapped[int] = mapped_column(Integer, default=0)                          # 此步骤消耗的 LLM Token 数量
    hit_docs: Mapped[list] = mapped_column(JSON, default=list)                            # RAG 步骤命中的知识库文档列表（JSON 数组，含 doc_id 和相关性分数）
    error_message: Mapped[str | None] = mapped_column(Text)                               # 步骤失败时的错误信息（None 表示成功）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 步骤执行时间


class ToolCallLog(Base):
    """工具调用日志模型，对应 tool_call_log 表。

    详细记录 Agent 每次调用外部工具（如查询 API 日志、查询错误码、
    调用诊断脚本等）的参数摘要、结果摘要、状态和耗时。
    与 AgentTrace 配合使用（通过 trace_id 关联），提供更细粒度的可观测性。
    """
    __tablename__ = "tool_call_log"                                                        # 对应数据库表名
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)      # 自增主键
    trace_id: Mapped[str] = mapped_column(String(64), index=True)                         # 关联的链路追踪 ID（索引加速按链路查询所有工具调用）
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 所属租户 ID（索引加速租户维度统计）
    tool_name: Mapped[str] = mapped_column(String(64))                                    # 被调用的工具名称（如 "query_api_log"、"lookup_error_code"）
    args_summary: Mapped[str] = mapped_column(Text, default="")                           # 工具调用参数摘要（避免存储完整参数导致存储膨胀）
    result_summary: Mapped[str] = mapped_column(Text, default="")                         # 工具返回结果摘要
    status: Mapped[str] = mapped_column(String(16), default="ok")                         # 调用状态：ok-成功，error-失败
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)                          # 工具调用耗时（毫秒），超时阈值参考 config.tool_timeout_seconds
    error_message: Mapped[str | None] = mapped_column(Text)                               # 调用失败时的错误信息（None 表示成功）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 工具调用时间


# ============ 工单 / 反馈 / 审计 ============

class Ticket(Base):
    """工单模型，对应 ticket 表。

    当 AI Agent 无法解决用户问题时（或用户主动要求），
    系统自动创建工单并转交给人工支持团队处理。
    工单包含 AI 的初步诊断结果，帮助支持人员快速理解问题背景。
    """
    __tablename__ = "ticket"                                                               # 对应数据库表名
    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)                  # 工单唯一 ID，主键（业务层生成，如 TKT-20240101-0001）
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 所属租户 ID（索引加速租户工单列表查询）
    user_id: Mapped[str] = mapped_column(String(64), index=True)                          # 提交工单的用户 ID（索引加速用户工单历史查询）
    category: Mapped[str] = mapped_column(String(32))                                     # 工单分类（如"API 报错"、"性能问题"、"账单问题"）
    priority: Mapped[str] = mapped_column(String(8))  # P0/P1/P2/P3                       # 优先级：P0-紧急，P1-高，P2-中，P3-低
    status: Mapped[str] = mapped_column(String(24), default="new", index=True)            # 工单状态（索引加速按状态过滤）：new/assigned/in_progress/resolved/closed
    title: Mapped[str] = mapped_column(String(255))                                       # 工单标题（简短描述问题）
    summary: Mapped[str] = mapped_column(Text, default="")                                # 问题详细描述
    related_request_ids: Mapped[list] = mapped_column(JSON, default=list)                 # 相关的 API 请求 ID 列表（用于关联 api_call_log 记录）
    related_endpoint: Mapped[str | None] = mapped_column(String(128))                     # 问题涉及的 API 端点路径
    error_code: Mapped[str | None] = mapped_column(String(64))                            # 触发工单的业务错误码
    evidence: Mapped[str] = mapped_column(Text, default="")                               # 用户提供的证据（如截图描述、日志片段）
    ai_diagnosis: Mapped[str] = mapped_column(Text, default="")                           # AI 自动生成的初步诊断结果（供支持人员参考）
    assignee: Mapped[str | None] = mapped_column(String(64))                              # 负责处理的支持人员用户名（None 表示未分配）
    conversation_id: Mapped[str | None] = mapped_column(String(64))                       # 关联的会话 ID（便于支持人员查看完整对话上下文）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 工单创建时间
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)   # 工单最后更新时间（状态变更、分配、备注时自动刷新）


class Feedback(Base):
    """用户反馈模型，对应 feedback 表。

    记录用户对 AI 回复效果的评价，用于：
      1. 统计 AI 解决率指标；
      2. 收集需人工介入的场景；
      3. 为模型优化提供训练信号。
    """
    __tablename__ = "feedback"                                                             # 对应数据库表名
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)      # 自增主键
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)                  # 反馈对应的会话 ID（索引加速会话维度查询）
    message_id: Mapped[str | None] = mapped_column(String(64))                            # 反馈针对的具体消息 ID（None 表示对整个会话评价）
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 所属租户 ID（索引加速租户维度统计）
    type: Mapped[str] = mapped_column(String(16))  # resolved/unresolved/need_human        # 反馈类型：resolved-已解决，unresolved-未解决，need_human-需要人工
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 反馈提交时间


class AuditLog(Base):
    """操作审计日志模型，对应 audit_log 表。

    记录系统中所有重要操作（如用户创建、权限变更、配置修改、数据删除等），
    满足合规要求，并为安全事件调查提供完整的操作历史。
    """
    __tablename__ = "audit_log"                                                            # 对应数据库表名
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)      # 自增主键（BigInteger 应对持续写入）
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True)                 # 操作涉及的租户 ID（None 表示系统级操作，索引加速租户审计查询）
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)                   # 执行操作的用户 ID（None 表示系统自动触发，索引加速用户操作记录查询）
    action: Mapped[str] = mapped_column(String(64))                                       # 操作类型（如 "create_user"、"disable_api_key"、"update_plan"）
    detail: Mapped[str] = mapped_column(Text, default="")                                 # 操作详情（JSON 或文本格式，记录操作前后的关键字段变化）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 操作发生时间


class TokenUsage(Base):
    """Token / 成本统计（M11）。

    按会话和租户维度记录每次 LLM API 调用消耗的 Token 数量，
    用于：
      1. 计算 AI 服务的实际成本，对应 Invoice 中的 LLM 费用分项；
      2. 分析各租户/会话的 Token 消耗趋势；
      3. 检测异常高消耗，触发告警或限流。
    """

    __tablename__ = "token_usage"                                                          # 对应数据库表名
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)      # 自增主键
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)                        # 所属租户 ID（索引加速租户维度 Token 统计）
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True)           # 关联会话 ID（索引加速会话维度查询，None 表示非会话场景的 LLM 调用）
    model: Mapped[str] = mapped_column(String(64))                                        # 使用的 LLM 模型名称（如 "qwen-turbo"、"qwen-plus"）
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)                        # 输入 Token 数（prompt + 上下文历史），通常大于输出 Token
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)                    # 输出 Token 数（模型生成的回复），是成本的主要组成部分
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)                         # 总 Token 数（prompt_tokens + completion_tokens），用于快速统计
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)                  # 本次 LLM 调用发生时间
