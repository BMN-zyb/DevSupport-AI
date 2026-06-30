# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""应用配置：从环境变量 / .env 读取，集中管理。

本模块使用 pydantic-settings 统一管理所有运行时配置，支持从 .env 文件或
系统环境变量中自动读取并做类型校验。通过 get_settings() + lru_cache 实现
单例模式，整个应用共享同一份配置对象，避免重复解析环境变量。

职责：
  - LLM / Embedding / Rerank 模型接入参数
  - MySQL / Milvus / Redis 连接信息
  - JWT 认证相关参数
  - 意图置信度、RAG 阈值等业务调节参数
  - 动态生成异步/同步 MySQL DSN 字符串
"""

# 标准库：lru_cache 用于将 get_settings() 的返回值缓存，保证全局单例
from functools import lru_cache

# pydantic-settings：BaseSettings 支持从环境变量/.env自动加载并类型校验
# SettingsConfigDict 用于声明 .env 文件路径、编码等元信息
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局应用配置类。

    所有字段均可通过同名环境变量或 .env 文件中的键值对覆盖。
    字段提供默认值仅作为本地开发/测试环境的开箱即用配置，
    生产环境必须通过环境变量注入真实密钥和地址。
    """

    # pydantic-settings 元配置：指定 .env 文件路径、编码，并忽略未声明的额外字段
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ----- LLM / Embedding / Rerank (DashScope, OpenAI 兼容) -----
    dashscope_api_key: str = ""                  # 阿里云 DashScope API 密钥，生产必填
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # LLM API 基础地址，兼容 OpenAI 格式
    llm_model_small: str = "qwen-turbo"          # 轻量模型：用于意图识别、实体提取等低延迟任务
    llm_model_large: str = "qwen-plus"           # 大模型：用于最终回复生成、摘要等需要高质量输出的任务
    embedding_model: str = "text-embedding-v3"   # 文本向量化模型名称，与 Milvus 集合维度须保持一致
    embedding_dim: int = 1024                    # 向量维度，须与 Milvus 集合建立时的维度完全匹配
    rerank_model: str = "gte-rerank-v2"          # 重排序模型：对 RAG 召回结果二次精排，提升相关性

    # ----- MySQL -----
    mysql_host: str = "127.0.0.1"               # MySQL 服务器主机地址
    mysql_port: int = 3307                       # MySQL 端口（默认非标准端口，避免与本机 MySQL 冲突）
    mysql_user: str = "devsupport"               # 数据库登录用户名
    mysql_password: str = "devsupport123"        # 数据库登录密码，生产环境应通过环境变量注入
    mysql_db: str = "devsupport"                 # 使用的数据库名称

    # ----- Milvus -----
    milvus_uri: str = "http://localhost:19531"   # Milvus 向量数据库服务地址（端口同样使用非标准值）
    milvus_collection: str = "knowledge_chunk"  # 存储知识库切片向量的集合名称

    # ----- Redis -----
    redis_url: str = "redis://localhost:6380/0"  # Redis 连接 URL（db=0），用于语义缓存、限流计数器等

    # ----- 认证 -----
    jwt_secret: str = "change-me-in-production-please"  # JWT 签名密钥，生产环境必须替换为高熵随机字符串
    jwt_algorithm: str = "HS256"                # JWT 签名算法，HMAC-SHA256
    jwt_expire_minutes: int = 720               # JWT 有效期（分钟），默认 12 小时

    # ----- 业务阈值 -----
    intent_confidence_threshold: float = 0.6   # 意图置信度低于此值触发澄清追问
    rag_score_threshold: float = 0.3           # rerank 分达标即判文档命中
    rag_vec_hit_threshold: float = 0.45        # 向量余弦达标即判文档命中（与上者取其一）
    tool_timeout_seconds: float = 3.0          # 单次工具调用超时
    semantic_cache_sim_threshold: float = 0.95  # 语义缓存命中所需的最低相似度

    # ----- 其它 -----
    app_env: str = "dev"                        # 运行环境标识：dev / staging / prod
    log_level: str = "INFO"                     # 日志级别，影响 logging.basicConfig 初始化

    @property
    def mysql_dsn_async(self) -> str:
        """生成异步 MySQL DSN（aiomysql 驱动）。

        FastAPI 路由及 Agent 运行时使用异步会话，需要 aiomysql 驱动。
        charset=utf8mb4 确保支持 emoji 及全 Unicode 字符集。

        Returns:
            str: 格式为 mysql+aiomysql://user:pass@host:port/db?charset=utf8mb4 的 DSN 字符串
        """
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    @property
    def mysql_dsn_sync(self) -> str:
        """生成同步 MySQL DSN（pymysql 驱动）。

        建表脚本、数据初始化脚本等非异步场景使用同步会话。

        Returns:
            str: 格式为 mysql+pymysql://user:pass@host:port/db?charset=utf8mb4 的 DSN 字符串
        """
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例（带缓存）。

    使用 lru_cache 装饰器确保 Settings() 只被实例化一次，
    后续调用直接返回缓存对象，避免重复读取和解析 .env 文件。

    Returns:
        Settings: 全局唯一的配置对象实例
    """
    return Settings()  # 实例化时自动从环境变量/.env 加载所有字段


# 模块级别的快捷访问变量，供其它模块直接 `from app.config import settings` 使用
settings = get_settings()
