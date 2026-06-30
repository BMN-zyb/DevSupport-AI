# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""FastAPI 应用入口。

本文件是整个后端服务的启动入口，负责：
  1. 初始化 FastAPI 应用实例，配置元信息（标题、描述、版本）；
  2. 注册 CORS 中间件，允许前端开发服务器（Vite 默认端口 5173）跨域访问；
  3. 注册健康检查路由（/health 和 /api/health），供容器编排（K8s/Docker）探活；
  4. 挂载所有业务路由模块（认证、聊天、会话、文档、工单、工作台、追踪、评测）。

使用方式：
    uvicorn app.main:app --reload
"""

# 标准库：logging 用于初始化全局日志配置
import logging

# FastAPI 核心框架
from fastapi import FastAPI
# CORS 中间件：处理跨域请求，允许前端开发服务器的浏览器请求通过
from fastapi.middleware.cors import CORSMiddleware

# 业务路由模块：每个模块对应一组相关 API 端点
from app.api import auth, chat, conversations, docs  # 认证、聊天、会话管理、文档查询
from app.api import eval as eval_api                  # 评测模块（避免与 Python 内置 eval 函数命名冲突）
from app.api import tickets, traces, workbench        # 工单管理、Agent 链路追踪、内部工作台
# 全局配置：读取日志级别、运行环境等参数
from app.config import settings

# 初始化全局日志：按配置中的 log_level 设置日志级别（INFO/DEBUG/WARNING 等）
logging.basicConfig(level=settings.log_level)
# 获取本模块专属 logger，便于在日志中区分来源模块
logger = logging.getLogger("devsupport")

# 创建 FastAPI 应用实例，配置 OpenAPI 文档展示的元信息
app = FastAPI(
    title="DevSupport AI",                                              # API 文档标题
    description="面向 API 开放平台的多 Agent 智能技术支持系统",              # API 文档描述
    version="0.1.0",                                                    # 应用版本号，同步于 __init__.py
)

# 注册 CORS 中间件：允许前端开发服务器（localhost:5173 / 127.0.0.1:5173，即 Vite 默认端口）发起跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # 仅允许本地前端开发服务器跨域，生产应配置真实域名
    allow_credentials=True,   # 允许携带 Cookie / Authorization 凭证头
    allow_methods=["*"],      # 允许所有 HTTP 方法（GET/POST/PUT/DELETE 等）
    allow_headers=["*"],      # 允许所有请求头（包括自定义头）
)


@app.get("/health", tags=["system"])      # 注册 /health 路由，供外部健康检查工具直接访问
@app.get("/api/health", tags=["system"])  # 同时注册 /api/health，供前端统一走 /api 前缀访问
async def health() -> dict:
    """健康检查。

    返回服务当前运行状态、所处环境和版本号。
    可被 Kubernetes liveness/readiness probe、Docker HEALTHCHECK 或监控系统调用。

    Returns:
        dict: 包含 status（固定为 "ok"）、env（运行环境）、version（版本号）的响应字典
    """
    return {"status": "ok", "env": settings.app_env, "version": app.version}  # 固定返回 ok，env 和 version 来自配置和应用元信息


# 挂载各业务路由模块，每个 router 内部定义了各自的路径前缀和端点
app.include_router(auth.router)           # 认证路由：登录、刷新 Token 等
app.include_router(chat.router)           # 聊天路由：用户与 AI Agent 交互的核心接口
app.include_router(conversations.router)  # 会话管理路由：查询、关闭会话，查看历史消息
app.include_router(docs.router)           # 知识库文档路由：文档查询与管理
app.include_router(tickets.router)        # 工单路由：创建、查询、更新工单
app.include_router(workbench.router)      # 工作台路由：内部支持人员使用的管理视图
app.include_router(traces.router)         # 链路追踪路由：查看 Agent 执行步骤和工具调用记录
app.include_router(eval_api.router)       # 评测路由：对 AI 回复质量进行人工或自动评测
