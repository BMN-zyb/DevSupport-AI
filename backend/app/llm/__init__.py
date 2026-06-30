# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""llm 包初始化文件。

本包是 DevSupport-AI 系统与大语言模型交互的核心层，包含以下两个子模块：
  - client：LLM 接入层，封装了对通义 DashScope 的 chat、流式 chat、
    embedding 和 rerank 等接口调用，统一管理重试策略和客户端单例。
  - router：模型分层路由，根据任务类型自动选择小模型（降本降延迟）
    或大模型（保质量），实现成本与质量的平衡。

上层业务代码通过 `from app.llm import client` 或 `from app.llm.router import model_for`
直接导入所需功能，本 __init__.py 仅作包标识，不做额外导出。
"""
