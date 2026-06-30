# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""cache 包初始化文件。

本包提供 DevSupport-AI 系统的缓存层功能，包含以下三个子模块：
  - redis_client：异步 Redis 客户端的单例管理，供其他缓存模块共用。
  - route_cache：路由缓存，缓存意图识别结果以跳过 LLM 推理，降低延迟。
  - semantic_cache：语义缓存，基于向量相似度匹配热点问题的已有答案。

各模块按需在业务层直接导入使用，本 __init__.py 仅作包标识，不做额外导出。
"""
