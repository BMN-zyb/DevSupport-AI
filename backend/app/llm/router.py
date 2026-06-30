# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""模型分层路由：简单任务走小模型降本降延迟，复杂任务走大模型保质量。

本模块实现了一个轻量级的模型选择策略，根据任务类型将请求分配到不同档位的模型：
  - 小模型（llm_model_small）：用于意图识别、闲聊、路由判断等对质量要求较低的任务，
    延迟低、成本少，适合高频调用场景。
  - 大模型（llm_model_large）：用于故障诊断、文档摘要、RAG 生成等需要深度推理的任务，
    质量高，适合对准确性要求严格的场景。

未知任务类型默认使用大模型，遵循"宁可多花成本，不允许质量降级"的保守策略。
"""

# 项目内部：读取小模型和大模型的实际模型名称配置（如 qwen-turbo、qwen-max）
from app.config import settings

# 任务 -> 模型档位
# 小模型任务集合：这些任务逻辑简单、模板化程度高，小模型即可胜任
_SMALL_TASKS = {"intent", "clarify", "chitchat", "route", "cache_match"}
# 大模型任务集合：这些任务需要深度理解和推理，必须使用大模型保证质量
_LARGE_TASKS = {"diagnose", "summarize", "rag_generate", "billing_explain"}


def model_for(task: str) -> str:
    """根据任务类型返回对应的模型名称。未知任务默认大模型（偏保守）。

    通过集合查找 O(1) 实现快速分档，避免引入复杂的规则引擎。
    对于 _LARGE_TASKS 中的任务或任何未知任务类型，均返回大模型，
    确保系统在面对新任务类型时不会因使用小模型而出现质量问题。

    参数:
        task (str): 任务类型标识符，如 "intent"、"diagnose"、"rag_generate" 等。

    返回:
        str: 对应档位的模型名称字符串，直接来自 settings 配置，可传入 client.chat() 使用。
    """
    if task in _SMALL_TASKS:          # 若任务属于小模型任务集合，则返回小模型名称
        return settings.llm_model_small  # 小模型：低延迟、低成本，适合简单任务
    return settings.llm_model_large   # 其他任务（含未知任务）均使用大模型，保证质量
