# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""LLM 接入层：通义 DashScope（OpenAI 兼容）。

提供 chat（含 function calling）、流式 chat、embedding、rerank。
所有调用真实请求 DashScope；token 用量随结果返回，供成本统计。

本模块是整个系统与大语言模型交互的唯一入口，对外暴露以下功能：
  - chat：单轮/多轮对话，支持 function calling，瞬时错误自动重试。
  - chat_stream：流式对话，逐 token 产出文本，降低首字节延迟。
  - embed / embed_one：文本向量化，用于语义缓存和向量检索。
  - rerank：文档重排序，提升 RAG 检索结果的相关性排序质量。

重试策略：仅对瞬时错误（超时/连接/限流/5xx）重试，最多 3 次，
指数退避 0.5s 起步、最大 4s，避免对参数错误等做无意义重试。
"""

# 标准库：asyncio 用于将同步 DashScope rerank SDK 放入线程池执行，避免阻塞事件循环
import asyncio
# 标准库：AsyncGenerator 用于类型注解流式生成器的返回类型
from collections.abc import AsyncGenerator
# 标准库：dataclass/field 用于定义结构化的对话结果数据类
from dataclasses import dataclass, field
# 标准库：lru_cache 用于 _client() 单例缓存，避免重复初始化 AsyncOpenAI 客户端
from functools import lru_cache

# 第三方库：DashScope 官方 SDK，用于调用 rerank 接口（同步 SDK）
import dashscope
# 第三方库：openai 用于异常类型引用，供重试策略过滤瞬时错误
import openai
# 第三方库：AsyncOpenAI 是 OpenAI 官方 Python SDK 的异步客户端，DashScope 兼容其接口
from openai import AsyncOpenAI
# 第三方库：tenacity 提供灵活的重试装饰器，配置重试次数、等待策略和触发条件
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# 项目内部：读取 API Key、模型名、Base URL、embedding 维度等运行时配置
from app.config import settings

# 仅对瞬时错误重试（超时/连接/限流/5xx），避免对参数错误等无意义重试
_TRANSIENT = (
    openai.APITimeoutError,       # 请求超时，可能因网络抖动导致，重试通常能成功
    openai.APIConnectionError,    # 连接失败，可能因网络问题导致，可重试
    openai.RateLimitError,        # 限流（429），等待后重试
    openai.InternalServerError,   # 服务端 5xx 错误，短暂故障，可重试
)
# 构造 tenacity 重试装饰器：最多重试 3 次，指数退避（0.5s 起步，最大 4s），命中后重新抛出原始异常
_retry = retry(
    stop=stop_after_attempt(3),                    # 最多尝试 3 次（含初次调用）
    wait=wait_exponential(multiplier=0.5, max=4),  # 指数退避：0.5s、1s、2s…最大 4s
    retry=retry_if_exception_type(_TRANSIENT),     # 仅对上述瞬时错误触发重试
    reraise=True,                                  # 超出重试次数后重新抛出异常，而非返回 None
)


@dataclass
class ChatResult:
    """封装单次对话 API 调用的完整返回结果。

    content 为模型文本回复；tool_calls 为 function calling 的调用列表；
    prompt/completion/total_tokens 用于成本统计与监控；model 记录实际使用的模型。
    """
    content: str                                    # 模型生成的文本内容（function calling 时可能为空字符串）
    tool_calls: list = field(default_factory=list)  # OpenAI 风格 tool_calls，每项含 id/name/arguments
    prompt_tokens: int = 0                          # 输入（提示词）消耗的 token 数
    completion_tokens: int = 0                      # 输出（补全）消耗的 token 数
    total_tokens: int = 0                           # 总 token 数 = prompt_tokens + completion_tokens
    model: str = ""                                 # 实际使用的模型名称，用于多模型环境下的追踪


@lru_cache  # lru_cache 确保同一进程内只初始化一个 AsyncOpenAI 客户端实例（单例）
def _client() -> AsyncOpenAI:
    """创建并返回全局共享的 DashScope 异步客户端（单例）。

    若 DASHSCOPE_API_KEY 未配置，则立即抛出 RuntimeError，
    使问题在启动阶段即暴露，而非等到首次调用 LLM 时才失败。

    返回:
        AsyncOpenAI: 指向 DashScope OpenAI 兼容接口的异步客户端。

    抛出:
        RuntimeError: 若环境变量 DASHSCOPE_API_KEY 未配置。
    """
    if not settings.dashscope_api_key:  # 检查 API Key 是否已配置，未配置则快速失败
        raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法调用 LLM")
    # 使用 DashScope 的 OpenAI 兼容 base_url，使标准 OpenAI SDK 透明对接 DashScope
    return AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.llm_base_url)


@_retry  # 应用重试装饰器，瞬时错误时自动重试最多 3 次
async def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.2,
) -> ChatResult:
    """单轮/多轮对话。tools 非空时启用 function calling。瞬时错误自动重试。

    参数:
        messages (list[dict]): OpenAI 格式的消息列表，包含 role 和 content。
        model (str | None): 使用的模型名；None 时使用配置中的大模型（llm_model_large）。
        tools (list[dict] | None): OpenAI 格式的工具定义列表；None 表示不启用 function calling。
        temperature (float): 生成温度，越低输出越确定，默认 0.2 适合问答任务。

    返回:
        ChatResult: 包含回复内容、tool_calls 和 token 用量的结构化结果。
    """
    model = model or settings.llm_model_large  # 未指定模型时默认使用大模型
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}  # 构造基础请求参数
    if tools:  # 若提供了工具定义，则在请求中启用 function calling
        kwargs["tools"] = tools               # 传入工具列表定义
        kwargs["tool_choice"] = "auto"        # "auto" 让模型自行决定是否调用工具
    resp = await _client().chat.completions.create(**kwargs)  # 发起异步 LLM 推理请求
    choice = resp.choices[0]   # 取第一个候选输出（通常只有一个）
    usage = resp.usage         # 获取 token 用量统计对象（可能为 None）
    tool_calls = []  # 初始化工具调用列表
    if choice.message.tool_calls:  # 若模型决定调用工具，则解析工具调用信息
        for tc in choice.message.tool_calls:  # 遍历每个工具调用
            tool_calls.append(
                {
                    "id": tc.id,                       # 工具调用的唯一 ID，用于后续结果关联
                    "name": tc.function.name,          # 被调用的函数名称
                    "arguments": tc.function.arguments,  # 函数参数（JSON 字符串格式）
                }
            )
    return ChatResult(
        content=choice.message.content or "",            # 文本内容，function calling 时可能为 None，转为空字符串
        tool_calls=tool_calls,                            # 工具调用列表（无调用时为空列表）
        prompt_tokens=usage.prompt_tokens if usage else 0,       # 输入 token 数（无 usage 时为 0）
        completion_tokens=usage.completion_tokens if usage else 0,  # 输出 token 数
        total_tokens=usage.total_tokens if usage else 0,            # 总 token 数
        model=model,                                                 # 记录实际使用的模型名
    )


async def chat_stream(
    messages: list[dict], *, model: str | None = None, temperature: float = 0.3
) -> AsyncGenerator[str, None]:
    """流式对话，逐段产出文本。

    使用 stream=True 开启服务端流式传输（Server-Sent Events），
    每收到一个文本 chunk 立即 yield 给调用方，实现打字机效果，降低感知延迟。
    注意：流式模式下无法获取 token 用量统计。

    参数:
        messages (list[dict]): OpenAI 格式的消息列表。
        model (str | None): 使用的模型名；None 时使用大模型。
        temperature (float): 生成温度，默认 0.3 略高于 chat，流式场景通常用于自由对话。

    生成:
        str: 每次 yield 一个文本片段（delta.content），直到流结束。
    """
    model = model or settings.llm_model_large  # 未指定模型时默认使用大模型
    # stream=True 告知服务端以 SSE 方式推送 chunk，而非等待完整响应
    stream = await _client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=True
    )
    async for chunk in stream:  # 异步迭代每个流式 chunk
        if chunk.choices and chunk.choices[0].delta.content:  # 跳过空内容的 chunk（如 finish_reason chunk）
            yield chunk.choices[0].delta.content  # 逐段产出文本片段


@_retry  # 应用重试装饰器，网络抖动或限流时自动重试
async def embed(texts: list[str]) -> list[list[float]]:
    """文本向量化（text-embedding-v3）。瞬时错误自动重试。

    将一批文本转换为固定维度的浮点向量，用于语义搜索和缓存相似度计算。
    批量处理多个文本可减少 API 调用次数，降低延迟和成本。

    参数:
        texts (list[str]): 需要向量化的文本列表，可包含一条或多条。

    返回:
        list[list[float]]: 与输入列表等长的向量列表，每个向量维度由 settings.embedding_dim 决定。
    """
    resp = await _client().embeddings.create(
        model=settings.embedding_model,   # 使用配置中指定的 embedding 模型（如 text-embedding-v3）
        input=texts,                      # 待向量化的文本列表
        dimensions=settings.embedding_dim  # 指定输出向量维度（如 1536），必须与 Qdrant 集合维度一致
    )
    return [d.embedding for d in resp.data]  # 按顺序提取每条文本的 embedding 向量


async def embed_one(text: str) -> list[float]:
    """对单条文本进行向量化的便捷封装。

    内部调用 embed([text]) 并取第一个结果，避免调用方每次手动处理列表。

    参数:
        text (str): 需要向量化的单条文本。

    返回:
        list[float]: 该文本的 embedding 向量。
    """
    return (await embed([text]))[0]  # 复用批量接口，取列表第一个（也是唯一一个）向量


async def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """文档重排序（gte-rerank）。返回 [{index, score}]，按相关性降序。

    DashScope rerank 为同步 SDK，放入线程池避免阻塞事件循环。

    对 RAG 检索出的候选文档按与 query 的相关性重新排序，
    将最相关的文档排在最前，提升最终生成答案的质量。

    参数:
        query (str): 用户查询文本，作为相关性判断的基准。
        documents (list[str]): 待重排序的文档文本列表。
        top_n (int | None): 返回前 N 个最相关文档；None 时返回全部。

    返回:
        list[dict]: 按相关性降序排列的结果列表，每项为 {"index": 原始索引, "score": 相关性得分}。
                    若 documents 为空，则直接返回空列表。
    """
    if not documents:  # 空文档列表时无需调用 API，直接返回，避免无效请求
        return []
    dashscope.api_key = settings.dashscope_api_key  # 配置 DashScope SDK 的全局 API Key
    top_n = top_n or len(documents)  # 未指定 top_n 时默认返回全部文档的排序结果

    def _call():
        """同步调用 DashScope rerank API 的内部函数，将在线程池中执行。"""
        return dashscope.TextReRank.call(
            model=settings.rerank_model,    # 使用配置中指定的 rerank 模型（如 gte-rerank）
            query=query,                    # 用户查询文本
            documents=documents,            # 待排序的文档列表
            top_n=top_n,                    # 返回前 N 个结果
            return_documents=False,         # 只返回索引和分数，不返回原始文档内容，减少传输量
        )

    # asyncio.to_thread 将同步函数放入默认线程池执行，避免阻塞异步事件循环
    resp = await asyncio.to_thread(_call)
    results = []  # 初始化结果列表
    if resp and resp.output and resp.output.results:  # 防御性检查，确保响应结构完整
        for r in resp.output.results:  # 遍历每个重排序结果
            results.append({"index": r.index, "score": float(r.relevance_score)})  # 提取原始索引和相关性得分
    return results  # 返回按相关性降序排列的 [{index, score}] 列表
