# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Agent 通用工具：JSON 解析 + 结构化卡片渲染。

本文件提供 DevSupport-AI 各专业 Agent 共用的工具函数，包含两类能力：
1. JSON 稳健解析（parse_json）：从 LLM 输出中安全提取 JSON，
   容忍 ```json 代码块围栏、前后多余文字等常见 LLM 输出格式噪声。
2. 结构化卡片处理（normalize_card / render_card）：将 LLM 生成的
   结构化诊断卡片（含结论、证据、建议步骤）规范化并渲染为 Markdown，
   供 summarize 节点展示、记忆存储和语义缓存使用。
"""

import json  # 用于解析 LLM 输出的 JSON 字符串
import re    # 用于正则清洗 LLM 输出的 ```json 代码块围栏及提取 JSON 对象

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)  # 贪婪匹配 JSON 对象（含跨行嵌套），用于从混合文本中提取 JSON


def parse_json(text: str) -> dict:
    """从 LLM 输出中稳健解析 JSON（容忍 ```json 围栏与多余文本）。

    LLM 输出可能包含 ```json ... ``` 代码块、前导说明文字或尾部注释，
    此函数按以下步骤处理：
    1. 去除首尾空白。
    2. 若有代码块围栏（```）则清除。
    3. 用正则找到 JSON 对象范围后尝试解析。
    任何步骤失败均安全返回空字典，调用方无需 try/except。

    Args:
        text: LLM 原始输出文本，可能含有多余格式内容。
    Returns:
        解析成功时返回 dict；输入为空或解析失败时返回 {}。
    """
    if not text:
        return {}  # 空输入直接返回空字典，避免后续正则操作空字符串
    text = text.strip()  # 去除首尾空白（换行、空格等）
    if text.startswith("```"):
        # 清除 Markdown 代码块围栏，兼容 ``` 和 ```json 两种形式
        # flags=re.MULTILINE 让 ^ 和 $ 匹配每一行的开头和结尾
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = _JSON_RE.search(text)  # 在清洗后的文本中查找 JSON 对象（{ ... }）
    if not m:
        return {}  # 未找到 JSON 结构（如 LLM 返回纯文本），安全返回空字典
    try:
        return json.loads(m.group(0))  # 解析找到的 JSON 字符串为 Python dict
    except json.JSONDecodeError:
        return {}  # JSON 格式错误（截断、转义问题等），安全返回空字典


def normalize_card(raw: dict) -> dict:
    """规范化结构化卡片字段。

    将 LLM 生成的原始卡片字典规范化，确保：
    - conclusion 始终为去除首尾空白的字符串。
    - evidence 和 steps 始终为非空字符串的列表（无论原始是 list 还是 str）。
    此函数是写入缓存或传递给 render_card 前的标准化步骤。

    Args:
        raw: LLM 返回的原始卡片字典，可能缺失字段或类型不规范。
    Returns:
        规范化后的卡片字典，包含 conclusion（str）、evidence（list）、steps（list）。
    """
    def _list(v):
        """内部辅助：将任意类型值规范化为非空字符串列表。

        Args:
            v: 原始值，可能是 list、str 或其他类型。
        Returns:
            去除空白后的非空字符串列表。
        """
        if isinstance(v, list):
            # 已是列表：逐项转字符串、去空白，过滤掉空字符串
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            # 单个字符串：包装为单元素列表
            return [v.strip()]
        return []  # 其他类型（None、数字等）或空字符串：返回空列表

    return {
        "conclusion": str(raw.get("conclusion", "")).strip(),  # 将结论字段转为字符串并去空白，缺失时为空串
        "evidence": _list(raw.get("evidence")),  # 规范化证据列表（支持 list 或 str 输入）
        "steps": _list(raw.get("steps")),         # 规范化建议步骤列表（支持 list 或 str 输入）
    }


def render_card(card: dict) -> str:
    """把结构化卡片渲染为 markdown（用于记忆/缓存/兜底显示）。

    将包含 conclusion / evidence / steps 的卡片字典渲染为人类可读的
    Markdown 格式文本，供 summarize 节点构建 draft_answer 使用，
    同时也作为语义缓存中存储的文本表示。

    各部分规则：
    - conclusion：加粗标题 + 内容，始终首先展示（结论先行原则）。
    - evidence：加粗标题 + 无序列表，每条证据一行。
    - steps：加粗标题 + 有序列表，步骤按序编号。
    仅输出非空的部分，各部分之间用双换行分隔。

    Args:
        card: 规范化后的结构化卡片字典，含 conclusion、evidence、steps。
    Returns:
        渲染后的 Markdown 格式字符串，可直接呈现给用户。
    """
    parts = []  # 存放各段落的 Markdown 文本，最后用双换行拼接
    if card.get("conclusion"):
        parts.append(f"**结论**：{card['conclusion']}")  # 结论段：加粗标签 + 内容
    if card.get("evidence"):
        # 证据段：标题 + 无序列表，每条证据以 "- " 开头
        parts.append("**证据**：\n" + "\n".join(f"- {e}" for e in card["evidence"]))
    if card.get("steps"):
        # 步骤段：标题 + 有序列表，enumerate 从 1 开始编号，格式为 "1. 步骤内容"
        parts.append("**建议步骤**：\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(card["steps"], 1)))
    return "\n\n".join(parts)  # 各段落之间用双换行分隔，符合 Markdown 段落规范
