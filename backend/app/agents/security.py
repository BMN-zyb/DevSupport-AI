# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""Security Agent：最终回复前的安全审查与脱敏（强制执行）。

- 对最终输出做敏感信息脱敏（API Key/Secret/Token/手机号/邮箱/身份证/签名）。
- 标注检测到的敏感类型，写入安全事件（可观测）。

本模块是 DevSupport-AI 多智能体系统中的安全专职子智能体。
职责：
  1. 在最终回复发送给用户之前，作为最后一道安全防线强制执行输出审查。
  2. 检测文本中可能泄露的敏感信息类型（如 API Key、密钥、Token、手机号、邮箱、
     身份证号、数字签名等），并记录敏感类型列表（可用于安全审计/告警）。
  3. 对检测到的敏感内容进行脱敏处理（如替换为 ***），确保不向用户暴露任何凭证信息，
     防止因 LLM 幻觉或日志数据泄露导致的安全事故。

设计原则：
  - 强制性：编排器应对所有最终回复调用此 Agent，不可绕过。
  - 无副作用：只读取和过滤文本，不修改任何业务状态。
  - 可观测：返回 sensitive_found 列表，供监控系统统计安全事件频率。
"""

# -----------------------------------------------------------------------
# 标准库导入
# -----------------------------------------------------------------------
# dataclass：定义轻量数据容器，存储安全审查结果
from dataclasses import dataclass, field

# -----------------------------------------------------------------------
# 项目内部模块导入
# -----------------------------------------------------------------------
from app.guardrail import desensitize  # 脱敏工具模块，提供 detect（检测）和 desensitize_text（脱敏）两个核心函数


@dataclass
class SecurityResult:
    """安全审查结果数据容器，由 review_output() 函数填充并返回给上层编排器。

    Attributes:
        clean_text:      脱敏后的安全文本，可直接发送给用户（替换了所有检测到的敏感信息）。
        sensitive_found: 检测到的敏感信息类型列表（如 ["api_key", "phone"]），
                         可用于安全日志记录和告警触发；未检测到时为空列表。
    """
    clean_text: str                                              # 脱敏后的安全文本（直接展示给用户）
    sensitive_found: list[str] = field(default_factory=list)    # 检测到的敏感类型列表（默认空列表）


def review_output(text: str) -> SecurityResult:
    """对 LLM 最终输出文本执行安全审查与脱敏，返回安全文本及敏感类型标记。

    该函数是安全审查的核心入口，顺序执行两个操作：
      1. 检测（detect）：扫描文本中的敏感信息模式，返回识别到的敏感类型列表。
      2. 脱敏（desensitize_text）：将敏感内容替换为掩码（如 *** 或 ****），
         保留文本可读性的同时消除安全风险。

    注意：检测和脱敏独立调用，不依赖彼此的结果，各自通过正则/规则引擎处理原始文本。

    Args:
        text: 待审查的原始文本（通常为 LLM 生成的最终回复内容）。

    Returns:
        SecurityResult 实例，包含脱敏后的安全文本和敏感类型列表。
        若无敏感信息：clean_text 与 text 内容相同，sensitive_found 为空列表。
    """
    # 检测文本中的敏感信息类型：返回识别到的敏感类型名称列表（如 ["api_key", "email"]）
    # 此步骤用于统计和告警，即使 found 为空，脱敏步骤依然执行（防御性设计）
    found = desensitize.detect(text)

    # 对原始文本执行脱敏：将所有敏感内容替换为掩码，生成安全的展示文本
    # 独立处理原始 text 而非依赖 found，确保脱敏覆盖率与检测结果一致
    clean = desensitize.desensitize_text(text)

    # 封装结果返回：clean_text 用于发送给用户，sensitive_found 用于安全日志/审计
    return SecurityResult(clean_text=clean, sensitive_found=found)
