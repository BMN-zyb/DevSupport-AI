# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/

# guardrail 包初始化文件
# 本包负责 DevSupport-AI 项目的护栏（Guardrail）功能，包含：
#   - desensitize.py：敏感信息识别与脱敏，支持 API Key / Token / 手机号 / 邮箱
#                    / 身份证 / 银行卡 / 签名等多种类型的检测与替换，
#                    用于用户输入、工具结果、最终输出三层防护。
#   - fallback.py：多级兜底策略，当整个 Agent 编排链发生不可恢复异常时，
#                 按意图类型返回预设规则话术，确保用户始终收到有意义的回复。
# 使用方：直接从子模块导入，如 `from app.guardrail.desensitize import desensitize_text`。
