# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
# eval 包初始化文件
#
# 职责说明：
#   本文件是 DevSupport-AI backend/eval 包的 __init__.py，
#   作用是将 eval 目录标识为 Python 包，使其可通过
#   python -m eval.run_eval 的方式调用评测脚本。
#
#   eval 包下包含以下评测相关文件：
#     - run_eval.py        : 离线评测运行器，跑多 Agent 链路并计算各项指标与 Badcase 归因
#     - dataset.jsonl      : 主评测数据集（JSONL 格式），含问题、标注意图、实体、引用等字段
#     - security_set.jsonl : 脱敏评测集（JSONL 格式），含含敏感信息的文本及脱敏类型标注
#
#   本文件不导出任何模块级对象，仅起 Python 包标识作用。
