# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
# benchmark 包初始化文件
#
# 职责说明：
#   本文件是 DevSupport-AI backend/benchmark 包的 __init__.py，
#   作用是将 benchmark 目录标识为 Python 包，使其可通过
#   python -m benchmark.loadtest 的方式调用压测脚本。
#
#   benchmark 包下包含以下性能测试文件：
#     - loadtest.py  : 并发压测脚本，测量 P50/P95/吞吐量，
#                      基于 AgentTrace 做阶段耗时分解，
#                      对比冷启动与缓存预热两轮以量化语义缓存优化效果。
#
#   本文件不导出任何模块级对象，仅起 Python 包标识作用。
