# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
# scripts 包初始化文件
#
# 职责说明：
#   本文件是 DevSupport-AI backend/scripts 包的 __init__.py，
#   作用是将 scripts 目录标识为 Python 包，使其可通过
#   python -m scripts.xxx 的方式调用各运维脚本。
#
#   scripts 包下包含以下运维脚本：
#     - init_db.py         : 建表脚本，初始化 MySQL 表结构和 Milvus collection
#     - seed_data.py       : 种子数据脚本，向 MySQL 写入演示/测试数据
#     - ingest_knowledge.py: 知识库入库脚本，将 Markdown 文档切片向量化写入 Milvus
#
#   本文件不导出任何模块级对象，仅起 Python 包标识作用。
