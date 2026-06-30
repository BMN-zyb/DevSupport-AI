# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""知识库 ingest 脚本：把 data/knowledge/*.md 切片向量化写入 Milvus。

用法（backend 目录下）：python -m scripts.ingest_knowledge

职责说明：
    本脚本是 DevSupport-AI 系统的知识库入库工具，负责：
    1. 读取 data/knowledge/ 目录下的全部 Markdown 格式知识文档。
    2. 对文档进行切片（chunking）并调用向量化模型生成 embedding。
    3. 将切片和向量写入 Milvus 向量数据库，供 RAG 检索链路使用。
    执行时会先清空旧的 collection（recreate=True），再重新写入，确保知识库内容最新。
"""

import asyncio  # 标准库：异步事件循环，用于驱动异步的 ingest_all 函数

from app.rag.ingest import ingest_all  # RAG 知识库入库核心函数：负责读取文档、切片、向量化、写入 Milvus


def main() -> None:
    """主函数：以同步方式触发异步知识库入库流程，并打印统计结果。

    执行流程：
        1. 调用 asyncio.run 驱动异步的 ingest_all，传入 recreate=True 表示重建 collection。
        2. ingest_all 返回统计字典，包含处理的文档数（documents）和切片数（chunks）。
        3. 打印入库完成信息。
    """
    stats = asyncio.run(ingest_all(recreate=True))  # 同步等待异步入库完成；recreate=True 先清空再重建，确保数据最新
    print(f"[ingest] 完成：文档 {stats['documents']} 篇，切片 {stats['chunks']} 个。")  # 打印入库统计摘要


if __name__ == "__main__":
    main()  # 直接运行脚本时执行主函数
