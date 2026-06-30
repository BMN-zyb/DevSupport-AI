# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""建表脚本：创建 MySQL 全部表 + Milvus collection。

用法（在 backend 目录下）：
    python -m scripts.init_db            # 增量建表
    python -m scripts.init_db --recreate # 删表重建 + 重建 Milvus collection

职责说明：
    本脚本是 DevSupport-AI 系统的数据库初始化工具，负责：
    1. 根据 SQLAlchemy ORM 模型定义在 MySQL 中建表（支持增量建表，已存在的表不会重建）。
    2. 在向量数据库 Milvus 中创建知识库 collection（用于存储文档向量，支撑 RAG 检索）。
    使用 --recreate 参数可强制删除并重建全部结构，适合开发调试阶段重置环境。
"""

import argparse  # 标准库：命令行参数解析，用于支持 --recreate 参数

from app import models  # noqa: F401  导入全部 ORM 模型，确保所有模型注册到 Base.metadata（副作用 import）
from app.db import Base, sync_engine  # Base: SQLAlchemy 声明式基类（含 metadata）；sync_engine: 同步数据库引擎
from app.rag import store  # RAG 向量存储模块，封装了 Milvus collection 的创建与管理逻辑


def main() -> None:
    """主函数：解析命令行参数，初始化 MySQL 表结构和 Milvus collection。

    命令行参数：
        --recreate: 若指定此参数，先删除全部 MySQL 表和 Milvus collection，再重新创建。
                    不指定时为增量模式，仅创建不存在的表，不影响已有数据。
    执行流程：
        1. 解析 --recreate 参数。
        2. 若 --recreate：删除所有 MySQL 表。
        3. 创建（或补充）MySQL 表。
        4. 创建（或重建）Milvus collection。
        5. 打印就绪信息。
    """
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument("--recreate", action="store_true", help="删表重建")  # 注册 --recreate 布尔标志
    args = parser.parse_args()  # 解析命令行参数，结果存入 args

    if args.recreate:  # 仅在明确传入 --recreate 时才执行破坏性操作
        print("[init_db] 删除所有表 ...")
        Base.metadata.drop_all(sync_engine)  # 按依赖顺序删除所有已注册的 MySQL 表（DDL DROP TABLE）
    print("[init_db] 创建 MySQL 表 ...")
    Base.metadata.create_all(sync_engine)  # 根据 ORM 模型定义创建尚不存在的表（DDL CREATE TABLE IF NOT EXISTS）
    tables = sorted(Base.metadata.tables.keys())  # 获取所有已注册表名并排序，用于打印确认
    print(f"[init_db] MySQL 表就绪（{len(tables)}）：{tables}")  # 输出表数量和表名列表

    print("[init_db] 创建 Milvus collection ...")
    store.ensure_collection(recreate=args.recreate)  # 创建或重建 Milvus collection（向量数据库，用于知识库 RAG 检索）
    print(f"[init_db] Milvus collection '{store.COLLECTION}' 就绪。")  # 打印 collection 名称确认创建完成


if __name__ == "__main__":
    main()  # 直接运行脚本时执行主函数
