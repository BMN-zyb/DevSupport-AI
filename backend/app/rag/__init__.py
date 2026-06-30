# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""RAG 模块包初始化文件。

本包（app.rag）包含 DevSupport-AI 知识库检索增强生成（RAG）流程的全部子模块：
- store.py:      Milvus 向量存储管理（collection 生命周期、插入、检索）
- ingest.py:     文档入库流程（Markdown 解析、切片、向量化、写库）
- retriever.py:  混合检索（向量检索 + BM25 → RRF 融合）
- reranker.py:   重排序（Cross-Encoder 精排，取 top_n）
- compressor.py: 上下文压缩（按 token 预算裁剪，构建 LLM 上下文）
"""
