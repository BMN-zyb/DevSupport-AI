# @author: 公众号：IT杨秀才
# @doc: 后端，AI Agent知识进阶，后端、AI大模型、场景题面试大全：https://golangstar.cn/
"""DevSupport AI 后端应用包。

本文件是 app 包的初始化模块（package init），在 Python 导入 app 包时首先被执行。
当前职责：
  - 声明包版本号 __version__，供其他模块通过 app.__version__ 访问；
  - main.py 中的 FastAPI 实例 version 参数与此版本号保持同步（均为 "0.1.0"）。
"""

# 包版本号：遵循语义化版本规范（SemVer），格式为 MAJOR.MINOR.PATCH
# 修改版本时须同步更新 main.py 中 FastAPI(version=...) 的参数
__version__ = "0.1.0"
