# DevSupport-AI 手敲路线图（plan.md）

> 目标：把这套**面向 API 开放平台的多 Agent 智能客服系统**（FastAPI + LangGraph 后端 + React/Vite 前端）从零手敲一遍，边敲边读懂。本文件给出**最优敲码顺序**——严格按依赖自底向上、贴合 README 的「交互层 / Agent 编排层 / 基础能力层」三层叙事，让每敲完一层都能**立刻自检**。
>
> 用法：新建一个空目录作练习工程（下记 `$DST`），把**当前目录**当参考答案。每敲完一个文件，`diff` 比对自查（忽略行尾空白）：
> ```bash
> diff <(sed 's/[[:space:]]*$//' "$DST/backend/app/config.py") \
>      <(sed 's/[[:space:]]*$//'  backend/app/config.py)   # 无输出=一致
> ```

---

## 一、两条铁律（先读）

### 🟢 铁律 1：`config.py` 全字段都有默认值 → 无 `.env` 也能 import

[backend/app/config.py](backend/app/config.py) 里 `dashscope_api_key: str = ""`，**所有字段都有默认值**，模块末尾 `settings = get_settings()` 在导入时构造单例但**永不失败**。所以：

> **你可以不装 Docker、不填任何 key，直接把整个后端手敲完并逐文件 `python -c "import app.xxx"` 自检。** DashScope key 只在**真正调用 LLM/Embedding/Rerank 时**才需要（Tier B）。

### 🔌 铁律 2：连接惰性 + 基础设施在 Docker 里 → 先离线敲完，再起服务

MySQL/Redis/Milvus 的连接都在函数内惰性建立，且由 `docker-compose.yml` 一键拉起（宿主机端口**刻意错开**：MySQL `3307`、Redis `6380`、Milvus `19531`）。因此外部服务留到需要「真跑」时再起。

**两级自检贯穿全程**：
| 级别 | 需要什么 | 命令样例 | 验证了什么 |
|---|---|---|---|
| **A · 离线导入** | 什么都不用（或空 `.env`） | `cd backend && python -c "import app.main; print('ok')"` | 语法 + import 接线正确 |
| **B · 联机运行** | Docker infra + 真实 DashScope key | `make infra-up` → `make setup` → `make run` → `make health` | 逻辑真的能跑通 |

> ⚠️ 本项目**没有 pytest 单测套件**（`pyproject.toml` 里 `testpaths=["tests"]` 但无 `tests/` 目录）。运行级验证靠 **Makefile 流水线**（`init-db / seed / ingest / run / health / eval / bench`）与 `/docs` 手测。

**不要手敲**：`data/knowledge/*.md`（知识库中文文档，文件名在终端里显示为乱码是 GBK/UTF-8 显示问题，文件本身没坏——**直接拷贝 `data/` 整个目录**即可）、`frontend/package-lock.json`（`npm install` 自动生成）、`node_modules/`、`.venv/`、各 `__pycache__/`、Docker 数据卷。

---

## 二、依赖分层（箭头 = 依赖）

所有 `__init__.py` 都是**空壳**（只有注释、无 re-export），跨模块一律 `from app.<pkg>.<module> import ...` 直连子模块——所以 `__init__.py` 在脚手架阶段全部留成注释占位即可，**永不阻塞 import**。

```
【基础层 · 数据与配置地基】
config ★ ← 根(全默认值,import 永不失败)
  ├─ db(config) ─► models(db) ★           security(config)
  └───────────────┴─────► deps(db+models+security)  ← FastAPI 依赖注入/鉴权

【基础能力层 · README「基础能力」】
llm: client(config) · router(config)                         ← 模型分层接入 DashScope
guardrail: desensitize · fallback(纯逻辑,叶子)               ← 脱敏 + 多级兜底
cache: redis_client(config) ─► route_cache · semantic_cache(+llm)   memory/session(cache) ← 缓存分层 + 会话记忆
rag: store(config) · reranker(llm) · compressor · retriever(llm+store) · ingest(db+llm+models+store) ← 混合检索+精排+压缩
tools: registry(config+db+guardrail+models) ─► apikey · billing_tools · logs · ticket_tools  ← 工具注册中心 + 4 工具
observability: cost(db+models) · trace(db+models)            ← Token 成本 + 链路追踪

【Agent 编排层 · README「编排层」】
agents: state · util (叶子)
        intent_router(llm) · security(guardrail) · ticket(tools)
        doc_rag(util+config+llm+rag)  ← 被 billing / api_diagnostic 复用的共享子 Agent
        billing(doc_rag+...) · api_diagnostic(doc_rag+...)
        supervisor ★(全部 agents + state/util/cache/config/guardrail/llm/memory/observability/tools) ← LangGraph 编排大脑,最后

【交互层 · README「交互层」】
schemas: auth · chat (pydantic 叶子)
api: auth · docs(rag) · eval(observability) · traces · conversations · tickets · chat(supervisor,SSE) · workbench ★
main ★(挂载 api + config)  ← FastAPI 应用装配
scripts: init_db(db+rag) · seed_data(models+security) · ingest_knowledge(rag)
eval/run_eval · benchmark/loadtest

【前端 · React + Vite + TS】
api.ts(叶子) → components(Highlight → DiagnosisCard; TraceFlow) → pages(×7) → App.tsx → main.tsx
```

### 阶段一览

| 阶段 | 主题（README 层） | 关键文件（约行） | Tier B 需要 |
|---|---|---|---|
| 0 | 脚手架 + infra | 目录树 / pyproject / **.env** / 空 `__init__` / `docker compose up` / `npm install` | Docker |
| 1 | 数据与配置地基 | config(121)·db(90)·models(411)·security(131)·deps(158) | — |
| 2 | LLM 接入层 | llm/client(241)·llm/router(39) | DashScope |
| 3 | 护栏·缓存·记忆 | guardrail(desensitize213/fallback60)·cache(redis32/route82/semantic139)·memory/session(128) | Redis |
| 4 | RAG 检索 | rag: store(198)·reranker(44)·compressor(78)·retriever(139)·ingest(196) + ingest_knowledge | Milvus+key |
| 5 | 工具层 | tools: registry(204)·apikey(92)·billing_tools(160)·logs(167)·ticket_tools(175) | MySQL |
| 6 | 可观测 | observability: cost(79)·trace(175) | MySQL |
| 7 | 多 Agent 编排 | state(66)·util(113)·intent_router(168)·security(73)·ticket(124)·doc_rag(184)·billing(138)·api_diagnostic(209)·supervisor(457) | 全栈 |
| 8 | schemas + API | schemas(auth48/chat48)·api(auth104/docs101/eval121/traces186/conversations159/tickets201/chat189/workbench319) | 全栈 |
| 9 | 装配 + 数据脚本 + 起服务 | main(74)·init_db(54)·seed_data(346) → `make setup/run/health` | 全栈 |
| 10 | 评估 + 压测 | eval/run_eval(155)·benchmark/loadtest(154) | 全栈 |
| 11 | 前端 | scaffold → api.ts(100) → components → pages(×7) → App(77) → main(17) | 全栈 |
| 12 | 端到端联调 | 登录 + 四大场景 | 全栈 |

---

## 三、分阶段详解（后端）

### 阶段 0 · 脚手架 + 基础设施

- 建目录（与参考同构）：`backend/app/{agents,api,cache,guardrail,llm,memory,observability,rag,schemas,tools}`、`backend/{scripts,eval,benchmark}`、`frontend/src/{components,pages}`、`data/knowledge`。
- 抄根级散文件：[docker-compose.yml](docker-compose.yml)、[Makefile](Makefile)、[README.md](README.md)。
- 抄 backend 配置：[backend/pyproject.toml](backend/pyproject.toml)；`cp backend/.env.example backend/.env`（key 可先留空，Tier A 不需要）。
- 建**空** `__init__.py`（每个 `app` 子包 + `scripts/eval/benchmark` + `app/__init__.py`）——全是注释占位，**不 re-export**，随时可 import。
- 拷 `data/`（知识库 md，勿手敲）。
- 后端 venv：`cd backend && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`。
- 前端依赖：`cd frontend && npm install`（生成 `package-lock.json` + `node_modules`，勿手敲）。
- （可选，联机才需要）`make infra-up` 起 MySQL/Redis/Milvus——**Milvus 首启约 1~2 分钟**，等 `make infra-status` 全 `healthy`。

**Tier A**：`cd backend && python -c "import app; print(app.__version__)"`。

---

### 阶段 1 · 数据与配置地基（config → db → models → security → deps）

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/config.py](backend/app/config.py) ★ | 121 | pydantic-settings 全局配置 + `settings` 单例 + MySQL DSN | 仅 pydantic |
| 2 | [backend/app/db.py](backend/app/db.py) | 90 | SQLAlchemy 异步/同步 engine + session + `Base` | config |
| 3 | [backend/app/models.py](backend/app/models.py) ★ | 411 | 全部 ORM 模型（租户/账号/工单/日志/账单/追踪…）——最大地基文件 | db |
| 4 | [backend/app/security.py](backend/app/security.py) | 131 | 密码哈希（bcrypt）+ JWT 签发/校验 | config |
| 5 | [backend/app/deps.py](backend/app/deps.py) | 158 | FastAPI 依赖：DB 会话 / 当前用户 / 租户隔离 | db, models, security |

**Tier A**：`python -c "import app.deps, app.models; print('foundation ok')"`

---

### 阶段 2 · LLM 接入层（client → router）· README 模型分层

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/llm/client.py](backend/app/llm/client.py) ★ | 241 | DashScope（OpenAI 兼容）：chat / 流式 / embedding / rerank，重试 + 单例 | config |
| 2 | [backend/app/llm/router.py](backend/app/llm/router.py) | 39 | `model_for()` 按任务分层选 small/large 模型（降本保质） | config |

**Tier A**：`python -c "import app.llm.client, app.llm.router; print('llm ok')"`
**Tier B**：填真实 `DASHSCOPE_API_KEY` 后，写 3 行脚本让 `client` 回一句话冒烟。

---

### 阶段 3 · 护栏 · 缓存 · 记忆 · README 脱敏兜底 + 缓存分层

安全护栏是纯逻辑叶子，可先敲并**离线单测**（脱敏正则最适合逐条验证）。

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/guardrail/desensitize.py](backend/app/guardrail/desensitize.py) ★ | 213 | 脱敏：Key/Token/手机/邮箱/身份证/银行卡/签名（三层防护，纯逻辑） | —（叶子） |
| 2 | [backend/app/guardrail/fallback.py](backend/app/guardrail/fallback.py) | 60 | 多级兜底话术（编排异常时按意图返回预设） | —（叶子） |
| 3 | [backend/app/cache/redis_client.py](backend/app/cache/redis_client.py) | 32 | 异步 Redis 单例 | config |
| 4 | [backend/app/cache/route_cache.py](backend/app/cache/route_cache.py) | 82 | 路由缓存（缓存意图结果，跳过 LLM） | cache.redis_client |
| 5 | [backend/app/cache/semantic_cache.py](backend/app/cache/semantic_cache.py) | 139 | 语义缓存（向量相似命中热点答案） | cache.redis_client, config, llm |
| 6 | [backend/app/memory/session.py](backend/app/memory/session.py) | 128 | Redis 历史消息窗口 + 实体记忆 | cache.redis_client |

**Tier A**：`python -c "import app.guardrail.desensitize, app.cache.semantic_cache, app.memory.session; print('ok')"`
**Tier B（脱敏可离线）**：`python -c "from app.guardrail.desensitize import desensitize_text; print(desensitize_text('key=sk-abc123def456'))"`（应看到密钥被替换）。

---

### 阶段 4 · RAG 检索系统 · README 混合检索 + Rerank

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/rag/store.py](backend/app/rag/store.py) ★ | 198 | Milvus collection 生命周期 + 插入 + 向量检索 | config |
| 2 | [backend/app/rag/reranker.py](backend/app/rag/reranker.py) | 44 | gte-rerank-v2 精排取 top_n | llm |
| 3 | [backend/app/rag/compressor.py](backend/app/rag/compressor.py) | 78 | 上下文压缩（按 token 预算裁剪） | —（叶子/llm） |
| 4 | [backend/app/rag/retriever.py](backend/app/rag/retriever.py) ★ | 139 | 混合检索：向量 + BM25 → RRF 融合 | llm, rag.store |
| 5 | [backend/app/rag/ingest.py](backend/app/rag/ingest.py) | 196 | 入库：md 解析 → 切片 → 向量化 → 写库 | db, llm, models, rag.store |
| 6 | [backend/scripts/ingest_knowledge.py](backend/scripts/ingest_knowledge.py) | 33 | CLI 知识库入库 | rag.ingest |

**Tier A**：`python -c "import app.rag.retriever, app.rag.ingest; print('rag ok')"`
**Tier B**：infra 起好后 `make init-db`（建 Milvus collection）→ `make ingest`（灌 `data/knowledge`）。

---

### 阶段 5 · 工具层（registry 先行 → 4 个工具）· README 工具能力

`registry` 定义注册中心/装饰器 + `load_tools()`；4 个工具用装饰器注册，故 registry 必须**先**敲。

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/tools/registry.py](backend/app/tools/registry.py) ★ | 204 | 工具注册中心 + 超时/脱敏包装 + `load_tools()` | config, db, guardrail, models |
| 2 | [backend/app/tools/apikey.py](backend/app/tools/apikey.py) | 92 | API Key 状态查询工具 | db, models, tools.registry |
| 3 | [backend/app/tools/logs.py](backend/app/tools/logs.py) | 167 | 调用日志查询工具（按 request_id 等） | db, models, tools.registry |
| 4 | [backend/app/tools/billing_tools.py](backend/app/tools/billing_tools.py) | 160 | 账单/用量查询工具 | db, models, tools.registry |
| 5 | [backend/app/tools/ticket_tools.py](backend/app/tools/ticket_tools.py) | 175 | 工单创建/更新工具 | db, models, tools.registry |

**Tier A**：`python -c "from app.tools import registry; registry.load_tools(); print('tools ok')"`

---

### 阶段 6 · 可观测（cost → trace）· README 成本 + 链路

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/observability/cost.py](backend/app/observability/cost.py) | 79 | 按会话/租户记录 LLM token 消耗 | db, models |
| 2 | [backend/app/observability/trace.py](backend/app/observability/trace.py) | 175 | 逐节点链路追踪（耗时/token/命中/状态） | db, models |

**Tier A**：`python -c "import app.observability.cost, app.observability.trace; print('obs ok')"`

---

### 阶段 7 · 多 Agent 编排 · README Agent 编排层（大脑）

叶子在前，共享子 Agent（`doc_rag`）居中，`supervisor` 整合全部、压轴。

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/agents/state.py](backend/app/agents/state.py) ★ | 66 | LangGraph 共享 `State`（TypedDict） | —（叶子） |
| 2 | [backend/app/agents/util.py](backend/app/agents/util.py) | 113 | LLM 输出 JSON 清洗/解析工具 | —（叶子） |
| 3 | [backend/app/agents/intent_router.py](backend/app/agents/intent_router.py) ★ | 168 | 意图路由（首节点） | llm, llm.router |
| 4 | [backend/app/agents/security.py](backend/app/agents/security.py) | 73 | 安全审查节点（固定串在最后，脱敏不被绕过） | guardrail |
| 5 | [backend/app/agents/ticket.py](backend/app/agents/ticket.py) | 124 | 工单 Agent | tools.registry |
| 6 | [backend/app/agents/doc_rag.py](backend/app/agents/doc_rag.py) ★ | 184 | 文档问答子 Agent（**被 billing/api_diagnostic 复用**） | agents.util, config, llm, rag |
| 7 | [backend/app/agents/billing.py](backend/app/agents/billing.py) | 138 | 账单 Agent | agents.doc_rag, agents.util, llm, tools.registry |
| 8 | [backend/app/agents/api_diagnostic.py](backend/app/agents/api_diagnostic.py) ★ | 209 | API 诊断 Agent（查日志+Key+文档） | agents.doc_rag, agents.util, db, llm, models, tools.registry |
| 9 | [backend/app/agents/supervisor.py](backend/app/agents/supervisor.py) ★ | 457 | LangGraph Supervisor：意图路由→并行专业 Agent→工单→汇总→安全审查（**最大文件**） | 全部 agents + state/util/cache/config/guardrail/llm/memory/observability/tools |

**Tier A**：`python -c "import app.agents.supervisor; print('agents ok')"`（编译整条编排链的 import）。
**Tier B**：全栈起好后由 `/api/chat` 或 `eval` 驱动。

---

### 阶段 8 · schemas + API 路由 · README 交互层（REST + SSE）

schemas 是 pydantic 叶子，先敲；路由由简到繁，`chat`（SSE 流式）与 `workbench`（最大）压后。

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/schemas/auth.py](backend/app/schemas/auth.py) | 48 | 登录/用户/Token DTO | —（叶子） |
| 2 | [backend/app/schemas/chat.py](backend/app/schemas/chat.py) | 48 | 对话/反馈/工单 DTO | —（叶子） |
| 3 | [backend/app/api/auth.py](backend/app/api/auth.py) | 104 | 登录鉴权路由 | db, deps, models, schemas.auth, security |
| 4 | [backend/app/api/docs.py](backend/app/api/docs.py) | 101 | 知识库文档路由 | deps, rag.ingest |
| 5 | [backend/app/api/traces.py](backend/app/api/traces.py) | 186 | 链路追踪查询 | db, deps, models |
| 6 | [backend/app/api/eval.py](backend/app/api/eval.py) | 121 | 运营指标/评估触发 | db, deps, models, observability |
| 7 | [backend/app/api/conversations.py](backend/app/api/conversations.py) | 159 | 会话历史 | db, deps, guardrail, models |
| 8 | [backend/app/api/tickets.py](backend/app/api/tickets.py) | 201 | 工单管理 | db, deps, models, schemas.chat |
| 9 | [backend/app/api/chat.py](backend/app/api/chat.py) ★ | 189 | 核心对话（SSE 流式，驱动 supervisor） | agents, db, deps, models, schemas.chat |
| 10 | [backend/app/api/workbench.py](backend/app/api/workbench.py) ★ | 319 | 技术支持工作台（诊断面板/接管，最大路由） | db, deps, guardrail, llm, models, schemas.chat |

**Tier A**：`python -c "import app.api.chat, app.api.workbench; print('api ok')"`

---

### 阶段 9 · 应用装配 + 数据脚本 + 起服务

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [backend/app/main.py](backend/app/main.py) ★ | 74 | FastAPI 装配：挂载全部路由 + 健康检查 + lifespan（`load_tools` 等） | api, config |
| 2 | [backend/scripts/init_db.py](backend/scripts/init_db.py) | 54 | 建 MySQL 表 + Milvus collection（`--recreate`） | db, rag.store |
| 3 | [backend/scripts/seed_data.py](backend/scripts/seed_data.py) | 346 | 灌种子（租户/账号/日志/账单，含预置账号） | db, models, security |

**Tier A（capstone）**：`python -c "import app.main; print(len(app.main.app.routes),'routes')"`——**一句话验证整个后端 import 接线零错**。
**Tier B（首次真跑）**：`make infra-up`（等 Milvus healthy）→ 填 `.env` key → `make setup`（= init-db + seed + ingest）→ `make run` → `make health`（返回 `{"status":"ok"}`）→ 开 `http://localhost:8000/docs`。

---

### 阶段 10 · 评估 + 压测

| 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|
| [backend/eval/run_eval.py](backend/eval/run_eval.py) | 155 | 标准评估集：意图/引用/脱敏等指标（`make eval`） | agents, cache.redis_client, db, guardrail |
| [backend/benchmark/loadtest.py](backend/benchmark/loadtest.py) | 154 | 并发压测 + 阶段耗时分解 + 缓存前后对比（`make bench`） | agents, cache.redis_client, db, models |

**Tier B**：`make eval` / `make bench`。

---

### 阶段 11 · 前端（React + Vite + TS）

顺序：脚手架 → API 客户端 → 组件（叶子）→ 页面 → 路由装配 → 入口。

| 顺序 | 文件 | 行 | 作用 | 依赖 |
|---|---|---|---|---|
| 1 | [frontend/package.json](frontend/package.json)·[tsconfig.json](frontend/tsconfig.json)·[vite.config.ts](frontend/vite.config.ts)·[index.html](frontend/index.html) | — | 脚手架（`npm install` 已在 Phase 0 跑过） | — |
| 2 | [frontend/src/api.ts](frontend/src/api.ts) ★ | 100 | 后端 API 客户端（fetch/SSE + token） | —（叶子） |
| 3 | [frontend/src/components/Highlight.tsx](frontend/src/components/Highlight.tsx) | 61 | 关键词高亮 | — |
| 4 | [frontend/src/components/DiagnosisCard.tsx](frontend/src/components/DiagnosisCard.tsx) | 41 | 诊断卡片 | Highlight |
| 5 | [frontend/src/components/TraceFlow.tsx](frontend/src/components/TraceFlow.tsx) | 50 | React Flow 链路可视化 | — |
| 6 | [frontend/src/pages/Login.tsx](frontend/src/pages/Login.tsx) | 69 | 登录页 | api |
| 7 | [frontend/src/pages/MyTickets.tsx](frontend/src/pages/MyTickets.tsx)·[Docs.tsx](frontend/src/pages/Docs.tsx)·[Metrics.tsx](frontend/src/pages/Metrics.tsx) | 35/70/80 | 工单/文档/运营指标页 | api |
| 8 | [frontend/src/pages/Conversations.tsx](frontend/src/pages/Conversations.tsx) | 106 | 会话列表 | api, DiagnosisCard, Highlight |
| 9 | [frontend/src/pages/Chat.tsx](frontend/src/pages/Chat.tsx) ★ | 171 | 智能助手对话页（SSE 流式） | api, DiagnosisCard, Highlight |
| 10 | [frontend/src/pages/Workbench.tsx](frontend/src/pages/Workbench.tsx) ★ | 202 | 技术支持工作台（链路+诊断+接管） | api, TraceFlow, Highlight, DiagnosisCard |
| 11 | [frontend/src/App.tsx](frontend/src/App.tsx) ★ | 77 | 路由装配（挂载 7 个页面） | api + 全部 pages |
| 12 | [frontend/src/main.tsx](frontend/src/main.tsx) | 17 | 渲染入口 | App |

**Tier A**：`cd frontend && npm run build`（tsc + vite 编译，验证类型与 import）。
**Tier B**：`make front` → 开 `http://localhost:5173`。

---

### 阶段 12 · 端到端联调

全栈起好（`make infra-up` + `make setup` + `make run` + `make front`），浏览器 `http://localhost:5173` 用预置账号（密码 `password123`）登录，跑 README 的四大场景：401 诊断 / 429 复合问题 / 签名文档问答 / 账单解释+高风险兜底。内部侧用 `support1` 看工作台链路可视化，`admin` 看运营指标并一键跑评估集。

---

## 四、收尾自查清单

- [ ] 空 `.env`、无 Docker 下，`cd backend && python -c "import app.main"` 通过（**capstone：整个后端 7500 行 import 接线零错**）。
- [ ] `cd frontend && npm run build` 通过（前端类型与 import 零错）。
- [ ] infra 起好 + 真实 key 后 `make setup` 一条龙成功；`make health` 返回 `{"status":"ok"}`；`/docs` 可见。
- [ ] 四大场景在前端跑通；`make eval` / `make bench` 有输出。
- [ ] 逐文件 `diff` 与原件零差异（或仅空白差异）。
- [ ] 读懂 `supervisor.py` 如何把「意图路由 → 专业 Agent 并行 → 工单 → 汇总 → 安全审查」编排成 LangGraph，以及安全审查为何固定串在最后。

## 五、给手敲者的三点提醒

1. **先离线、后联机**：全默认值配置 + 惰性连接 → 用空 `.env`、不起 Docker 就能把后端全敲完并逐层 `import` 自检；infra + key 留到 Phase 9 真跑。
2. **`__init__.py` 是空壳**：所有跨模块导入都直连子模块（`from app.rag.store import ...`），`__init__` 不 re-export，脚手架阶段留注释占位即可，永不阻塞。
3. **共享子 Agent 与顺序**：`doc_rag` 被 `billing`/`api_diagnostic` 复用，必须先于它们；`supervisor` 依赖全部 6 个域 Agent，必须最后；`tools/registry` 必须先于 4 个工具。
