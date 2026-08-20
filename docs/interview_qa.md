# 面试题问集 — Personal Knowledge Agent

> 每个主题按「难点 → 遇到的问题 → 怎么解决 → 为什么这么选」组织，并附**面试官追问**与**应答要点**。
> 所有内容都基于项目真实代码与真实踩坑，可直接背诵、可被深挖。

---

## 目录

1. [混合检索：为什么向量 + BM25 双路？](#1-混合检索)
2. [RRF 融合：为什么选 RRF 而不是加权求和？](#2-rrf-融合)
3. [中文检索：jieba + TF-IDF 的局限与处理](#3-中文检索)
4. [重排序：为什么 Cross-Encoder？为什么可降级？](#4-重排序)
5. [[[wikilink]] 图谱：为什么做 BFS 扩展？](#5-wikilink-图谱)
6. [增量索引：为什么 watchdog 而不是定时重建？](#6-增量索引)
7. [单 Agent → 多角色 → Supervisor：为什么演进？](#7-多-agent-编排演进)
8. [workflow 式 vs Supervisor 式：怎么选？](#8-workflow-vs-supervisor)
9. [MCP 协议：为什么用它暴露工具？](#9-mcp-协议)
10. [中间件体系：为什么 5 层？](#10-中间件体系)
11. [全链路降级：无 API Key 也能跑？](#11-全链路降级)
12. [评测：为什么 LLM-as-judge？](#12-评测方法)
13. [踩坑 1：子线程 asyncio.run 导致 Agent 挂起](#13-踩坑-1异步挂起)
14. [踩坑 2：Qdrant 本地模式文件锁冲突](#14-踩坑-2qdrant-锁冲突)
15. [踩坑 3：LLM API 无超时导致评测卡死](#15-踩坑-3llm-api-超时)

---

## 1. 混合检索

**难点**：纯向量检索对中文关键词、专有名词（`retry_middleware.py`、`@server.tool()`）不敏感；纯 BM25 又无法理解语义近义（"踩坑"与"遇到的问题"）。

**问题**：初期只用向量检索时，含英文术语/精确文件名的查询召回差。

**解决**：向量（Dense）+ BM25（Sparse）双路并行检索，RRF 融合。

**为什么**：
- 向量捕捉语义，BM25 捕捉精确词匹配，两者互补（Hybrid Search 是工业界 RAG 标配）
- 中文场景下 BM25 用 jieba 分词，`" ".join(jieba.cut(text))` 后建 TF-IDF 矩阵
- 成本低：BM25 纯内存计算，无 API 依赖，还能兜底 embedding 服务不可用的情况

**追问**：为什么不用 elasticsearch？
> 答：项目是个人知识库规模（22 篇笔记），Qdrant 已承担向量存储；BM25 用 scikit-learn 几百行即可实现，避免引入 ES 的运维成本。若数据量到百万级才会考虑 ES/专门的稀疏检索服务。

## 2. RRF 融合

**难点**：两路结果分数**不可直接比较**（余弦相似度 vs BM25 分数，量纲完全不同）。

**问题**：直接对分数加权求和会偏向分数绝对值大的一路。

**解决**：Reciprocal Rank Fusion —— 只依赖排名不依赖分数：

```
score(d) = Σ 1 / (k + rank(d))     # k=60
```

**为什么**：
- 对分数尺度鲁棒，两路结果天然公平
- 实现 10 行代码，无需调权重（对比 weighted-sum 需要人工调参）
- 去重键用 `heading ||| text`，避免同一 chunk 被双路重复计入

**追问**：RRF 的 k 为什么取 60？
> 答：k 越小，排名靠前的项权重越大；k=60 是原论文（Cormack et al.）的经验值，在多数数据集上表现稳定。我的消融测试也验证了 RRF 优于单路。

## 3. 中文检索

**难点**：TF-IDF 对中文分词敏感，且忽略词序。

**问题**：直接按字符切分，中文短语（"重试逻辑"）无法形成有效 token；jieba 引入后 BM25 命中率显著提升。

**解决**：jieba 分词 + TfidfVectorizer；查询和文档用同一分词流程保证一致性。

**为什么**：中文没有天然空格分隔，必须分词；jieba 轻量、离线、易集成，符合项目"本地优先"定位。

**追问**：jieba 的缺陷？
> 答：新词/专业术语可能切错（如 `bge-reranker-v2-m3`），可通过自定义词典解决；本项目规模小，影响可接受。

## 4. 重排序

**难点**：向量检索的相似度是"单塔"粗排，无法细粒度比较 query-doc 相关性。

**问题**：top-15 候选中，真正相关的可能排在第 8-10 名。

**解决**：BGE-Reranker-v2-m3（Cross-Encoder）对融合后的候选精排，取 top-5。

**为什么**：
- Cross-Encoder 把 query 和 doc 拼接后交互计算，精度远高于双塔 embedding
- 但推理成本高，所以**只对 top-15 精排**（两阶段：粗排召回 → 精排重排）

**追问**：为什么做成可选？
> 答：FlagEmbedding 是重依赖（需下载模型权重），且纯本地推理慢。我把重排设计为可选：没有它就按 fusion_score 降序兜底——这体现"降级优先"的工程习惯。

## 5. [[wikilink]] 图谱

**难点**：Obsidian 笔记天然通过 [[wikilink]] 关联，但向量检索看不到这种"显式关系"。

**问题**：搜"ellie 重试"只命中 [[ellie 踩坑记录]]，却漏掉它链接的 [[Code Review 踩坑记录]]。

**解决**：解析所有 [[wikilink]] 建**内存邻接表**，检索结果出来后对 top 命中做 **BFS 1-hop 扩展**，把关联笔记补进结果。

**为什么**：
- 显式链接是"人工标注的相关性"，比 embedding 隐式相似更可信
- BFS 1-hop 足够（Obsidian 笔记关联通常一层就到目标），多 hop 会引入噪音
- 内存邻接表（dict of set）构建 O(n)、查询 O(1)，22 篇笔记毫秒级

**追问**：为什么不用图数据库（Neo4j）？
> 答：规模小、关系简单，dict-of-set 完全够用且零依赖；图数据库是规模大/需图查询语言时才值得。

## 6. 增量索引

**难点**：vault 是活的，用户会增删改笔记，全量重建代价高。

**问题**：每次启动全量 ingest + embedding，22 篇笔记也要几十秒；改一篇笔记也要全量重来。

**解决**：watchdog 监听 vault 目录，文件变更（create/modify/delete）时增量更新对应 chunk 与 wikilink 图。

**为什么**：watchdog 是文件系统事件驱动（inotify/FSEvents/ReadDirectoryChangesW），比"定时轮询"实时且省资源；增量更新只动变更文件，比全量重建快一个数量级。

**追问**：增量更新的一致性怎么保证？
> 答：删除文件时同步从向量库删对应 doc_id、从邻接表删节点；边界情况（改文件名）按"删旧建新"处理。规模小，未引入事务，但这正是可讲的取舍点。

## 7. 多 Agent 编排演进

**难点**：单 Agent（create_agent）把"查询分析 → 检索 → 综合"全塞在一个 LLM 循环里，靠 prompt 引导，**决策不可见、不可控**。

**问题**：README 里的多步推理示例（搜笔记 → 查图谱 → 查源码）只能靠 prompt 期望 LLM 自觉执行，没有结构保证。

**解决**：演进为显式状态机——
1. **workflow 式**（`multi_agent`）：Planner 拆查询 → Executor 并行检索 → Summarizer 综合 → Critic 检查 → 不满意则 Rewrite 重查（闭环）
2. **Supervisor 式**（`supervisor_agent`）：总调度 LLM 每轮看全局历史，动态路由到 search/code/graph worker，直到选 answer

**为什么**：
- 决策从"prompt 期望"变成"图结构/状态机保证"——这就是 README 里"只有 LangGraph 状态机能做"的落点
- Critic 双通道（规则通道判空 + LLM judge 判信息缺口）避免纯 LLM 判断的不可靠
- 轮数上限（2 / 6）保证任何情况下不死循环

**追问**：为什么不用 LlamaIndex 的 agent 或别的框架？
> 答：LangGraph 提供显式状态、条件边、checkpointer 和中间件体系，能精确表达"闭环重检索"这类控制流；LlamaIndex 强在检索管线本身，两者分工（这也回答了"为什么用 LangGraph 不用纯 LlamaIndex"）。

## 8. workflow vs Supervisor

**难点**：多 Agent 编排有两种主流模式，选错会过度设计或能力不足。

**决策**：
| | workflow 式 | Supervisor 式 |
|---|---|---|
| 路由时机 | 图结构静态写死 | 每轮 LLM 动态决策 |
| 适用 | 流程固定、角色明确 | 任务边界不固定、需临场判断 |
| 代价 | 延迟低、可控 | 每轮多一次 LLM 调用 |

**为什么这么选**：当前任务（知识问答）worker 少、流程相对固定，两种都可行——所以我**两个都实现了并存**，用同一评测集量化对比（质量/延迟/轮数），用数据说话而不是拍脑袋。

**追问**：为什么不用 handoffs（Agent 互交）？
> 答：handoffs 适合子任务需要互相移交上下文的场景；我的 worker 之间无依赖，Supervisor 中央视野足够，去中心化只会增加复杂度。

## 9. MCP 协议

**难点**：项目能力（检索 vault/源码）要被 Claude Desktop、Cursor 等外部 Agent 使用。

**问题**：直接函数调用只在本进程内可用；HTTP API 又要自定义协议、各家适配。

**解决**：用 MCP（Model Context Protocol）暴露 7 个工具，外部客户端通过 stdio 拉起 `src/mcp/server.py` 即插即用。

**为什么**：MCP 是 Anthropic 推的开放标准，Claude Desktop/Cursor 原生支持——一次实现，多端复用，不用给每家写适配器。

**追问**：MCP 和普通工具函数什么区别？
> 答：MCP 是**跨进程协议**（JSON-RPC + stdio/SSE），工具元数据（name/description/参数 schema）由服务端声明，客户端动态发现；普通函数调用是进程内、编译期绑定。我的 7 个 LangChain 工具零改动直接包成 MCP tool。

## 10. 中间件体系

**难点**：生产级 Agent 需要考虑安全、上下文长度、失败恢复，而不是"能回答就行"。

**解决**：5 层中间件链（LangGraph/LangChain middleware）：
- **Guardrails**：模型进 Agent 前做主题检查（block_off_topic 可配）
- **Summarization**：上下文超 130k tokens 时压缩，保留最近 30k
- **ToolRetry**：工具调用失败重试（3 次）
- **ModelRetry**：模型调用失败重试（保守 1 次，避免本地排队）
- **Fallback**：主模型失败自动降级到备选模型链（Gemini → Claude，仅当有对应 key）

**为什么**：把横切关注点从 agent 主逻辑剥离，中间件可独立测试、可插拔组合。

**追问**：Guardrails 怎么实现的？
> 答：before_agent 钩子用独立小模型对用户输入做分类检查，命中敏感/离题则短路；判断模型独立于主模型，避免污染主对话上下文。

## 11. 全链路降级

**难点**：个人项目在无 API Key / 网络差 / 模型限流时，系统不能崩。

**解决**：每个 LLM 调用点都有 fallback——
- Planner 失败 → 用原查询当唯一子查询
- Critic 失败 → 默认"充分"（不阻塞流程）
- Summarizer 失败 → 直接拼接检索结果
- Reranker 不可用 → 按 fusion_score 排序

**为什么**：降级优先（degrade gracefully）是工程成熟度标志；副作用是**全部 86 个单测不需要 API Key 就能跑**（mock LLM 路径），测试成本几乎为零。

**追问**：降级后答案质量如何保证？
> 答：降级是"可用性兜底"，正常路径仍是完整 LLM 流程；评测数据只统计正常路径的质量。降级保证的是"不崩"，不是"答得好"。

## 12. 评测方法

**难点**：怎么证明"检索/编排真的变好了"而不是自我感觉。

**解决**：10 条带 ground-truth 的 Q&A 数据集 + 双维度评估：
- **检索侧**：Recall@5 消融（纯向量 / 纯 BM25 / 混合 RRF / 完整管线），见 `ablation.py`
- **回答侧**：LLM-as-judge 按标准答案打 1-5 分 + RAGAS（faithfulness/relevancy/context precision）
- **编排侧**：三种 graph 在同一数据集上比质量分 / 延迟 / 轮数（`benchmark_graphs.py`）

**为什么**：
- 消融证明每个环节的增量价值（RRF 加了什么、图谱扩展加了什么）
- LLM-judge 与人类评分相关性已被验证，个人项目用它比人工标注 100 条可行得多
- RAGAS 补上忠实度等维度，防止"答得漂亮但编造"

**追问**：10 条够吗？
> 答：不够，是 MVP 评估。计划扩充到 50+ 条；但 10 条已能暴露检索缺陷（消融中不同策略 recall 差异明显），方向性结论可靠。

## 13. 踩坑 1：异步挂起

**问题**：跑三 graph 评测时，`knowledge_agent` 在**子线程里 `asyncio.run()`** 调 LangGraph 的 async 调用，进程静默挂起 23 分钟零输出。

**根因**：GuardrailsMiddleware 注册的是 async hook（`abefore_agent`），同步 `invoke` 直接抛 `TypeError: No synchronous function provided`；而子线程里自己 `asyncio.run()` 会与 LangGraph/LangSmith 共享的 async 栈冲突，挂死。

**解决**：
1. 单 Agent 改用 `await docs_agent.ainvoke(...)`（async 调用路径）
2. 评测主流程改成**单一 async 循环**（`asyncio.run` 只调一次），三 graph 统一 await
3. 节点全同步的 multi/supervisor 图用 `asyncio.to_thread` 包 `sync invoke`（避免阻塞事件循环）

**教训**：混合 async/sync 调用栈是 Agent 工程最常见的隐形坑；凡是"中间件注册了 async 钩子"的 agent，必须统一走 async 调用路径。

## 14. 踩坑 2：Qdrant 锁冲突

**问题**：评测时 `PermissionError: ./qdrant_data_v2\.lock` —— Qdrant 本地模式对数据目录加**文件锁**，而 `langgraph dev` server 正在运行已持有锁。

**解决**：评测脚本把索引复制到隔离目录 `_bench_qdrant/`，环境变量 `QDRANT_PATH` 指向副本（只读使用），并在 `.gitignore` 中排除该目录。

**为什么**：不杀正在运行的 dev server（那是用户的服务），复制索引成本低（SQLite 文件拷贝），评测与线上隔离。

**教训**：本地嵌入式存储（Qdrant local / SQLite）都有锁语义，多进程并发访问前必须确认归属；工具隔离目录是低成本解法。

## 15. 踩坑 3：LLM API 超时

**问题**：评测连续调用 SiliconFlow API 时，multi/supervisor 图出现**单个请求无限挂起**（HTTP 无超时），外层 240s 超时只是记录 error，底层线程还在空转。

**解决**（已部分落地）：
1. ✅ **graph 级超时已落地**：`router_graph.py` 对子 graph 设 `SINGLE_TIMEOUT=180` / `MULTI_TIMEOUT=240`，超时返回提示语；multi_agent 工具调用设 `DEFAULT_TOOL_TIMEOUT=30`（`fut.result(timeout=...)`）
2. ✅ **评测结果落盘已落地**：`benchmark_graphs.py` 每完成一组查询即把 raw results 写 JSON，断点可续
3. ⏳ **LLM 请求级 timeout 待补**：`_init_model()` 尚未传 `timeout`（`init_chat_model(..., timeout=...)` / httpx timeout），单请求仍可能无限挂起（to_thread 包裹时 `asyncio.wait_for` 取消不了线程）

**教训**：所有外部调用必须显式设超时——"外层任务超时"不等于"底层请求被取消"，尤其同步阻塞调用包在 `to_thread` 里时，`asyncio.wait_for` 取消不了线程。

---

## 面试官最爱的一题总结

**"你这个项目最深的点在哪？"**

> "三个点：一是混合检索管线（向量 + BM25 → RRF → Cross-Encoder 重排 → wikilink 图谱扩展），每个环节都有消融数据支撑；二是多 Agent 编排，我在同一代码库里实现了 workflow 式和 Supervisor 式两种模式并用同一评测集量化对比；三是工程韧性——5 层中间件、全链路降级（无 Key 也能跑）、轮数上限防死循环、86 个单测无需凭据可跑。过程中踩过异步调用栈挂起、Qdrant 文件锁、API 无超时三个真实坑，都沉淀成了排查方法论。"
