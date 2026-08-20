# 简历项目写法 — Personal Knowledge Agent

> 面向 AI 应用 / 后端 / 全栈岗位的三种写法模板 + 量化要点 + 追问防御。
> 原则：**不堆技术名词，每个 bullet 都要有"设计决策 + 量化结果"**。

---

## 一、一句话定位（放简历顶部 / 项目名旁）

> **Agent 化个人知识管理系统**：基于 LangGraph + LlamaIndex 的多 Agent 编排系统，连接 Obsidian 笔记库实现混合检索与知识图谱推理，三种编排模式（单 Agent / workflow 多角色 / Supervisor 总调度）同库并存、量化对比。

---

## 二、STAR 项目经历模板（可直接粘）

### 版本 A：AI 应用 / Agent 方向（推荐主力版）

**项目名**：Agent 化个人知识管理系统（Personal Knowledge Agent）
**技术栈**：Python · LangGraph · LlamaIndex · Qdrant · scikit-learn · jieba · MCP · Next.js 16

**项目简介**：面向个人知识库的 RAG Agent，支持混合检索、[[wikilink]] 知识图谱推理与多 Agent 编排，通过 MCP 协议供 Claude Desktop / Cursor 等外部 Agent 调用。

**核心工作（bullet，选 3-4 条）**：

1. **设计并实现混合检索管线**：向量（BGE-M3 embedding + Qdrant）与 BM25（jieba 分词）双路检索，经 RRF 融合后由 BGE-Reranker 精排，再沿 [[wikilink]] 做 BFS 图谱扩展；通过消融实验验证各环节增量价值（纯向量 → 混合 RRF 召回显著提升）。

2. **实现三种 Agent 编排模式并量化对比**：单 Agent（5 层中间件）→ workflow 式多角色（Planner-Executor-Summarizer-Critic，含"不满意即重检索"闭环）→ Supervisor 总调度（动态路由 + 全局历史 + 轮数上限）；同一评测集上对比质量分 / 延迟 / 轮数，用数据决策架构。

3. **构建全链路降级与工程韧性**：无 API Key 也可运行（Planner 回退原查询、Critic 默认充分、Summarizer 拼接结果）；86 个单元测试全部免凭据可跑；轮数上限（2/6）防死循环；watchdog 增量索引免全量重建。

4. **以 MCP 协议暴露 7 个检索工具**：外部 Agent（Claude Desktop / Cursor）经 stdio 即插即用，与进程内工具共用同一实现，零重复代码。

### 版本 B：后端 / 工程方向（淡化 AI 概念，强化系统设计）

1. **构建 7 模块 RAG 服务**：ingestion → chunking → Qdrant 向量索引 → 混合检索 → 重排 → 图谱扩展 → watchdog 增量更新；FastAPI + LangGraph 双入口，CORS/鉴权/缓存完备。

2. **设计可观测与可降级体系**：LangSmith trace、5 层中间件（Guardrails / 上下文压缩 / 工具重试 / 模型重试 / 模型降级链）、全链路 fallback，保障 API Key 缺失时服务不崩。

3. **实现多 Agent 状态机**：基于 LangGraph StateGraph 实现 workflow 式与 Supervisor 式两种编排，含防死循环上限、非法决策降级、消息级历史累积。

### 版本 C：全栈方向（补上前端）

1. **Next.js 16 前端**：对话流 + 知识图谱可视化（react-force-graph-2d）+ 笔记浏览 + 配置热切换（后端 `POST /config` 免重启换 vault）。

---

## 三、量化数据速查（写进 bullet 的数字）

| 数据 | 值 | 用法 |
|---|---|---|
| 单元测试 | 86 个 | "86 个单测全通过、免 API Key 可跑" |
| 评测集 | 10 条 ground-truth Q&A | "10 条带标准答案的评测集" |
| LangGraph graph | 3 种编排 | "三种编排模式同库对比" |
| 中间件 | 5 层 | "5 层中间件链" |
| 工具 | 7 个 / MCP 暴露 7 个 | "7 工具全量 MCP 暴露" |
| 检索候选 | top-15 → 重排 → top-5 | "15 候选精排至 5" |
| 检索轮数上限 | 2（闭环）/ 6（Supervisor） | "防死循环上限" |
| 代码量 | ~5.8k 行 Python（src）+ ~2k 行测试 | 可写可不写 |

> ⚠️ 数据规模（22 篇笔记）**不要主动写进 bullet**，面试问到时再解释（见下）。

---

## 四、面试官常见追问防御

**Q：数据才 22 篇笔记，检索效果有说服力吗？**
> "数据规模是个人项目的客观限制，但这个项目的重点是可扩展的检索管线和编排架构——数据量翻 100 倍不需要改架构，只需要换 Qdrant 集群。而且我有消融实验，每个环节的增量价值是测出来的，不是感觉出来的。"

**Q：哪些代码是你写的，哪些复用的？**
> "我复用了上一项目（健康知识 Agent）的模型注册表、中间件和鉴权模块——**架构复用本身就是设计决策**。新写的是：RAG 管线 7 模块、wikilink 图谱、7 个检索工具、三种编排 graph、评测体系。复用的部分我能讲清楚每个模块的职责。"

**Q：为什么不用现成的 RAG 框架（如 Dify / RAGFlow）？**
> "框架能做 demo，但我要的是对管线的完全掌控和透明中间件体系——这也是面试展示'会用框架 + 理解原理'两个层次的地方。用 LangGraph/LlamaIndex 组装，每一层都可观测、可替换、可量化。"

**Q：三种编排模式，哪个最好？**
> "没有绝对最好，取决于场景：workflow 式延迟低、可控，适合流程固定的问答；Supervisor 式灵活、能临场换策略，适合任务边界不固定。所以我把两个都实现了并用数据对比——这本身就是工程设计里'用数据决策'的体现。"

**Q：RAG 幻觉怎么处理？**
> "三层：一是检索侧双通道 Critic（规则判空 + LLM judge 判信息缺口），不满足就重检索而不是硬答；二是 Summarizer prompt 强制标注来源、无据不答；三是评测里 RAGAS faithfulness 指标专门监控幻觉，超标就回溯是检索问题还是生成问题。"

---

## 五、不同岗位微调建议

| 岗位 | 突出点 | 弱化点 |
|---|---|---|
| AI 应用 / Agent 工程师 | 多 Agent 编排、混合检索、LLM-judge 评测、MCP | 前端、FastAPI 细节 |
| 后端工程师 | 状态机设计、降级/重试/超时、Qdrant 存储、watchdog 增量 | 模型细节、prompt 设计 |
| 全栈 / 前端 | Next.js 前端 + 图谱可视化 + API 对接 | 中间件内部实现 |

---

## 六、简历写法红线

1. ❌ 不要写"个人知识库"当项目名 → ✅ "Agent 化知识管理系统"
2. ❌ 不要写"我用了 LangGraph/LlamaIndex" → ✅ "设计并实现……"（动词 + 结果）
3. ❌ 不要写"22 篇笔记" → ✅ "10 条 ground-truth 评测 + 消融实验"
4. ❌ 不要堆 8 个技术名词 → ✅ 3-4 条 bullet，每条一个设计决策
5. ✅ 每条 bullet 能展开 2 分钟深度追问，讲不清的就不写
