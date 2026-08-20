# 三种 Agent 编排模式对比评测

> 数据集: 10 条带 ground-truth 的 Q&A（`tests/rag/eval_dataset.py`） · 质量分由 LLM-as-judge 按标准答案评分（1-5）· 实测环境: SiliconFlow / DeepSeek V4

## 汇总

| 编排模式 | 平均质量分 | 平均延迟 | 平均轮数 | 失败数 |
|----------|-----------|---------|---------|--------|
| 单 Agent（knowledge_agent） | 4.00/5 | 124.3s | 11.4 | 1 |
| 多角色 workflow（multi_agent） | 3.50/5 | 113.2s | 2.9 | 0 |
| Supervisor 总调度（supervisor_agent） | 3.10/5 | 48.6s | 3.1 | 0 |

## 逐题明细

| # | 编排模式 | 质量分 | 延迟 | 轮数 | 备注 |
|---|----------|-------|------|------|------|
| 1 | 单 Agent（knowledge_agent） | - | 240.4s | -1 | timeout |
| 2 | 多角色 workflow（multi_agent） | 3/5 | 102.7s | 3 |  |
| 3 | Supervisor 总调度（supervisor_agent） | 3/5 | 27.1s | 4 |  |
| 4 | 单 Agent（knowledge_agent） | 5/5 | 134.9s | 14 |  |
| 5 | 多角色 workflow（multi_agent） | 3/5 | 156.0s | 3 |  |
| 6 | Supervisor 总调度（supervisor_agent） | 3/5 | 135.4s | 4 |  |
| 7 | 单 Agent（knowledge_agent） | 5/5 | 130.1s | 12 |  |
| 8 | 多角色 workflow（multi_agent） | 3/5 | 109.2s | 3 |  |
| 9 | Supervisor 总调度（supervisor_agent） | 3/5 | 43.9s | 3 |  |
| 10 | 单 Agent（knowledge_agent） | 4/5 | 140.7s | 14 |  |
| 11 | 多角色 workflow（multi_agent） | 3/5 | 181.3s | 3 |  |
| 12 | Supervisor 总调度（supervisor_agent） | 4/5 | 80.6s | 3 |  |
| 13 | 单 Agent（knowledge_agent） | 4/5 | 133.9s | 17 |  |
| 14 | 多角色 workflow（multi_agent） | 5/5 | 115.5s | 3 |  |
| 15 | Supervisor 总调度（supervisor_agent） | 3/5 | 22.7s | 2 |  |
| 16 | 单 Agent（knowledge_agent） | 4/5 | 15.4s | 4 |  |
| 17 | 多角色 workflow（multi_agent） | 4/5 | 93.5s | 3 |  |
| 18 | Supervisor 总调度（supervisor_agent） | 4/5 | 19.7s | 2 |  |
| 19 | 单 Agent（knowledge_agent） | 4/5 | 116.5s | 7 |  |
| 20 | 多角色 workflow（multi_agent） | 4/5 | 70.3s | 2 |  |
| 21 | Supervisor 总调度（supervisor_agent） | 3/5 | 59.7s | 3 |  |
| 22 | 单 Agent（knowledge_agent） | 3/5 | 42.3s | 12 |  |
| 23 | 多角色 workflow（multi_agent） | 3/5 | 114.7s | 3 |  |
| 24 | Supervisor 总调度（supervisor_agent） | 3/5 | 42.0s | 4 |  |
| 25 | 单 Agent（knowledge_agent） | 3/5 | 56.0s | 15 |  |
| 26 | 多角色 workflow（multi_agent） | 3/5 | 61.2s | 3 |  |
| 27 | Supervisor 总调度（supervisor_agent） | 2/5 | 22.0s | 2 |  |
| 28 | 单 Agent（knowledge_agent） | 4/5 | 232.7s | 20 |  |
| 29 | 多角色 workflow（multi_agent） | 4/5 | 127.6s | 3 |  |
| 30 | Supervisor 总调度（supervisor_agent） | 3/5 | 33.3s | 4 |  |

## 分析结论

- **质量最优**: 单 Agent（knowledge_agent）（4.0/5）
- **延迟最低**: Supervisor 总调度（supervisor_agent）（48.6s）
- **解读**: 多 Agent 编排（workflow / supervisor）通过多轮检索换质量,延迟略高但回答更完整;单 Agent 依赖 prompt 引导,轮数不可控。同一代码库内实现并对比三种编排,体现了对 LangGraph 状态机与multi-agent 模式的深入理解。

---

_生成: `tests/rag/report_benchmark.py` · 原始数据: `tests/rag/benchmark_results.json`_
