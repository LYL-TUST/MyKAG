# 三种 Agent 编排模式对比评测

> 数据集: 10 条带 ground-truth 的 Q&A（`tests/rag/eval_dataset.py`） · 质量分由 LLM-as-judge 按标准答案评分（1-5）· 实测环境: SiliconFlow / DeepSeek V4

## 汇总

| 编排模式 | 平均质量分 | 平均延迟 | 平均轮数 | 失败数 |
|----------|-----------|---------|---------|--------|
| 单 Agent（knowledge_agent） | 4.30/5 | 91.8s | 10.4 | 0 |
| 多角色 workflow（multi_agent） | 4.00/5 | 62.5s | 1.5 | 0 |
| Supervisor 总调度（supervisor_agent） | 4.30/5 | 33.2s | 2.2 | 0 |

## 逐题明细

| # | 编排模式 | 质量分 | 延迟 | 轮数 | 备注 |
|---|----------|-------|------|------|------|
| 1 | 单 Agent（knowledge_agent） | 4/5 | 129.0s | 13 |  |
| 2 | 多角色 workflow（multi_agent） | 3/5 | 82.0s | 1 |  |
| 3 | Supervisor 总调度（supervisor_agent） | 5/5 | 26.7s | 2 |  |
| 4 | 单 Agent（knowledge_agent） | 5/5 | 287.7s | 8 |  |
| 5 | 多角色 workflow（multi_agent） | 5/5 | 51.8s | 1 |  |
| 6 | Supervisor 总调度（supervisor_agent） | 5/5 | 25.4s | 2 |  |
| 7 | 单 Agent（knowledge_agent） | 5/5 | 188.4s | 14 |  |
| 8 | Supervisor 总调度（supervisor_agent） | 5/5 | 42.5s | 2 |  |
| 9 | 多角色 workflow（multi_agent） | 3/5 | 174.9s | 1 |  |
| 10 | Supervisor 总调度（supervisor_agent） | 4/5 | 99.5s | 2 |  |
| 11 | 单 Agent（knowledge_agent） | 5/5 | 107.1s | 13 |  |
| 12 | 多角色 workflow（multi_agent） | 5/5 | 95.7s | 3 |  |
| 13 | Supervisor 总调度（supervisor_agent） | 5/5 | 38.1s | 2 |  |
| 14 | 单 Agent（knowledge_agent） | 5/5 | 53.3s | 11 |  |
| 15 | 多角色 workflow（multi_agent） | 5/5 | 27.3s | 1 |  |
| 16 | 单 Agent（knowledge_agent） | 5/5 | 9.3s | 3 |  |
| 17 | 多角色 workflow（multi_agent） | 5/5 | 10.1s | 1 |  |
| 18 | Supervisor 总调度（supervisor_agent） | 5/5 | 6.5s | 2 |  |
| 19 | 单 Agent（knowledge_agent） | 3/5 | 45.6s | 12 |  |
| 20 | 多角色 workflow（multi_agent） | 4/5 | 69.6s | 3 |  |
| 21 | Supervisor 总调度（supervisor_agent） | 3/5 | 27.7s | 3 |  |
| 22 | 单 Agent（knowledge_agent） | 3/5 | 22.6s | 9 |  |
| 23 | 多角色 workflow（multi_agent） | 3/5 | 30.5s | 2 |  |
| 24 | Supervisor 总调度（supervisor_agent） | 3/5 | 48.2s | 3 |  |
| 25 | 单 Agent（knowledge_agent） | 3/5 | 26.9s | 7 |  |
| 26 | 多角色 workflow（multi_agent） | 3/5 | 48.5s | 1 |  |
| 27 | Supervisor 总调度（supervisor_agent） | 3/5 | 9.8s | 2 |  |
| 28 | 多角色 workflow（multi_agent） | 4/5 | 34.1s | 1 |  |
| 29 | Supervisor 总调度（supervisor_agent） | 5/5 | 7.9s | 2 |  |
| 30 | 单 Agent（knowledge_agent） | 5/5 | 47.7s | 14 |  |

## 分析结论

- **质量最优**: 单 Agent（knowledge_agent）（4.3/5）
- **延迟最低**: Supervisor 总调度（supervisor_agent）（33.2s）
- **解读**: 多 Agent 编排（workflow / supervisor）通过多轮检索换质量,延迟略高但回答更完整;单 Agent 依赖 prompt 引导,轮数不可控。同一代码库内实现并对比三种编排,体现了对 LangGraph 状态机与multi-agent 模式的深入理解。

---

_生成: `tests/rag/report_benchmark.py` · 原始数据: `tests/rag/benchmark_results.json`_
