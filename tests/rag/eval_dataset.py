"""RAGAS evaluation dataset for Personal Knowledge Agent.

Contains ground-truth Q&A pairs based on the Obsidian vault content.
Used with the RAGAS framework to measure:
- Faithfulness (答案是否忠实于检索到的上下文)
- Context Relevancy (上下文是否相关)
- Context Precision (检索结果排序是否精确)
- Answer Correctness (答案是否正确)
"""

# Each test case: (question, expected_answer, relevant_note_names)
EVAL_DATASET = [
    {
        "question": "ellie 项目支持哪些 LLM 后端？",
        "expected_answer": "支持 5 个模型后端：DeepSeek API、OpenAI、Anthropic、Google Gemini、本地 Ollama（qwen2.5:7b）。",
        "relevant_notes": ["ellie 模型后端", "ellie 项目概述"],
    },
    {
        "question": "ellie 的 RAG 模块用了什么技术？为什么选它？",
        "expected_answer": "使用了 TF-IDF 做关键词匹配检索。选择 TF-IDF 是因为零依赖设计目标——不需要引入 embedding 模型和向量数据库。",
        "relevant_notes": ["ellie RAG 模块", "向量检索 vs TF-IDF"],
    },
    {
        "question": "Code Review Agent 有几个 MCP Server？各负责什么？",
        "expected_answer": "有 4 个 Server：rules_engine（规则匹配）、diff_parser（PR diff 解析）、ast_context（AST 调用图上下文扩展）、feedback_loop（反馈学习）。",
        "relevant_notes": ["Code Review Agent 概述", "MCP 协议设计"],
    },
    {
        "question": "AST 调用图分析的压缩率是多少？怎么做到的？",
        "expected_answer": "大型重构达到 98% 压缩率（#5554，800行→16行），小型 bug fix 达到 57%。通过 tree-sitter 构建函数调用图，只提取被修改函数的调用链上的代码上下文。",
        "relevant_notes": ["AST 调用图分析", "四大核心成果"],
    },
    {
        "question": "ellie 的工具注册和 Code Review Agent 的工具注册有什么区别？",
        "expected_answer": "ellie 使用 @tool 装饰器做进程内函数调用注册。Code Review Agent 通过 MCP 协议的 @server.tool() 做跨进程 JSON-RPC 注册。核心区别是：ellie 是单进程简单设计，Code Review Agent 是关注点正交的多进程解耦设计。",
        "relevant_notes": ["ellie 工具系统", "LangChain Tool 注册机制", "MCP 协议设计"],
    },
    {
        "question": "DeepSeek API 怎么接入？用什么 base_url？",
        "expected_answer": "通过 OpenAI-compatible API 接入。可以直接用 `api.deepseek.com`，也可以通过 SiliconFlow 代理（`api.siliconflow.cn/v1`）。使用 openai Python SDK 或 langchain-openai 即可。",
        "relevant_notes": ["DeepSeek API 接入", "ellie 模型后端"],
    },
    {
        "question": "个人知识 Agent 的检索管线是怎么设计的？",
        "expected_answer": "Dense(向量相似度) 和 Sparse(BM25) 混合检索 → RRF 融合 → BGE-Reranker-v2-m3 重排序 → [[wikilink]] 1-hop 上下文扩展。",
        "relevant_notes": ["向量检索 vs TF-IDF", "ellie RAG 模块"],
    },
    {
        "question": "ellie 踩过哪些坑？",
        "expected_answer": "5 个主要踩坑：Tool call 格式不统一、Checkpoint 文件并发写冲突、Streaming 与 Tool call 的交互、TF-IDF 对中文不友好、web_fetch 被反爬。",
        "relevant_notes": ["ellie 踩坑记录"],
    },
    {
        "question": "Code Review Agent 的四大核心成果是什么？",
        "expected_answer": "1. AST 上下文压缩 57-98%；2. 四级工具闸口；3. 多语言调用图分析；4. MCP 协议解耦。验证数据：75 项测试全过，5 种语言支持。",
        "relevant_notes": ["四大核心成果", "Code Review Agent 概述"],
    },
    {
        "question": "为什么 ellie 选择零依赖？新项目为什么不遵循零依赖？",
        "expected_answer": "ellie 选零依赖是为了完全可控 + 面试展示理解底层原理。新项目(个人知识 Agent)用 LangGraph + LlamaIndex 是因为 RAG 管线太复杂不值得自己写，且面试需要展示'会用框架 + 理解框架'两个层次。",
        "relevant_notes": ["零依赖设计决策", "ellie 架构设计"],
    },
]
