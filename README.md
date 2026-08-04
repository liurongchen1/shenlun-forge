# ShenlunForge

**面向 LLM Agent 的任务驱动型上下文压缩库**
**Task-Driven Context Compression for LLM Agents**

*一个轻量级 Python 库，在将对话上下文发送给 LLM 之前进行智能压缩，灵感来源于结构化分析方法论。*
*A lightweight Python library that intelligently compresses conversation context before sending it to LLMs, inspired by structured analytical methodology.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 为什么选择 ShenlunForge？ / Why ShenlunForge?

LLM Agent 面临**上下文溢出**问题——随着对话增长，完整历史会超出 token 限制或变得过于昂贵。现有方案（截断、简单摘要、滑动窗口）会丢失关键信息，或在无关内容上浪费 token。
LLM agents suffer from **context overflow** — as conversations grow, the full history exceeds token limits or becomes too expensive. Existing solutions (truncation, simple summarization, sliding windows) lose critical information or waste tokens on irrelevant content.

**ShenlunForge 采用不同的方法**：它像专家分析师处理信息一样对待上下文压缩——首先理解*需要什么*，然后只提取相关事实，将重复内容合并并加上描述性标签，最后在严格的 token 预算内格式化输出。
**ShenlunForge takes a different approach**: it treats context compression like an expert analyst processing information — first understanding *what's needed*, then extracting only relevant facts, merging duplicates with descriptive labels, and formatting everything within a strict token budget.

## 核心创新：ACS 四阶段流水线 / The Innovation: ACS Four-Stage Pipeline

ShenlunForge 将**结构化分析方法论**映射到 LLM 上下文管理，形成四阶段分析压缩策略（Analytical Compression Strategy, ACS）：
ShenlunForge maps a proven structured analytical methodology to LLM context management, forming a four-stage Analytical Compression Strategy (ACS):

| 阶段 / Stage | ShenlunForge 模块 / Module | 功能 / What It Does |
|-----------|-------------------|--------------|
| **分析** (Analyze) | `ACSAnalyzer` | 理解当前任务，确定哪些信息重要 / Understands the current task and determines what information matters |
| **检索** (Retrieve) | `ACSRetriever` | 从完整上下文中查找并评分相关信息 / Finds and scores relevant information from the full context |
| **处理** (Process) | `ACSProcessor` | 合并重复项，提取摘要标签，规范化表达 / Merges duplicates, extracts summary labels, normalizes expressions |
| **格式化** (Format) | `ACSFormatter` | 在 token 预算内输出结构化压缩文本 / Outputs structured, compressed text within token budget |

### 差异化对比 / What Makes This Different?

| 特性 / Feature | ShenlunForge | 简单摘要 / Simple Summary | 滑动窗口 / Sliding Window | RAG |
|---------|:-----------:|:--------------:|:--------------:|:---:|
| 任务感知压缩 / Task-aware compression | ✅ | ❌ | ❌ | ❌ |
| 层级合并 / Hierarchical merging | ✅ | ❌ | ❌ | 部分 / Partial |
| 摘要标签 / Summary labels | ✅ | ❌ | ❌ | ❌ |
| 硬性 token 预算 / Hard token budget | ✅ | ❌ | ✅ | ❌ |
| 表达规范化 / Expression normalization | ✅ | ❌ | ❌ | ❌ |
| 无需额外索引 / No extra retrieval index | ✅ | ✅ | ✅ | ❌ |

## 安装 / Installation

```bash
# 基础安装（包含 tiktoken 用于精确 token 计数）
# Basic installation (includes tiktoken for accurate token counting)
pip install shenlun-forge

# 含 OpenAI 兼容 API 支持（推荐用于 LLM 驱动策略）
# With OpenAI-compatible API support (recommended for LLM-powered strategies)
pip install shenlun-forge[openai]

# 含 LangChain 适配器
# With LangChain adapter
pip install shenlun-forge[langchain]

# 完整安装（所有可选依赖）
# Full installation (all optional dependencies)
pip install shenlun-forge[full]

# 开发安装
# Development installation
pip install shenlun-forge[dev]
```

## 快速开始 / Quick Start

### Python API

```python
import asyncio
from shenlun_forge import ShenlunForge

async def main():
    # 初始化 LLM 提供者 / Initialize with your LLM provider
    forge = ShenlunForge(
        model="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com/v1",
    )

    # 对话历史 / Conversation history
    messages = [
        {"role": "user", "content": "I'm building a restaurant app with FastAPI."},
        {"role": "assistant", "content": "Great! Let's use FastAPI with SQLAlchemy..."},
        # ... 更多消息 / more messages ...
    ]

    # 为当前查询压缩上下文 / Compress context for the current query
    result = await forge.compress(
        messages=messages,
        current_query="What database did we decide on?",
        max_tokens=4000,
    )

    print(result.compressed_text)
    print(f"压缩率 / Compression ratio: {result.compression_ratio:.1%}")
    print(f"节省 token / Token savings: {result.original_tokens - result.compressed_tokens} tokens")

asyncio.run(main())
```

### 选择策略 / Choosing a Strategy

```python
from shenlun_forge import ShenlunForge

# ACS 全量分析流水线（需要 LLM，最佳质量）
# Full ACS pipeline (requires LLM, best quality)
forge = ShenlunForge.create(strategy="acs", model="deepseek-chat", api_key="...")

# 简单策略（无需 LLM，基于规则，最快速）
# Simple strategy (no LLM needed, rule-based, fastest)
forge = ShenlunForge.create(strategy="simple")

# HCS 混合策略（ACS + 逐字代码块，最适合编码任务）
# HCS hybrid strategy (ACS + verbatim code blocks, best for coding tasks)
forge = ShenlunForge.create(strategy="hcs", model="deepseek-chat", api_key="...")
```

### CLI 用法 / CLI Usage

```bash
# 压缩 JSON 消息文件 / Compress a JSON messages file
shenlun-forge compress --messages messages.json \
    --query "What did we decide about the database?" \
    --strategy acs \
    --max-tokens 4000 \
    --api-key "sk-..." \
    --base-url "https://api.deepseek.com/v1"

# 从 stdin 读取消息 / Read messages from stdin
echo '[{"role":"user","content":"Hello"}]' | \
    shenlun-forge compress --stdin --strategy simple --query "Summary"

# 使用环境变量配置凭据 / Use environment variables for credentials
export SHENLUN_FORGE_API_KEY="sk-..."
export SHENLUN_FORGE_BASE_URL="https://api.deepseek.com/v1"
export SHENLUN_FORGE_MODEL="deepseek-chat"
shenlun-forge compress --messages chat.json --query "Recap" --verbose

# 保存输出到文件 / Save output to file
shenlun-forge compress -m messages.json -q "Summary" -o result.json

# 列出可用策略 / List available strategies
shenlun-forge strategies

# 显示版本 / Show version
shenlun-forge version
```

**CLI 输出格式 / CLI output format：**
```json
{
  "compressed_text": "...",
  "original_tokens": 12500,
  "compressed_tokens": 3200,
  "compression_ratio": 0.256,
  "stats": {
    "strategy": "acs",
    "total_duration_ms": 2340,
    "original_messages": 30,
    "relevant_blocks": 18,
    "compressed_groups": 8
  }
}
```

### LangChain 集成 / With LangChain

```python
from shenlun_forge.adapters.langchain_adapter import LangChainShenlunForge

forge = LangChainShenlunForge(
    model="deepseek-chat",
    api_key="your-key",
    max_tokens=4000,
)

# 作为可调用组件集成到流水线中 / Use as a callable in your pipeline
result = forge.invoke({
    "messages": chat_history,
    "query": current_question,
})

# 或作为回调处理器自动压缩 / Or as a callback handler for automatic compression
handler = forge.as_callback_handler()
```

## 架构 / Architecture

```
shenlun_forge/
├── __init__.py          # 公共 API 导出 / Public API exports
├── types.py             # 数据模型 / Data models (ContextBlock, TaskProfile, InfoType, etc.)
├── base.py              # 抽象基类 / Abstract base classes
├── llm.py               # LLM 提供者 / LLM provider (OpenAI-compatible)
├── analyzer.py          # 阶段 1: 任务分析 / Step 1: Task analysis (ACSAnalyzer)
├── retriever.py         # 阶段 2: 信息提取 / Step 2: Information extraction (ACSRetriever)
├── processor.py         # 阶段 3: 合并与规范化 / Step 3: Merge & normalize (ACSProcessor)
├── formatter.py         # 阶段 4: Token 预算格式化 / Step 4: Token-budget formatting (ACSFormatter)
├── compressor.py        # 主编排器 / Main orchestrator (ShenlunForge class)
├── registry.py          # 策略注册表 / Strategy registry for extensibility
├── cli.py               # 命令行接口 / Command-line interface
├── strategies/
│   ├── __init__.py
│   ├── simple.py        # 规则回退（无需 LLM）/ Rule-based fallback (no LLM)
│   └── hybrid.py        # ACS + 逐字代码增强 / ACS + verbatim code augmentation (HCSCompressor)
└── adapters/
    ├── __init__.py
    └── langchain_adapter.py  # LangChain 集成 / LangChain integration
```

## 工作原理 / How It Works

### 阶段 1：分析 — 任务解析 / Step 1: Analyze — Task Analysis

ACSAnalyzer 检查当前查询和对话，确定以下信息：
The ACSAnalyzer examines the current query and conversation to determine:

- **任务类型 / Task type**：QA、代码生成、分析、创意写作等 / QA, code generation, analysis, creative, etc.
- **关键要素 / Key elements**：需要哪些具体信息 / What specific information is needed
- **优先级权重 / Priority weights**：每类信息的重要程度 / How important each type of information is
- **输出格式 / Output format**：顺序、结构化或要点列表 / Sequential, structured, or bullet points

当 LLM 不可用时，快速规则分类器使用关键词匹配作为回退。
When LLM is unavailable, a fast rule-based classifier uses keyword matching as fallback.

### 阶段 2：检索 — 信息提取 / Step 2: Retrieve — Information Extraction

ACSRetriever 基于 6 个维度对每条消息块评分：
The ACSRetriever scores every message block based on 6 dimensions:

1. **关键词相关性 / Keyword relevance**：与当前查询的匹配度 / Match to current query
2. **信息类型 / Information type**：事实、动作、问题、数据等 / Facts, actions, problems, data, etc.
3. **时效性 / Recency**：较新消息通常更相关 / Newer messages generally more relevant
4. **角色重要性 / Role importance**：工具结果 > 用户指令 > 助手解释 / Tool results > user instructions > assistant explanations
5. **信号词 / Signal words**："but"、"therefore"、"important" 等
6. **内容密度 / Content density**：代码、表格、结构化数据得分更高 / Code, tables, structured data score higher

### 阶段 3：处理 — 核心创新 / Step 3: Process — Core Innovation

ACSProcessor 执行三个关键操作：
The ACSProcessor performs three key operations:

1. **合并同类项 / Merge duplicates**：语义相似的块被分组 / Semantically similar blocks are grouped
2. **提炼摘要标签 / Extract labels**：每组获得一个简洁的 2-6 字符标签 / Each group gets a concise 2-6 character label
3. **规范化表达 / Normalize**：非正式语言转为技术表达 / Informal language is converted to technical expressions

### 阶段 4：格式化 — 输出 / Step 4: Format — Formatted Output

ACSFormatter 完成以下工作：
The ACSFormatter:

- 按相关性和信息类型排列块优先级 / Prioritizes blocks by relevance and information type
- 在 token 预算内贪心填充最高优先级内容 / Greedily fills the token budget with highest-priority content
- 按任务配置格式化输出（顺序、结构化或要点）/ Formats output according to the task profile
- 预算紧张时优雅截断 / Truncates gracefully when budget is tight

## 基准测试结果 / Benchmark Results

### 单场景对比（5 种策略）/ Single-Scenario Comparison (5 Strategies)

| 策略 / Strategy | 事实保留 / Facts Retained | 压缩率 / Compression Ratio | 说明 / Notes |
|----------|:---:|:---:|-------|
| **Full ACS** | 100% | 58% | 最佳事实保留，强压缩 / Best fact retention with strong compression |
| HCS | 100% | 65% | 混合内容良好平衡 / Good balance for mixed content |
| Direct LLM | 80% | 45% | 简单摘要，丢失细节 / Simple summarization, loses details |
| Sliding Window | 70% | 75% | 完全丢失早期上下文 / Loses early context entirely |
| Simple (no LLM) | 60% | 50% | 快速但质量较低 / Fast but low quality |

### 6 场景批量实验 / 6-Scenario Batch Experiment

| 场景 / Scenario | Full ACS | HCS | Simple | Sliding Window |
|----------|:---:|:---:|:---:|:---:|
| 安全分析 / Security Analysis | **10/10** | 2/10 | 0/10 | 0/10 |
| 被拒方案 / Rejected Alternatives | **8/8** | 7/8 | 2/8 | 2/8 |
| 代码细节 / Code Detail | 1/10 | 1/10 | 1/10 | 1/10 |
| 事实回忆 / Fact Recall | **16/16** | 15/16 | 12/16 | 12/16 |
| 架构设计 / Architecture | **13/15** | 13/15 | 11/15 | 11/15 |
| 任务/时间线 / Tasks/Timeline | 7/10 | 2/10 | **10/10** | **10/10** |

### 消融研究（5 个变体 × 6 场景）/ Ablation Study (5 Variants × 6 Scenarios)

| 变体 / Variant | 说明 / Description | 平均得分 / Avg Score |
|---------|-------------|:---------:|
| **Full ACS** | 完整四阶段流水线 / Complete 4-step pipeline | **最高 / Highest overall** |
| HCS | ACS + 逐字代码 / ACS + verbatim code | 代码密集任务最佳 / Best for code-heavy tasks |
| Direct-Only | 仅 LLM 摘要 / LLM summarization only | 中等表现 / Moderate performance |
| Analyze-Only | 分析+检索（跳过合并/格式化）/ Analyze + Extract (skip merge/format) | 事实回忆良好 / Good for factual recall |
| Simple | 纯规则驱动 / Pure rule-based | 最快速，质量最低 / Fastest, lowest quality |

### 关键发现 / Key Findings

- **Full ACS** 在推理和事实回忆任务中占优（6 个场景中 4 个最佳）/ dominates in reasoning and fact recall tasks (4/6 scenarios best)
- **HCS** 在需要精确保留代码块时表现出色 / excels when code blocks need exact preservation
- **Simple** 在时间线/顺序任务中出人意料地有效（无 LLM 成本！）/ surprisingly effective for timeline/sequential tasks (no LLM cost!)
- 代码细节保留是所有策略的共同挑战 / Code detail retention is a universal challenge across all strategies

## 配置 / Configuration

### LLM 提供者 / LLM Provider

```python
from shenlun_forge import OpenAIProvider

# 兼容任何 OpenAI 兼容 API / Works with any OpenAI-compatible API
provider = OpenAIProvider(
    model="deepseek-chat",        # 或 / or "gpt-4o", "claude-3-haiku", etc.
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    temperature=0.1,               # 低温度保证分析一致性 / Low temperature for consistent analysis
    max_tokens=4096,
)
```

### 自定义组件 / Custom Components

```python
from shenlun_forge import ShenlunForge
from shenlun_forge.analyzer import ACSAnalyzer
from shenlun_forge.retriever import ACSRetriever

forge = ShenlunForge(
    analyzer=ACSAnalyzer(llm=provider, use_fast_path=True),
    retriever=ACSRetriever(llm=provider, min_score_threshold=0.2),
    # 使用默认处理器和格式化器 / Use default processor and formatter
)
```

### 任务类型 / Task Types

```python
from shenlun_forge import TaskType

# 强制指定任务类型（跳过自动检测）
# Force a specific task type (skip auto-detection)
result = await forge.compress(
    messages=messages,
    current_query="Fix this bug",
    task_type="code_gen",  # 强制代码聚焦压缩 / Forces code-focused compression
    max_tokens=4000,
)
```

### 添加自定义策略 / Adding Custom Strategies

扩展策略注册表以添加自定义压缩策略：
Extend the strategy registry to add your own compression strategies:

```python
# 在代码中或通过猴子补丁 / In your code or via monkey-patching
from shenlun_forge.registry import get_strategy_class

# 注册表是懒加载的，可以在导入时添加条目
# The registry is lazy-loaded, so you can add entries at import time
```

## 依赖要求 / Requirements

- Python 3.9+
- `pydantic>=2.0` — 数据验证和模型 / Data validation and models
- `tiktoken>=0.5.0` — 精确 token 计数 / Accurate token counting
- 可选 / Optional: `openai>=1.0` — OpenAI 兼容 LLM 提供者 / OpenAI-compatible LLM provider
- 可选 / Optional: `langchain-core>=0.1` — LangChain 适配器集成 / LangChain adapter integration

## 研究背景 / Research Background

本方法受结构化分析方法论启发，该方法论要求：
This approach is inspired by structured analytical methodology, where analysts must:

1. 理解问题要求 / Understand the question requirements
2. 从冗长材料中提取相关事实 / Extract relevant facts from lengthy materials
3. 合并重复项并添加摘要标签 / Merge duplicates and add summary labels
4. 在严格的字数限制内呈现 / Present within strict word limits

我们对 21 个代表性上下文压缩系统（2023-2026）的学术调研发现，**现有系统均不同时具备**以下四个特征：
Our academic survey of 21 representative context compression systems (2023-2026) found that **no existing system** combines all four of these features:

- 任务驱动压缩策略 / Task-driven compression strategy
- 语义去重的层级合并 / Hierarchical merging with semantic deduplication
- 显式摘要标签提取 / Explicit summary label extraction
- 硬性 token 预算执行 / Hard token budget enforcement

最接近的工作包括 GraphRAG（层级但基于图）、SUPO（任务驱动但仅 RL）和 HTSIR（层级但基于文档结构）。
The closest works are GraphRAG (hierarchical but graph-based), SUPO (task-driven but RL-only), and HTSIR (hierarchical but document-structure-based).

## 许可证 / License

MIT 许可证 — 详见 LICENSE 文件 / MIT License — see LICENSE file for details.

## 引用 / Citation

```bibtex
@software{shenlunforge2026,
  title={ShenlunForge: Task-Driven Context Compression for LLM Agents},
  author={Liu, Rongchen},
  year={2026},
  url={https://github.com/liurongchen1/shenlun-forge}
}
```

## 贡献 / Contributing

欢迎贡献！我们特别需要以下方面的帮助：
Contributions welcome! Areas we'd love help with:

- 更多框架适配器 / More framework adapters (AutoGen, CrewAI, LlamaIndex)
- 额外压缩策略 / Additional compression strategies
- 上下文压缩评估基准数据集 / Benchmark datasets for context compression evaluation
- 多语言表达规范化映射 / Multilingual expression normalization maps
