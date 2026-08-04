# ShenlunForge v2: 申论+策论 双引擎技术方案

## 一、框架升级概览

### 从单层到双层
```
v1 (申论语境):  审题 → 找点 → 加工 → 书写 → 输出压缩上下文
v2 (双引擎):    审题 → 找点 → 加工 → 书写 → 【策论引擎】→ 输出压缩上下文 + 策略建议
```

### 哲学基础
| 层次 | 传统方法论 | ShenlunForge映射 | 核心能力 |
|---|---|---|---|
| 第一层 | **申论** | Analyzer→Retriever→Processor→Formatter | 信息提取与重组 |
| 第二层 | **策论** | Strategist（审势→权衡→定策→施策） | 问题分析与策略生成 |

---

## 二、策论引擎详细设计

### 2.1 新增组件：Strategist

```
┌─────────────────────────────────────────────────────┐
│                  Strategist 策论引擎                  │
├─────────────┬─────────────┬─────────────┬───────────┤
│   审势      │   权衡      │   定策      │   施策    │
│  Situation  │  Trade-off  │  Strategy   │  Action   │
│  Analysis   │  Analysis   │  Generation │  Planning │
└─────────────┴─────────────┴─────────────┴───────────┘
```

### 2.2 四步详解

#### Step 5: 审势 (Situation Analysis)
**目标**：从压缩后的上下文中识别态势、问题、矛盾

**策论映射**：渠道2反推法
| 反推路径 | 操作 | LLM Prompt指令 |
|---|---|---|
| 问题→态势 | 从对话中提取显式/隐式问题 | "识别当前项目/任务中存在的问题和障碍" |
| 原因→根因 | 分析问题产生的原因 | "分析这些问题/障碍的根本原因" |
| 经验→优势 | 提取已有的成功做法 | "识别对话中已验证可行的方案/经验" |
| 教训→风险 | 提取失败或受阻的做法 | "识别对话中被证明无效或有风险的方案" |

**输入**：Formatter输出的压缩上下文 + 原始Query
**输出**：`SituationReport`
```python
class SituationReport:
    problems: list[Problem]        # 识别出的问题
    root_causes: list[Cause]       # 根因分析
    strengths: list[Strength]      # 已有优势/经验
    risks: list[Risk]              # 已知风险/教训
    overall_assessment: str        # 态势总体判断
```

#### Step 6: 权衡 (Trade-off Analysis)
**目标**：对可能的解决方案进行多维度评估

**策论映射**：四性自查清单
| 策论维度 | ShenlunForge实现 | 评估内容 |
|---|---|---|
| 针对性 | Relevance Score | 方案是否精准回应query的核心诉求？ |
| 可行性 | Feasibility Score | 方案是否受限于上下文中的实际约束（预算、技术、时间）？ |
| 操作性 | Actionability Score | 方案是否具体到可执行？谁做？怎么做？ |
| 系统性 | Coverage Score | 方案是否覆盖多维度？还是只解决单点？ |

**策论映射**：五大分类维度作为评估框架
1. 思想观念维度 — 认知是否到位？
2. 制度规则维度 — 流程/规范是否完善？
3. 资源保障维度 — 人/财/物是否充足？
4. 监管执行维度 — 执行/监督是否到位？
5. 技术支撑维度 — 工具/技术是否匹配？

**输出**：`TradeoffReport`
```python
class TradeoffReport:
    candidate_strategies: list[Strategy]
    # 每个Strategy包含:
    # - description: str
    # - dimension: str  (属于哪个维度)
    # - relevance: float (针对性评分)
    # - feasibility: float (可行性评分)
    # - actionability: float (操作性评分)
    # - coverage: float (系统性评分)
    # - dependencies: list[str]
    # - risks: list[str]
```

#### Step 7: 定策 (Strategy Generation)
**目标**：生成分层策略建议（上策/中策/下策）

**策论映射**：对策升维
| 升维方向 | 低阶策略 | 高阶策略 |
|---|---|---|
| 思维升维 | 头痛医头 | 系统治理 |
| 主体多元 | 单一主体 | 多方协同 |
| 工具数智 | 传统手段 | 数字化/AI赋能 |

**策论映射**：渠道3六大维度万能库（作为LLM的prior knowledge补充）
```python
STRATEGY_DIMENSIONS = {
    "cognitive": "思想观念 — 提高认知、宣传教育、树立典型",
    "institutional": "制度规则 — 完善制度、细化条例、建立机制",
    "resource": "资源保障 — 资金投入、人才培养、技术引入",
    "supervision": "监管执行 — 日常监管、专项整治、绩效考核",
    "technical": "技术支撑 — 平台搭建、数据打通、智能分析",
    "collaborative": "社会协同 — 多方联动、第三方评估、公众参与"
}
```

**输出**：`StrategicPlan`
```python
class StrategicPlan:
    recommended: Strategy      # 上策：最优推荐
    alternatives: list[Strategy]  # 中策/下策：备选方案
    rationale: str              # 推荐理由
    risk_mitigation: list[str]  # 风险缓释措施
```

#### Step 8: 施策 (Action Planning)
**目标**：将策略转化为具体可执行的步骤

**策论映射**：标准句式 `主体 + 动作 + 对象 + 结果`

**策论映射**：四性验证（最终质检）
- 针对性检查：每个action是否对应一个具体问题？
- 可行性检查：每个action是否在上下文约束内可实现？
- 操作性检查：每个action是否有明确的主体和步骤？
- 系统性检查：所有action是否覆盖多个维度？

**输出**：`ActionPlan`
```python
class ActionPlan:
    actions: list[Action]
    # 每个Action包含:
    # - actor: str        (谁来做)
    # - action: str       (做什么)
    # - target: str       (对什么)
    # - expected_result: str (达到什么效果)
    # - timeline: str     (时间要求)
    # - dimension: str    (属于哪个维度)
    
    verification: VerificationResult
    # 四性验证结果
```

---

## 三、策论三渠道 → ShenlunForge 信息提取策略

### 渠道1：直接提取（已有能力）
```
对话中显式给出的答案/方案/决策
→ Retriever直接匹配提取
→ 无需LLM推理
```

### 渠道2：反推推理（LLM核心价值）
```
对话中的问题、原因、经验、教训
→ Strategist审势阶段进行反推
→ 问题→对策 / 原因→对策 / 经验→对策 / 教训→对策
→ 这是Shenlun策略碾压Simple/Sliding的核心原因
```

### 渠道3：维度补充（LLM先验知识）
```
六大维度万能库作为prior knowledge
→ Strategist定策阶段引导LLM从多维度思考
→ 确保策略的系统性和全面性
```

---

## 四、完整Pipeline（v2）

```
输入: messages[] + query + token_budget
         │
    ┌────▼────┐
    │ 审题    │ Analyzer: 理解query意图 + 评估复杂度
    │ (Analyze)│ 输出: QueryAnalysis (意图/复杂度/范围/约束)
    └────┬────┘
         │
    ┌────▼────┐
    │ 找点    │ Retriever: 从messages中定位相关信息
    │ (Retrieve)│ 渠道1: 直接提取显式信息
    └────┬────┘  输出: RelevantFacts[]
         │
    ┌────▼────┐
    │ 加工    │ Processor: 结构化重组 + 去重 + 排序
    │ (Process) │ 输出: StructuredContext
    └────┬────┘
         │
    ┌────▼────┐
    │ 书写    │ Formatter: 在token budget内生成压缩输出
    │ (Format) │ 输出: CompressedContext (token-constrained)
    └────┬────┘
         │
    ┌────▼────────────────────────────────────┐
    │         策论引擎 Strategist              │
    ├──────────┬──────────┬──────────┬────────┤
    │ 审势     │ 权衡     │ 定策     │ 施策   │
    │ 识别态势 │ 多维评估 │ 分层策略 │ 执行计划│
    │ 问题/原因│ 四性打分 │ 上中下策 │ 主体+  │
    │ 优势/风险│ 五维框架 │ 维度覆盖 │ 动作+  │
    │ (渠道2)  │ (渠道3)  │ 升维思考 │ 对象+  │
    │          │          │          │ 结果   │
    └────┬────┴────┬─────┴────┬─────┴────┬───┘
         │
    ┌────▼────┐
    │ 最终输出 │ CompressedContext + StrategicInsights
    └─────────┘
```

---

## 五、与实验结果的对应

### 为什么Shenlun在复杂任务上碾压？
因为Shenlun天然具备"渠道2反推"能力：

| 实验场景 | 需要的策论能力 | Shenlun表现 | Simple表现 |
|---|---|---|---|
| 安全推理 | 跨消息反推：从分散的安全讨论中推理出完整安全体系 | 80% ✅ | 0% ❌ |
| 被否决方案 | 反推：从"我们不用X"反推出完整的方案对比 | 100% ✅ | 25% ❌ |
| 代码细节 | 反推：从代码片段反推设计意图和实现逻辑 | 80% ✅ | 10% ❌ |
| 事实回忆 | 直接提取为主 | 100% ✅ | 75% |
| 架构概览 | 直接提取+部分反推 | 87% ✅ | 73% |
| 待办时间线 | 纯直接提取（显式结构化信息） | 70% | 100% ✅ |

### 关键发现
**当任务需要"反推"时，Simple/Sliding完全失效，只有LLM策略能胜任。**
这验证了策论"渠道2"的理论价值。

---

## 六、待讨论

1. **Strategist是否需要独立LLM调用？**
   - 方案A：与Formatter共用一次调用（在prompt中同时要求输出压缩上下文+策略建议）
   - 方案B：独立调用（更精细，但增加延迟）
   
2. **策论引擎的触发条件？**
   - 总是启用？还是只在query复杂度超过阈值时启用？
   - 实验数据显示：简单结构化查询（待办/时间线）Simple反而更好
   
3. **论文叙事：**
   - "基于中国传统问题解决方法论的LLM上下文压缩与认知增强框架"
   - 申论解决"信息过载"问题，策论解决"决策支持"问题
   - 双层设计：先压缩、再增强
