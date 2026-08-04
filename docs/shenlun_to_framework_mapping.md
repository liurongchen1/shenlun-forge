# 申论方法论 → shenlun-forge 框架优化完整映射

> 整理自备考资料中的所有申论写作方法论，逐条映射到框架各组件的改进点。
> 目标：先理清楚，再统一落地。

---

## 总览：申论方法论 vs 框架管线

```
申论写作流程：  审题 → 找要点 → 归类 → 排序 → 按格式输出
框架管线：    Analyzer → Retriever → Processor → Formatter
              (Step 1)    (Step 2)     (Step 3)     (Step 4)
```

---

## 一、Analyzer（审题器）改进点

### 1.1 已有 ✅
- `TaskType` 分类：QA/SUMMARY/CODE_GEN/ANALYSIS/CREATIVE/EXTRACTION/TOOL_USE/CONVERSATION/AUTO
- 通过 LLM 分析 `current_query` 判断任务类型
- 生成 `TaskProfile`（token_budget、key_elements、output_format、priority_rules）

### 1.2 缺失 ❌ — 需要新增

#### (A) 分类维度字段 `classification_dimension`

**来源**：申论九大分类法 + 信号二"看题干关键词选分类法"

**问题**：当前 Analyzer 知道"用户想分析"，但不知道"按什么维度组织信息"。

**改动**：`TaskProfile` 新增字段：
```python
@dataclass
class TaskProfile:
    ...
    # 新增：信息组织维度
    # 可选值：subject(主体) / process(流程) / domain(领域) / 
    #         element(要素) / objective(主客观) / level(层级) / 
    #         manifestation(表现形态) / stakeholder(利益相关方) / layer(结构层次)
    classification_dimension: str = "auto"
```

**触发逻辑（信号二）**：
```python
# 根据 query 关键词自动推断分类维度
DIMENSION_KEYWORDS = {
    "subject":       ["谁", "哪方", "政府", "企业", "个人", "部门"],
    "process":       ["流程", "步骤", "怎么", "如何", "阶段", "先后"],
    "domain":        ["领域", "技术", "管理", "资源", "制度", "资金"],
    "element":       ["组件", "模块", "部分", "组成", "要素"],
    "objective":     ["主观", "客观", "人为", "自然"],
    "level":         ["个人", "组织", "社会", "国家", "宏观", "微观"],
    "manifestation": ["共性问题", "个性问题", "普遍", "个别"],
    "stakeholder":   ["态度", "立场", "支持", "反对", "赞成"],
    "layer":         ["表层", "深层", "根本", "制度", "根源"],
}
```

#### (B) 输出逻辑结构字段 `output_logic`

**来源**：申论三种逻辑结构（并列式/因果式/递进式）

**改动**：`TaskProfile` 新增字段：
```python
@dataclass
class TaskProfile:
    ...
    # 新增：输出逻辑结构
    # parallel（并列）/ causal（因果：原因→对策）/ progressive（递进：表→中→深）
    output_logic: str = "auto"  # auto = 根据 classification_dimension 自动推断
```

**自动推断规则**：
- `classification_dimension = "subject"` → `output_logic = "parallel"`（并列式）
- `classification_dimension = "process"` → `output_logic = "progressive"`（递进式）
- `classification_dimension = "layer"` → `output_logic = "progressive"`（递进式）
- `classification_dimension = "domain"` → `output_logic = "causal"`（因果式）

#### (C) TaskType 与 output_format 的细化映射

**来源**：申论各题型答题结构

| 申论题型 | TaskType | output_format | output_logic |
|----------|----------|---------------|--------------|
| 归纳概括 | EXTRACTION | bullet | parallel |
| 综合分析 | ANALYSIS | structured | causal |
| 提出对策 | ANALYSIS | structured | causal |
| 贯彻执行(应用文) | TOOL_USE | structured | parallel |
| 大作文 | CREATIVE | structured | progressive |

---

## 二、Retriever（信息提取器）改进点

### 2.1 已有 ✅
- 关键词重叠度评分（0.3权重）
- 时间近远评分（0.2权重）
- 角色加权评分（0.15权重）
- 信号词检测（0.15权重）
- 内容密度评分
- priority_rules 匹配
- InfoType 分类（9种类型）

### 2.2 缺失 ❌ — 需要新增

#### (A) 首尾位置加权

**来源**：申论"首尾最容易出答案"

**当前问题**：Retriever 的 recency bonus 只考虑了"越新越重要"，但没有给**段首段尾**额外加权。

**改动**：在 `_calculate_relevance` 中新增：
```python
# 7. 首尾位置加权（申论方法论：首尾最容易出答案）
# 检测 block 在消息列表中的位置特征
is_near_start = (position < total * 0.2)   # 前20%
is_near_end   = (position > total * 0.8)   # 后20%
if is_near_start or is_near_end:
    score += 0.05  # 首尾加权
```

#### (B) 词频特征加权

**来源**：申论"反复出现的词=命题人暗示"

**当前问题**：没有统计 block 中关键词的词频，只做了 keyword overlap。

**改动**：在 `_calculate_relevance` 中新增：
```python
# 8. 词频加权（申论方法论：反复出现的词=暗示）
content_tokens = self._tokenize([block.content])
if content_tokens:
    from collections import Counter
    freq = Counter(content_tokens)
    # 找到与 query 重叠且高频的词
    overlap_high_freq = sum(
        freq[w] for w in query_keywords if w in freq and freq[w] >= 2
    )
    if overlap_high_freq > 0:
        score += min(overlap_high_freq * 0.03, 0.10)
```

#### (C) 来源权威性加权

**来源**：申论"名人发言、关键表态必记必抄"

**当前问题**：角色加权（user 0.8 / assistant 0.6 / system 0.3）只区分了角色，没有区分"权威性"。

**改动**：增强 `_calculate_relevance` 中的角色加权逻辑：
```python
# 9. 来源权威性加权（申论方法论：关键表态必记必抄）
# 检测是否有权威来源特征
authority_signals = [
    "officially", "confirmed", "decision", "approved", "final",
    "官方", "确认", "决策", "最终", "规定", "制度",
    "definitely", "must", "require", "mandate",
    "必须", "要求", "规定", "确定", "决定",
]
auth_count = sum(1 for s in authority_signals if s in block.content.lower())
if auth_count > 0:
    score += min(auth_count * 0.05, 0.15)
```

#### (D) 结构层次检测

**来源**：申论"按结构/层次分类"——表层现象→中层制度→深层根源

**改动**：新增层次检测：
```python
# 10. 结构层次加权（申论方法论：深层原因比表层现象更有价值）
deep_signals = [
    "根本原因", "根源", "制度性", "结构性", "深层次",
    "root cause", "fundamental", "structural", "systemic",
]
surface_signals = [
    "表面", "现象", "表现为", "看起来",
    "surface", "appears", "symptom", "manifestation",
]
if any(s in block.content.lower() for s in deep_signals):
    score += 0.08  # 深层分析加权
elif any(s in block.content.lower() for s in surface_signals):
    score -= 0.03  # 表层现象轻微降权
```

---

## 三、Processor（整理合并器）改进点

### 3.1 已有 ✅
- LLM 智能合并（语义相似 block 合并）
- 规则回退合并（按 InfoType + 位置）
- 总括词提取（从 Markdown 标题/加粗文本/冒号前文字提取）
- 口语→书面语规范化（NORMALIZE_MAP）
- 内容去重（基于 content hash）

### 3.2 缺失 ❌ — 需要新增

#### (A) 共性问题 vs 个性问题识别

**来源**：申论"按表现形态分类"——共性问题 vs 个性问题

**当前问题**：合并时只看语义相似度，没有区分"跨 block 重复出现的"和"只在单个 block 出现的"。

**改动**：在 `_merge_content` 中新增：
```python
def _classify_uniqueness(self, merged_content: str, all_blocks: list) -> str:
    """判断内容是共性问题还是个性问题。
    
    共性问题：在多个 block 中出现（跨主题重复）
    个性问题：只在一个 block 中出现（独特信息）
    """
    content_keywords = set(self._tokenize([merged_content]))
    occurrence_count = 0
    for block in all_blocks:
        block_keywords = set(self._tokenize([block.content]))
        if len(content_keywords & block_keywords) / max(len(content_keywords), 1) > 0.3:
            occurrence_count += 1
    
    if occurrence_count >= 3:
        return "common"     # 共性问题
    elif occurrence_count == 1:
        return "unique"     # 个性问题
    else:
        return "limited"    # 有限重复
```

**输出标注**：在 `CompressedBlock.metadata` 中记录：
```python
compressed.metadata["uniqueness"] = uniqueness  # "common" / "unique" / "limited"
```

#### (B) 合并时识别"问题→做法→意义"三元组

**来源**：申论"给材料贴标签——问题、做法、意义"

**当前问题**：合并时只看 InfoType，没有将相关的 PROBLEM → ACTION → EFFECT 关联起来。

**改动**：在 `_merge_blocks` 后，新增一步"因果链组装"：
```python
def _assemble_causal_chains(self, compressed_blocks: list[CompressedBlock]) -> list[CompressedBlock]:
    """将 PROBLEM → ACTION → EFFECT 关联为因果链。
    
    如果连续的几个 block 分别是问题、对策、效果，
    则标注它们属于同一条因果链，便于 Formatter 按因果式输出。
    """
    for i, block in enumerate(compressed_blocks):
        if block.info_type == InfoType.PROBLEM:
            # 向后查找关联的 ACTION 和 EFFECT
            chain = [block]
            for j in range(i+1, min(i+4, len(compressed_blocks))):
                if compressed_blocks[j].info_type == InfoType.ACTION:
                    chain.append(compressed_blocks[j])
                elif compressed_blocks[j].info_type in (InfoType.EFFECT_POS, InfoType.EFFECT_NEG):
                    chain.append(compressed_blocks[j])
            
            if len(chain) >= 2:
                # 标注因果链关系
                for c in chain:
                    c.metadata["causal_chain_id"] = f"chain_{i}"
```

#### (C) 合并质量的三标准检查

**来源**：申论"分论点写完用三个标准检查"

**改动**：在 `process` 方法末尾新增质量检查：
```python
def _validate_output(self, compressed_blocks: list[CompressedBlock]) -> list[CompressedBlock]:
    """申论三标准检查：
    ① 能不能从材料里找到依据？（source traceability）
    ② 能不能单独写一段？（self-containment）
    ③ 三个论点有没有重复？（no redundancy）
    """
    validated = []
    seen_content = set()
    
    for block in compressed_blocks:
        # ① source traceability: 必须有 original_indices
        if not block.original_indices:
            continue
        
        # ② self-containment: 内容不能太短（至少50字符）
        if len(block.content.strip()) < 50:
            continue
        
        # ③ no redundancy: 检查与前序 block 的重复度
        content_sig = block.content[:100]
        is_duplicate = False
        for prev_sig in seen_content:
            # 简单重复检测
            if content_sig == prev_sig:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        
        seen_content.add(content_sig)
        validated.append(block)
    
    return validated
```

---

## 四、Formatter（格式化输出器）改进点

### 4.1 已有 ✅
- 三种输出格式：sequential / structured / bullet
- Token 预算控制（二分查找截断）
- Block 优先级排序（InfoType + 压缩率 + 原始位置）

### 4.2 缺失 ❌ — 需要新增

#### (A) 输出逻辑结构支持

**来源**：申论三种逻辑结构（并列式/因果式/递进式）

**当前问题**：Formatter 有三种格式（sequential/structured/bullet），但没有"逻辑结构"概念。structured 格式只是按 `summary_label` 分组，不考虑逻辑关系。

**改动**：新增 `output_logic` 处理逻辑：
```python
async def format(self, blocks, task_profile):
    ...
    output_logic = task_profile.output_logic
    
    if output_logic == "causal":
        result = self._format_causal(blocks, token_budget)
    elif output_logic == "progressive":
        result = self._format_progressive(blocks, token_budget)
    else:  # parallel (default)
        result = self._format_parallel(blocks, token_budget)

def _format_causal(self, blocks, token_budget):
    """因果式：原因/问题 → 对策/做法 → 效果/影响"""
    # 按 InfoType 排序：CAUSE → PROBLEM → ACTION → EFFECT
    type_order = {
        InfoType.CAUSE: 0,
        InfoType.PROBLEM: 1,
        InfoType.ACTION: 2,
        InfoType.EFFECT_POS: 3,
        InfoType.EFFECT_NEG: 3,
    }
    sorted_blocks = sorted(blocks, key=lambda b: type_order.get(b.info_type, 9))
    return self._format_structured(sorted_blocks, token_budget)

def _format_progressive(self, blocks, token_budget):
    """递进式：表层现象 → 中层制度 → 深层根源"""
    # 按 metadata["structural_layer"] 排序
    layer_order = {"surface": 0, "middle": 1, "deep": 2}
    sorted_blocks = sorted(
        blocks, 
        key=lambda b: layer_order.get(b.metadata.get("structural_layer", "middle"), 1)
    )
    return self._format_structured(sorted_blocks, token_budget)

def _format_parallel(self, blocks, token_budget):
    """并列式：按主体/领域/要素并列排列"""
    # 按 summary_label 分组，组间并列
    return self._format_structured(blocks, token_budget)
```

#### (B) 分类维度驱动的分组策略

**来源**：申论九大分类法

**当前问题**：structured 格式只按 `summary_label` 分组，没有根据 `classification_dimension` 选择分组策略。

**改动**：
```python
def _group_by_dimension(self, blocks, dimension):
    """根据分类维度选择不同的分组策略。"""
    from collections import defaultdict
    
    if dimension == "subject":
        # 按主体分组：从 block 内容中提取主体关键词
        groups = self._group_by_stakeholder(blocks)
    elif dimension == "process":
        # 按流程阶段分组：事前/事中/事后
        groups = self._group_by_process_stage(blocks)
    elif dimension == "domain":
        # 按领域分组：从 InfoType 映射到领域
        groups = self._group_by_domain(blocks)
    elif dimension == "level":
        # 按层级分组：个体→组织→社会→国家
        groups = self._group_by_level(blocks)
    else:
        # 默认按 summary_label 分组
        groups = defaultdict(list)
        for block in blocks:
            groups[block.summary_label or "其他"].append(block)
    
    return groups
```

#### (C) 输出头部标注（分类维度标识）

**来源**：申论"开头亮明总论点"

**改动**：在输出开头加一行分类维度标识：
```python
def _add_dimension_header(self, output: str, task_profile: TaskProfile) -> str:
    """在输出头部标注分类维度（类似申论的'开头亮明总论点'）。"""
    dimension_labels = {
        "subject": "按主体分类",
        "process": "按流程分类",
        "domain": "按领域分类",
        "element": "按要素分类",
        "level": "按层级分类",
        "layer": "按结构层次分类",
    }
    dim = task_profile.classification_dimension
    if dim != "auto" and dim in dimension_labels:
        header = f"## {dimension_labels[dim]}\n"
        return header + output
    return output
```

---

## 五、v0.3 Router 设计参考

### 5.1 申论方法论对 Router 的核心启发

**核心思想**：申论说"分论点不是靠灵感，是材料给你的几个角度"。Router 不需要自己发明项目，而是从消息的语义中提取项目归属。

### 5.2 三个信号 → Router 的三级分类策略

| 信号 | 申论含义 | Router 翻译 |
|------|---------|------------|
| 信号一：看材料本身结构 | 消息的内在结构暗示分类方式 | Router 分析 block 的关键词分布，自动识别主体/流程/领域维度 |
| 信号二：看题干关键词 | 用户的问题暗示分类方式 | Analyzer 根据 query 关键词选择 `classification_dimension` |
| 信号三：看要点间逻辑关系 | block 之间的逻辑暗示排序方式 | Processor 分析保留 block 的因果关系，决定输出顺序 |

### 5.3 九大分类法 → Router 的分类策略库

Router 维护一个"分类策略库"，根据信号自动选择：

```python
class ProjectRouter:
    def __init__(self):
        self.classification_strategies = {
            "subject": SubjectClassifier(),       # 按利益相关方分类
            "process": ProcessClassifier(),        # 按流程/时间分类
            "domain": DomainClassifier(),          # 按技术/管理/资源等领域分类
            "element": ElementClassifier(),        # 按组成部分分类
            "objective": ObjectiveClassifier(),    # 按主客观分类
            "level": LevelClassifier(),            # 按重要性/层级分类
            "manifestation": ManifestClassifier(), # 按共性问题/个性问题分类
            "stakeholder": StakeholderClassifier(),# 按利益相关方态度分类
            "layer": LayerClassifier(),            # 按表层/中层/深层分类
        }
```

每个 Classifier 独立实现，Router 根据信号一（材料结构）和信号二（query 关键词）选择最合适的策略。

---

## 六、改动优先级排序

| 优先级 | 改动项 | 组件 | 改动量 | 预期收益 |
|--------|--------|------|--------|---------|
| P0 | `TaskProfile` 新增 `classification_dimension` + `output_logic` | types.py | 小 | 高——解锁所有下游改进 |
| P1 | Analyzer 新增信号二：query 关键词 → 分类维度 | analyzer.py | 中 | 高——核心创新点 |
| P2 | Retriever 新增首尾位置加权 | retriever.py | 小 | 中——直接提升准确率 |
| P3 | Retriever 新增词频特征加权 | retriever.py | 小 | 中——直接提升准确率 |
| P4 | Retriever 新增来源权威性加权 | retriever.py | 小 | 中——"关键表态必记" |
| P5 | Retriever 新增结构层次检测 | retriever.py | 小 | 中——深层原因加权 |
| P6 | Processor 新增共性问题/个性问题识别 | processor.py | 中 | 中——输出质量提升 |
| P7 | Processor 新增因果链组装 | processor.py | 中 | 高——支持因果式输出 |
| P8 | Processor 新增三标准质量检查 | processor.py | 小 | 中——防止输出退化 |
| P9 | Formatter 新增 output_logic 处理 | formatter.py | 中 | 高——并列/因果/递进 |
| P10 | Formatter 新增分类维度分组 | formatter.py | 中 | 高——多维度组织输出 |
| P11 | Formatter 新增输出头部标注 | formatter.py | 小 | 低——锦上添花 |
| P12 | v0.3 Router 分类策略库 | router.py (新) | 大 | 极高——多项目隔离核心 |

**建议落地顺序**：P0 → P1 → P2-P5（Retriever 一起改）→ P6-P8（Processor 一起改）→ P9-P11（Formatter 一起改）→ P12（v0.3 大改）

---

## 七、改动影响评估

### 对现有实验结果的影响
- P0-P1：不影响，只是新增字段，默认值 `auto` 保持原行为
- P2-P5：Retriever 评分变化，可能影响 fact retention 分数，但预期是**正向**提升
- P6-P8：Processor 合并更精细，可能影响压缩率，但事实保留率预期提升
- P9-P11：Formatter 输出格式变化，不影响压缩率

### 对论文的影响
- P0-P1：论文方法论章节可大幅充实——申论方法论与框架的完整映射
- P2-P5：Retriever 改进可作为消融实验的新维度
- P6-P8：Processor 改进可作为论文"核心创新"的加强证据
- P9-P11：Formatter 改进可作为"输出质量"的量化指标

---

*文档版本：v1.0（完整映射文档）*
*日期：2026-08-03*
*状态：待用户确认后统一落地*
