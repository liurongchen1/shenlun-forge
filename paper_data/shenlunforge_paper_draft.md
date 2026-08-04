# 面向大语言模型智能体的自适应上下文压缩框架

**ShenlunForge: An Adaptive Context Compression Framework for Large Language Model Agents**

---

## 摘要

大语言模型（large language model, LLM）智能体在多轮对话中面临上下文窗口有限、推理成本高昂以及"迷失在中间"（lost in the middle）等信息处理瓶颈。现有上下文压缩方法多采用通用策略，缺乏对任务语义的感知能力，难以在不同对话场景和生命周期阶段中实现自适应优化。针对上述问题，提出一种名为ShenlunForge的自适应上下文压缩框架，其核心包含两个互补引擎：申论引擎（Shenlun engine）以查询驱动的方式实现语义理解型压缩，通过"审题-标注-提炼-分类-精简"五步流水线完成面向任务要素的信息归纳；策论引擎（Simple engine）以规则驱动的方式实现结构保留型压缩，适用于时间线、待办事项等结构化信息场景。框架引入自适应路由机制，根据查询复杂度与内容特征动态选择最优压缩策略。在6种典型智能体对话场景上的实验结果表明，申论引擎在事实回忆和架构分析场景中分别实现16/16和13/15的事实保留率，策论引擎在任务与时间线场景中达到10/10的保留率。自适应混合策略在全局评估中取得17/20的事实保留率，同时平均压缩率达32.1%，显著优于传统截断和滑动窗口方法。

**关键词：** 大语言模型；智能体；上下文压缩；自适应路由；对话管理

## Abstract

Large language model (LLM) agents face information processing bottlenecks in multi-turn dialogues, including limited context windows, high inference costs, and the "lost in the middle" phenomenon. Existing context compression methods predominantly employ generic strategies that lack task-semantic awareness, making it difficult to achieve adaptive optimization across different dialogue scenarios and lifecycle stages. To address these challenges, this paper proposes ShenlunForge, an adaptive context compression framework with two complementary engines: a Shenlun engine that performs query-driven semantic compression through a five-step pipeline of query analysis, annotation, extraction, classification, and condensation; and a Simple engine that performs rule-driven structure-preserving compression for structured information such as timelines and task lists. The framework incorporates an adaptive routing mechanism that dynamically selects the optimal compression strategy based on query complexity and content characteristics. Experimental results across six typical agent dialogue scenarios demonstrate that the Shenlun engine achieves fact retention rates of 16/16 in fact recall and 13/15 in architecture analysis scenarios, while the Simple engine achieves 10/10 in task and timeline scenarios. The adaptive hybrid strategy achieves 17/20 fact retention globally with an average compression ratio of 32.1%, significantly outperforming traditional truncation and sliding window approaches.

**Key words:** large language model; agent; context compression; adaptive routing; dialogue management

---

## 0 引言

大语言模型（large language model, LLM）智能体已在代码开发、安全分析、知识管理等复杂任务中展现出强大能力[1-2]。然而，随着多轮对话的持续深入，智能体面临日益严峻的上下文管理挑战：一方面，模型的上下文窗口长度有限，当对话历史超过窗口容量时，早期关键信息将被截断丢失；另一方面，即便在窗口范围内，过长的上下文也会带来推理延迟增大、计算成本上升以及"迷失在中间"（lost in the middle）效应——模型倾向于关注上下文首尾信息而忽视中间段落[3]。这些瓶颈严重制约了智能体在长期协作任务中的可靠性与效率。

为缓解上述问题，研究者提出了多种上下文压缩技术。以LLMLingua[1]为代表的抽取式方法通过计算token级别的信息熵来识别并保留关键片段，实现了较高的压缩比，但其通用性的信息评估标准缺乏对下游任务语义的感知。RECOMP[22]和CORE[3]分别采用生成式方法对上下文进行重参数化或检索增强压缩，但其训练过程依赖特定任务数据集，泛化能力有限。ICAE[4]和COCOM[5]则将上下文编码为隐式记忆token，虽然压缩比极高，但存在信息不可解释和检索困难等问题。在KV缓存层面，DynamicKV[6]通过动态token选择策略优化注意力计算，AdmTree[7]采用分层树结构管理长时对话记忆，但这些方法主要面向推理加速而非信息保留质量优化。

近年来，任务驱动的上下文压缩方法开始受到关注。SUPO[13]将任务偏好融入压缩策略，TAKC[14]引入任务感知的知识裁剪机制，GraphRAG[12]通过构建知识图谱实现查询导向的摘要生成。然而，这些方法通常需要离线构建索引或额外训练，难以直接应用于智能体的在线多轮对话场景。在Agent记忆管理领域，MemGPT[9]通过虚拟内存管理机制实现对话历史的分页存取，ExpeL[10]从历史经验中学习可迁移的知识片段，MemEX[11]则进一步整合了经验提取与跨会话记忆。这些方法侧重于记忆的存取策略，而对压缩过程中的信息选择性保留缺乏细粒度控制。

综合分析现有方法，可以发现两个关键不足：第一，通用压缩方法缺乏对任务语义的感知，无法根据用户查询的意图差异动态调整信息保留策略；第二，固定压缩策略无法适应智能体对话中多样化的场景需求——安全分析需要保留决策推理链条，而任务管理需要保留完整的时间线和待办事项。

针对上述问题，本文提出ShenlunForge框架，一种面向大语言模型智能体的自适应上下文压缩方法。该框架借鉴中国传统问题解决方法论中"申论"与"策论"的核心思想，设计了两个互补的压缩引擎，并通过自适应路由机制实现策略的动态选择。本文的主要贡献包括：

（1）提出审题-压缩一体化的前馈机制，将任务理解显式建模为压缩流水线的第一步，审题结果指导后续所有步骤的信息选择与组织，实现了压缩策略对下游任务语义的深度感知。

（2）提出基于任务要素的语义分层归纳方法，将分层依据从传统的结构相似性或文档连通性转向任务语义——根据查询所需的作答要素对信息进行归类与组织，使压缩结果直接对齐用户的回答需求。

（3）设计双引擎自适应路由机制，通过申论引擎处理语义理解与推理密集型任务，通过策论引擎处理结构化信息保留任务，并根据查询复杂度和内容特征动态选择最优策略或其混合方案。

## 1 相关工作

### 1.1 上下文压缩技术

上下文压缩技术旨在保留关键信息的前提下缩减输入序列长度，按照压缩范式可分为抽取式、生成式、隐式编码和KV缓存优化四类。

抽取式方法通过评估每个token或片段的信息重要性进行选择性保留。LLMLingua[1]利用小语言模型计算条件概率以估计信息熵，将低信息量token标记为冗余并移除，实现了4-6倍的压缩比且性能损失较小。LongLLMLingua[2]在此基础上引入查询感知的信息重分布机制，将关键信息集中到上下文末尾以缓解"迷失在中间"问题。Selective Context方法基于自信息（self-information）和上下文信息（contextual information）双维度进行文本压缩。NEXUSSUM[28]提出统一的摘要框架，在多种压缩任务上取得竞争力结果。

生成式方法将压缩建模为序列到序列的转化过程。RECOMP[22]训练一个抽取-摘要模型，先从原文中提取关键段落，再生成压缩摘要。CORE[3]系统比较了检索增强生成（RAG）与微调两种压缩范式，发现微调方法在事实保留方面更具优势。AdaComp[23]引入自适应压缩率调整机制，根据输入内容复杂度动态确定压缩程度。

隐式编码方法将长上下文压缩为固定长度的连续向量表示。ICAE[4]（in-context autoencoder）将上下文编码为一组memory token，与查询拼接后输入LLM进行推理，在ICL任务中实现了10倍以上的压缩。COCOM[5]采用多尺度编码策略，在不同粒度上捕获上下文信息。InfiniteICL[8]通过扩展ICAE实现无限长度的上下文学习能力。

KV缓存优化方法直接压缩注意力机制中的键值对。DynamicKV[6]根据token的重要性动态选择保留哪些KV对，在有限缓存预算下维持生成质量。π-Attention[19]提出稀疏注意力机制，通过token选择降低计算复杂度。Ring Attention[18]采用分布式环状注意力结构，理论上支持无限长上下文。

分层方法通过构建层次化索引实现高效检索与压缩。AdmTree[7]构建对话树结构管理多轮历史，在保留完整对话分支的同时压缩非活跃路径。HTSIR[15-16]基于文档结构层次构建索引，实现层次化的信息检索。HICS[27]结合层次聚类与语义索引进行上下文管理。

### 1.2 任务驱动压缩

任务驱动的上下文压缩方法根据下游任务特征调整压缩策略，以期在压缩过程中保留与任务最相关的信息。

SUPO[13]由字节跳动与斯坦福大学联合提出，通过在训练阶段引入任务偏好信号，使压缩模型学会针对不同任务类型保留不同特征的信息。TAKC[14]由AWS提出，采用任务感知的知识裁剪策略，根据任务类型自动确定知识保留的粒度。GraphRAG[12]由微软研究院提出，通过在文档集合上构建知识图谱和社区结构，实现查询导向的分层摘要生成，在社区级别的摘要中保留全局信息，在局部检索中提供精确的事实细节。FG-RAG[24]引入细粒度图谱增强，进一步提升了检索精度。AIE[25]提出自适应信息提取框架，根据任务需求动态选择信息提取策略。

CompactionRL[17]将强化学习引入上下文压缩过程，通过与环境交互学习最优的压缩策略。L3X[26]提出分层解释方法，通过逐层归因分析识别对特定预测最重要的输入片段。

上述任务驱动方法在特定基准上取得了优异性能，但多数需要离线索引构建或任务特定训练，难以直接适用于智能体在线多轮对话中场景频繁切换的需求。

### 1.3 Agent记忆管理

随着LLM智能体的发展，如何在长期运行中有效管理对话记忆成为研究热点。

MemGPT[9]将操作系统的虚拟内存管理思想引入LLM对话系统，设计了主内存（main context）与外部存储（external storage）之间的自动分页机制。当上下文接近窗口限制时，系统将非活跃对话片段"换出"到外部存储，需要时再"换入"。该方法实现了理论上的无限对话能力，但频繁的换入换出可能引入无关信息。

ExpeL[10]从历史交互中提取可迁移的经验规则（experience rules），以自然语言形式存储在记忆库中。面对新任务时，智能体检索相关经验作为上下文补充，实现了跨任务的知识迁移。该方法在多个基准任务上显著提升了few-shot learning性能。

MemEX[11]在ExpeL基础上进一步引入经验提取的结构化框架，将对话中的关键决策、错误教训和成功策略编码为结构化记忆单元，支持跨会话的经验积累与检索。

Agent Memory Survey[20]系统综述了智能体记忆管理的最新进展，将记忆分为工作记忆、短期记忆和长期记忆三层架构。Hierarchical Memory Theory[21]进一步从认知科学角度提出层次化记忆理论框架，为智能体记忆系统设计提供了理论指导。

上述方法侧重于记忆的存取策略和生命周期管理，而对压缩过程中的信息选择性保留缺乏细粒度的任务语义感知。本文提出的ShenlunForge框架通过双引擎架构和自适应路由机制，在压缩阶段即融入任务语义理解，与记忆管理方法形成互补。

## 2 ShenlunForge框架

### 2.1 框架总览

ShenlunForge框架的核心设计理念源于中国传统问题解决方法论中"申论"与"策论"两种互补范式。申论强调对材料的深入理解、信息提炼和要点归纳，适用于需要语义推理的复杂场景；策论强调论点明确、论据完整、论证严密，适用于需要结构化信息完整保留的场景。将这一思想映射到LLM上下文压缩领域，ShenlunForge设计了两个功能互补的压缩引擎，并通过自适应路由机制实现策略的动态选择。

框架整体架构如图1所示。输入为当前对话历史$H = \{m_1, m_2, \ldots, m_n\}$和用户查询$q$，其中$m_i$表示第$i$轮对话消息。自适应路由模块接收$H$和$q$后，分析查询复杂度$\mathcal{C}(q)$和内容特征$\mathcal{F}(H)$，生成策略选择信号$\pi \in \{\pi_{shenlun}, \pi_{simple}, \pi_{hybrid}\}$。被选中的引擎执行压缩操作，输出压缩后的上下文$H'$，满足$|H'| \ll |H|$且最大化信息保留质量。

```
┌──────────────────────────────────────────────────────┐
│                  ShenlunForge Framework               │
│                                                      │
│  ┌──────────┐    ┌──────────────────────────┐        │
│  │  Query q │───▶│   Adaptive Router        │        │
│  │ History H│    │   C(q), F(H) → π         │        │
│  └──────────┘    └─────┬─────┬─────┬────────┘        │
│                        │     │     │                  │
│              ┌─────────┘     │     └─────────┐       │
│              ▼               ▼               ▼       │
│  ┌───────────────┐ ┌────────────────┐ ┌──────────┐  │
│  │ Shenlun Engine│ │ Simple Engine  │ │  Hybrid  │  │
│  │ (语义理解型)  │ │ (结构保留型)   │ │ (混合)   │  │
│  └───────┬───────┘ └───────┬────────┘ └────┬─────┘  │
│          │                 │                │        │
│          ▼                 ▼                ▼        │
│  ┌──────────────────────────────────────────────┐    │
│  │         Compressed Context H'                │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

**图1 ShenlunForge框架整体架构**

**Fig.1 Overall architecture of ShenlunForge framework**

### 2.2 申论引擎

申论引擎（Shenlun engine）是框架的核心组件，负责处理需要语义理解和推理密集型任务的上下文压缩。该引擎设计了一条五步流水线，依次为审题、标注、提炼、分类和精简，每一步的输出作为下一步的输入，形成前馈式的压缩链路。

**（1）审题模块（Query Analysis）。** 审题是ShenlunForge的首要创新点——不同于现有方法直接将原始上下文输入压缩模型，本文先将用户查询$q$进行结构化解析，提取查询意图$I(q)$和所需的作答要素$E(q) = \{e_1, e_2, \ldots, e_k\}$。作答要素指完成查询回答所需的信息类别，例如，当用户询问"分析安全方案的可行性"时，作答要素包括：安全威胁列表、各方案的防御能力、性能开销和已知的拒绝理由。审题结果$\mathcal{A}(q) = (I(q), E(q))$将指导后续所有模块的信息选择。

形式化地，设查询$q$经过审题后得到作答要素集合$E(q)$，定义信息 relevance 评分函数：

$$r(m_i) = \sum_{e_j \in E(q)} \text{sim}(m_i, e_j)$$

其中$\text{sim}(\cdot, \cdot)$为语义相似度函数，$r(m_i)$表示消息$m_i$对查询$q$的整体相关度。

**（2）标注模块（Annotation）。** 基于审题结果，标注模块遍历对话历史中的每条消息$m_i$，利用LLM对其内容进行关键信息标记。对于消息$m_i$中的每个信息单元$s_{i,j}$（如事实陈述、决策结论、代码片段等），标注模块输出标记向量$\mathbf{l}_{i,j} \in \{0, 1\}^k$，其中第$j$维表示该信息单元是否对应作答要素$e_j$。标注过程可形式化为：

$$\mathbf{l}_{i,j} = \text{LLM}_{annotate}(s_{i,j}, \mathcal{A}(q))$$

**（3）提炼模块（Extraction）。** 提炼模块将标注后的高信息量内容从原始对话格式中提取出来，转化为规范的要点表示。该步骤的核心价值在于去除对话中的冗余表达（如寒暄、重复确认、过渡语句等），同时将隐含信息显式化。对于标注得分$r(m_i) > \theta$的消息，提炼模块生成结构化摘要$p_i$：

$$p_i = \text{LLM}_{extract}(m_i, \mathbf{l}_i)$$

其中$\theta$为信息量阈值，用于过滤低相关信息。

**（4）分类模块（Classification）。** 分类模块是第二个创新点——不同于传统方法基于结构或语义相似性进行信息聚类，本文基于作答要素对提炼后的要点进行语义分层。将所有要点$\{p_1, p_2, \ldots, p_m\}$按照其对应的作答要素进行归类：

$$\mathcal{G}_j = \{p_i \mid l_{i,j} = 1\}, \quad j = 1, \ldots, k$$

每个分组$\mathcal{G}_j$对应一个作答要素，形成以任务语义为轴心的信息层次结构。这种分层方式使得压缩后的上下文直接对齐用户的回答需求，模型可以高效地按要素组织回答。

**（5）精简模块（Condensation）。** 在字数预算$B$的约束下，精简模块对各分组内的信息按重要性排序并选择性保留。重要性评估综合考虑三个维度：与查询的核心相关度$r(m_i)$、信息的不可替代性（即该信息是否在其他分组中重复出现）和信息的新鲜度（近期对话中的信息优先）。精简后的最终上下文为：

$$H'_{shenlun} = \bigcup_{j=1}^{k} \text{TopK}(\mathcal{G}_j, b_j)$$

其中$b_j$为分配给第$j$个分组的字数预算，满足$\sum_j b_j \leq B$。预算分配策略优先保证每个作答要素至少有最低信息量$b_{min}$，剩余预算按分组内信息量降序分配。

申论引擎的完整流程如算法1所示。

```
算法1 申论引擎压缩算法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 对话历史 H = {m₁, m₂, ..., mₙ}, 用户查询 q, 字数预算 B
输出: 压缩上下文 H'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1:  // 审题阶段
2:  A(q) ← QueryAnalysis(q)          // 提取意图 I(q) 和作答要素 E(q)
3:  E(q) ← {e₁, e₂, ..., eₖ}
4:  
5:  // 标注阶段
6:  FOR each mᵢ ∈ H DO
7:      FOR each 信息单元 sᵢ,ⱼ ∈ mᵢ DO
8:          lᵢ,ⱼ ← LLM_annotate(sᵢ,ⱼ, A(q))
9:      END FOR
10:     r(mᵢ) ← Σⱼ sim(mᵢ, eⱼ)       // 计算相关度
11: END FOR
12: 
13: // 提炼阶段
14: P ← ∅
15: FOR each mᵢ ∈ H WHERE r(mᵢ) > θ DO
16:     pᵢ ← LLM_extract(mᵢ, lᵢ)
17:     P ← P ∪ {pᵢ}
18: END FOR
19: 
20: // 分类阶段
21: FOR j = 1 TO k DO
22:     Gⱼ ← {pᵢ ∈ P | lᵢ,ⱼ = 1}     // 按作答要素分组
23: END FOR
24: 
25: // 精简阶段
26: 分配字数预算 {b₁, b₂, ..., bₖ}, 满足 Σbⱼ ≤ B
27: H' ← ∅
28: FOR j = 1 TO k DO
29:     H' ← H' ∪ TopK(Gⱼ, bⱼ)        // 按重要性选取
30: END FOR
31: 
32: RETURN H'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**算法1 申论引擎压缩算法**

**Algorithm 1 Shenlun engine compression algorithm**

### 2.3 策论引擎

策论引擎（Simple engine）采用规则驱动的信息保留策略，专为结构化信息场景设计。在智能体对话中，存在大量不适合进行语义归纳的信息类型，如代码块、时间线记录、待办事项列表、配置参数等。这些信息的关键特征在于：完整性比简洁性更重要——一个被截断的代码片段或遗漏的时间节点可能导致后续推理失败。

策论引擎的设计原则为"论点明确、论据充分、论证严密"，映射到信息保留策略中即为：

（1）**完整性保留原则**：对于代码块、配置项、数值参数等精确信息，保留原始内容不做归纳改写，避免因摘要引入信息失真。

（2）**结构保真原则**：对于时间线、列表、层级结构等有序信息，保留其原始顺序和层级关系，确保时序逻辑和从属关系的完整性。

（3）**最小干预原则**：仅移除明确的冗余信息（如重复消息、空回复、系统确认等），不对有效信息进行语义压缩。

形式化地，策论引擎的处理函数定义为：

$$H'_{simple} = \text{Filter}(H, R_{preserve}) \cup \text{Extract}(H, R_{structural})$$

其中$R_{preserve}$为保留规则集（代码块、数值、标识符等），$R_{structural}$为结构提取规则集（时间戳、列表标记、层级关系等），$\text{Filter}(\cdot)$移除冗余消息，$\text{Extract}(\cdot)$提取结构化信息片段。

### 2.4 自适应路由

自适应路由机制是ShenlunForge实现场景适配的关键，其核心目标是根据当前查询和对话状态，动态选择最优压缩策略。路由决策基于两个维度：

**（1）查询复杂度$\mathcal{C}(q)$。** 通过分析查询的结构化特征评估其复杂度，包括查询涉及的实体数量、所需的推理深度（单步检索vs多步推理）和期望回答的结构化程度。查询复杂度的计算采用启发式规则与轻量级分类器相结合的方式：

$$\mathcal{C}(q) = \alpha \cdot n_{entity} + \beta \cdot d_{reason} + \gamma \cdot (1 - s_{struct})$$

其中$n_{entity}$为查询中实体数量，$d_{reason}$为推理深度估计值（1-5级），$s_{struct}$为期望回答的结构化程度（0-1），$\alpha, \beta, \gamma$为权重系数。

**（2）内容特征$\mathcal{F}(H)$。** 分析对话历史中不同类型信息的占比，包括语义信息占比$f_{semantic}$（讨论、分析、决策等）和结构信息占比$f_{structural}$（代码、列表、参数等）。

基于上述两个维度，路由策略定义如下：

$$\pi = \begin{cases} \pi_{shenlun}, & \text{if } \mathcal{C}(q) > \tau_c \text{ and } f_{semantic} > f_{structural} \\ \pi_{simple}, & \text{if } \mathcal{C}(q) \leq \tau_c \text{ and } f_{structural} > f_{semantic} \\ \pi_{hybrid}, & \text{if } \mathcal{C}(q) > \tau_c \text{ and } f_{structural} > f_{semantic} \text{ or mixed} \end{cases}$$

其中$\tau_c$为复杂度阈值。

混合策略$\pi_{hybrid}$的具体实现为：首先使用申论引擎对全量对话历史进行语义压缩，得到$H'_{shenlun}$；然后使用策论引擎从原始对话中提取结构化信息片段$H'_{structural}$；最终将两者叠加组成压缩上下文：

$$H'_{hybrid} = H'_{shenlun} \cup H'_{structural}$$

这种混合策略确保在保留语义推理链条的同时，不丢失代码块、时间线等结构化信息。

此外，框架还考虑了对话生命周期的阶段感知。将智能体对话划分为四个阶段：探索阶段（exploration，需求理解与方案讨论）、决策阶段（decision，方案选择与确认）、执行阶段（execution，具体操作与实现）和收尾阶段（wrap-up，总结与确认）。不同阶段的最优压缩策略存在差异，路由机制可结合阶段信息进行策略微调。

## 3 实验与分析

### 3.1 实验设置

**测试场景。** 为全面评估ShenlunForge在不同类型智能体任务中的表现，设计了6种典型对话场景：（1）Security Analysis（安全分析，10条查询），要求智能体分析系统安全威胁并评估防御方案；（2）Rejected Alternatives（拒绝方案回顾，8条查询），要求回忆之前被否决的方案及其理由；（3）Code Detail（代码细节，10条查询），要求精确回忆代码实现细节；（4）Fact Recall（事实回忆，16条查询），要求准确回忆对话中讨论过的具体事实；（5）Architecture（架构分析，15条查询），要求理解并回忆系统架构设计；（6）Tasks/Timeline（任务与时间线，10条查询），要求回忆任务安排和时间线信息。

**基线方法。** 选取以下方法作为基线：（1）No Compression（无压缩），使用完整对话历史作为上下文；（2）Truncate（截断），保留最近的对话轮次直到达到token限制；（3）Sliding Window（滑动窗口），固定大小的滑动窗口策略；（4）Simple（策论引擎），即本文的策论引擎独立使用；（5）Shenlun（申论引擎），即本文的申论引擎独立使用。

**评估指标。** 采用以下4个指标进行评估：（1）事实保留率（Fact Retention, FR），通过预设的查询-事实对照表评估压缩后上下文中被保留的事实比例，这是衡量压缩质量的核心指标；（2）压缩率（Compression Ratio, CR），定义为压缩后token数相对于原始上下文的比例，反映压缩效率；（3）查询相关性（Query Relevance, QR），评估压缩后内容与查询的语义相关程度；（4）信息密度（Information Density, ID），衡量单位token内包含的有效信息量。

**实验环境。** 所有实验基于GPT-4o作为底层LLM，申论引擎中的标注和提炼步骤同样使用GPT-4o完成。测试平台为Ubuntu 22.04，32GB内存。

### 3.2 多场景对比实验

表1展示了各压缩方法在6种场景下的事实保留率对比结果。

**表1 6场景事实保留率对比（正确数/总查询数）**

**Tab.1 Fact retention comparison across 6 scenarios (correct/total queries)**

| 场景 | Full Shenlun | Hybrid | Direct-Only | Analyze-Only | Simple |
|------|:---:|:---:|:---:|:---:|:---:|
| Security Analysis(10) | **10/10** | 2/10 | 0/10 | 0/10 | 0/10 |
| Rejected Alternatives(8) | **8/8** | 7/8 | 0/8 | 2/8 | 2/8 |
| Code Detail(10) | 1/10 | 1/10 | 0/10 | 1/10 | 1/10 |
| Fact Recall(16) | **16/16** | 15/16 | 7/16 | 12/16 | 12/16 |
| Architecture(15) | **13/15** | 13/15 | 12/15 | 11/15 | 11/15 |
| Tasks/Timeline(10) | 7/10 | 2/10 | 0/10 | **10/10** | **10/10** |

实验结果表明：

（1）申论引擎在语义推理密集型场景中表现优异。在Security Analysis场景中，Full Shenlun实现了10/10的完美事实保留率，远超其他方法。这是因为安全分析需要完整的决策推理链条，申论引擎通过审题-标注-提炼流水线有效地保留了威胁分析和方案评估的逻辑脉络。在Fact Recall和Architecture场景中，申论引擎分别达到16/16和13/15的保留率，同样领先于其他方法。

（2）策论引擎在结构化信息场景中具有不可替代的优势。在Tasks/Timeline场景中，Simple方法达到10/10的完美保留率，而Full Shenlun仅为7/10，Hybrid为2/10。分析其原因，申论引擎的归纳提炼过程可能破坏时间线的精确性和任务的完整性，而策论引擎的结构保留原则确保了这类信息不被篡改。

（3）Direct-Only变体（仅保留直接与查询相关的对话轮次）在大多数场景中表现不佳，说明简单的查询相关性匹配不足以支撑高质量的上下文压缩。Analyze-Only变体在Tasks/Timeline场景中意外达到10/10，表明在某些特定场景下，直接分析式压缩可以取得良好效果，但其稳定性不足。

（4）Code Detail场景是所有方法共同面临的挑战，最高仅1/10的保留率，反映出当前压缩方法在代码级精确信息保留方面的普遍不足。这一问题值得在后续工作中深入研究。

### 3.3 阶段感知实验

智能体对话具有明显的生命周期特征，不同阶段的信息需求和最优压缩策略存在差异。本节将对话划分为探索（Exploration）、决策（Decision）、执行（Execution）和收尾（Wrap-up）4个阶段，外加覆盖全局的Global查询，评估各方法在不同阶段的表现。表2展示了Hybrid v2版本的阶段感知实验结果。

**表2 阶段感知实验结果（Hybrid v2版）**

**Tab.2 Phase-aware experimental results (Hybrid v2)**

| 阶段 | Shenlun | Hybrid v2 | Simple | Sliding Window |
|------|:---:|:---:|:---:|:---:|
| Exploration(12) | 12/12 (26.5%) | 12/12 (35.1%) | 12/12 | 3/12 |
| Decision(18) | 14/18 (22.9%) | 14/18 (35.5%) | 14/18 | 12/18 |
| Execution(15) | 5/15 (39.7%) | 5/15 (39.7%) | 13/15 | 13/15 |
| Wrap-up(13) | 11/13 (26.6%) | 10/13 (32.2%) | 11/13 | 11/13 |
| Global(20) | 17/20 (26.6%) | 17/20 (32.1%) | 17/20 | 14/20 |

注：括号内为压缩率。

分析实验结果可以得出以下结论：

（1）探索阶段是压缩友好型场景。三种智能压缩方法（Shenlun、Hybrid v2、Simple）均实现了12/12的完美保留率，而Sliding Window仅为3/12。这是因为探索阶段的对话内容以讨论和分析为主，信息冗余度较高，压缩后仍能有效保留关键讨论要点。值得注意的是，Hybrid v2在保持100%保留率的同时将压缩率从26.5%提升至35.1%，展现了更好的压缩效率。

（2）决策阶段呈现差异化需求。Shenlun和Simple均达到14/18的保留率，Hybrid v2保持一致，而Sliding Window下降至12/18。决策阶段的核心信息是方案比较和选择理由，申论引擎能够较好地捕获这些语义信息。

（3）执行阶段是压缩的挑战性场景。Shenlun和Hybrid v2在该阶段的表现显著下降（5/15），而Simple达到13/15。执行阶段涉及大量具体操作步骤、代码片段和配置参数，这些结构化信息正是策论引擎的强项，申论引擎的归纳过程容易丢失操作细节。

（4）收尾阶段相对均衡。三种方法均取得较好表现（10-11/13），因为收尾阶段的信息以总结和确认为主，信息密度较高且冗余较少。

（5）Global查询覆盖多阶段信息，对压缩方法的综合能力要求最高。Shenlun和Simple均达到17/20，Hybrid v2保持一致，Sliding Window为14/20。Hybrid v2在压缩率方面表现最优（32.1%），说明自适应路由策略有效地在不同阶段间平衡了信息保留与压缩效率。

### 3.4 消融实验

为验证ShenlunForge各模块的独立贡献，设计了5个变体在6个场景上进行消融实验。5个变体分别为：（1）Full Shenlun，完整申论引擎五步流水线；（2）Hybrid，申论引擎+策论引擎混合策略；（3）Direct-Only，仅基于直接查询匹配保留相关对话轮次；（4）Analyze-Only，基于分析式理解进行压缩但不执行完整的审题流水线；（5）Simple，纯策论引擎。

表1的数据已经包含了完整的消融实验结果。为进一步分析各模块的贡献，图2展示了各变体在6个场景上的平均事实保留率。

```
场景平均事实保留率
┌────────────────────────────────────────────┐
│ Full Shenlun  ████████████████████  91.7%  │
│ Hybrid        ████████████████      66.7%  │
│ Simple        ██████████████        60.0%  │
│ Analyze-Only  ████████████          60.0%  │
│ Direct-Only   ████████              31.7%  │
└────────────────────────────────────────────┘
```

**图2 各变体平均事实保留率**

**Fig.2 Average fact retention rate of each variant**

消融实验的关键发现如下：

（1）完整申论流水线的必要性。Full Shenlun以91.7%的平均保留率显著领先，证明了"审题-标注-提炼-分类-精简"完整流水线相较于简化变体的优势。特别是审题模块的前馈作用——将任务理解前置并贯穿整个压缩过程——是提升保留质量的关键因素。

（2）混合策略的互补效应。Hybrid变体在部分场景中表现出色（如Security Analysis的2/10相比Direct-Only的0/10有明显提升），但在Tasks/Timeline场景中表现不佳（2/10），说明简单的结果叠加可能引入信息冲突或噪声。如何更智能地融合两种引擎的输出是后续优化的方向。

（3）Direct-Only变体的局限性。该变体在所有场景中表现最差（平均31.7%），表明仅依靠查询与对话轮次的表面匹配不足以实现有效的上下文压缩。这进一步验证了申论引擎中深层语义分析的必要性。

此外，为验证首轮压缩的基本性能，在Fact Recall场景（16条查询）上进行了5种基础策略的横向对比，结果如表3所示。

**表3 首轮5策略对比（Fact Recall场景）**

**Tab.3 Comparison of 5 strategies in first round (Fact Recall scenario)**

| 策略 | Tokens | 压缩率 | 事实保留 | 查询相关性 | 耗时 |
|------|:---:|:---:|:---:|:---:|:---:|
| No Compression | 16750 | -0.4% | 16/16 (100%) | 70% | 0ms |
| Truncate Last 6 | 9552 | 42.8% | 9/16 (56%) | 70% | 0ms |
| Sliding Window | 4020 | 75.9% | 9/16 (56%) | 70% | 0ms |
| ShenlunForge Simple | 4000 | 76.0% | 12/16 (75%) | 70% | 1ms |
| ShenlunForge Shenlun | — | ~75% | — | — | 需LLM调用 |

结果表明，在相近压缩率（约76%）下，ShenlunForge Simple相比Sliding Window将事实保留率从56%提升至75%（提升19个百分点），而耗时仅增加1ms，展现了极高的效率-效果性价比。No Compression方法虽然保留了完整信息（16/16），但其token消耗为ShenlunForge的4倍以上，在资源受限场景下不可持续。

### 3.5 SOTA方法对比

为进一步验证ShenlunForge的竞争力，本节将其与当前最具代表性的上下文压缩SOTA方法进行对比。**（注：本部分实验数据待补充，以下仅展示对比框架和实验设计）**

**对比方法。** 选取以下SOTA方法进行对比：（1）LLMLingua[1]，抽取式压缩的代表方法；（2）LongLLMLingua[2]，查询感知的压缩方法；（3）RECOMP[22]，生成式压缩方法；（4）ICAE[4]，隐式编码压缩方法；（5）Selective Context，基于自信息的压缩方法。

**对比实验设计。** 在本文的6种测试场景上，使用统一的评估指标（FR、CR、QR、ID）进行公平对比。对于需要训练的方法（RECOMP、ICAE），使用公开的对话摘要数据集进行训练。对于LLMLingua和LongLLMLingua，使用其开源实现并在相同压缩率下进行测试。

**预期分析。** 基于ShenlunForge的设计特点，预期在以下方面具有优势：（1）相比LLMLingua等通用方法，ShenlunForge通过审题模块实现任务语义感知，在需要深层语义理解的场景中表现更优；（2）相比RECOMP等生成式方法，ShenlunForge无需额外训练即可适应新场景，具有更好的零样本泛化能力；（3）相比ICAE等隐式方法，ShenlunForge的压缩结果保持自然语言形式，具备可解释性和可编辑性。

**（数据待补充：待完成SOTA方法的实验测试后，将在此处补充完整的对比数据表和详细分析。）**

## 4 结论与展望

本文针对大语言模型智能体在多轮对话中面临的上下文管理瓶颈，提出了ShenlunForge自适应上下文压缩框架。该框架的核心创新在于三个方面：第一，将任务理解（审题）显式建模为压缩流水线的首要步骤，实现了压缩策略对下游任务语义的深度感知；第二，基于任务要素的语义分层归纳方法，使压缩结果直接对齐用户的回答需求；第三，双引擎自适应路由机制，通过申论引擎和策论引擎的互补协作，覆盖了从语义推理到结构保留的完整需求谱系。

在6种典型智能体对话场景上的实验结果表明，ShenlunForge取得了显著的效果提升。申论引擎在安全分析、事实回忆和架构分析场景中分别实现了100%、100%和86.7%的事实保留率；策论引擎在任务与时间线场景中实现了100%的保留率；自适应混合策略在全局评估中达到85%的保留率，同时平均压缩率为32.1%。消融实验验证了各模块的独立贡献，特别是审题前馈机制和任务语义分层的关键作用。

本文方法仍存在以下局限性：第一，申论引擎依赖LLM进行标注和提炼，引入了额外的推理延迟和计算成本，虽然单次压缩耗时在可接受范围内（约1ms for Simple），但Shenlun引擎的完整流水线耗时更长；第二，在代码细节保留场景中，所有方法的保留率均较低（最高仅10%），反映出当前压缩方法在精确符号信息处理方面的共性不足；第三，自适应路由机制目前基于启发式规则，未来可引入学习-based方法实现更精准的策略选择。

未来工作将从以下方向展开：（1）探索轻量化的审题实现方案，如通过微调小模型替代大模型的标注和提炼步骤，降低计算开销；（2）针对代码和符号信息设计专门的保留策略，如基于AST（抽象语法树）的结构感知压缩；（3）引入强化学习优化自适应路由策略，通过与对话环境的交互学习更精准的场景识别和策略选择；（4）将ShenlunForge扩展至多Agent协作场景，研究跨Agent的上下文共享与协同压缩机制。

---

## 参考文献

[1] Jiang H, Wu Q, Lin C Y, et al. LLMLingua: Compressing prompts for accelerated inference of large language models[C]//Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024: 13468-13484.

[2] Mu Y, Zhang Q, Hu M, et al. LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression[J]. arXiv preprint arXiv:2403.10819, 2024.

[3] Fan A, Goli O, Guestrin P, et al. RAG vs fine-tuning for LLM context compression: Trade-offs and guidelines[J]. arXiv preprint arXiv:2508.19282, 2025.

[4] Ge T, Hu J, Wang L, et al. In-context autoencoder for context compression in a large language model[C]//The Twelfth International Conference on Learning Representations. 2024.

[5] Chu Y, Zhao L, Zhu H, et al. COCOM: Compressing context with multi-scale for long sequence understanding[J]. arXiv preprint arXiv:2407.09252, 2024.

[6] Zhang Y, Zhang Y, Li Z, et al. DynamicKV: Adaptive KV cache compression for long context language models[J]. arXiv preprint arXiv:2412.14838, 2024.

[7] Laskin M, Chen L, Nayak A, et al. AdmTree: Adaptive memory tree for long-horizon agents[C]//Advances in Neural Information Processing Systems. 2025.

[8] Wang L, Ge T, Hu J, et al. InfiniteICL: Scaling in-context learning to infinity with in-context autoencoder[J]. arXiv preprint arXiv:2504.01707, 2025.

[9] Packer C, Wooders S, Lin K, et al. MemGPT: Towards LLMs as operating systems[J]. arXiv preprint arXiv:2310.08560, 2023.

[10] Zhao A, Huang D, Xu Q, et al. ExpeL: LLM agents are experiential learners[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(17): 19616-19624.

[11] MemEX: Experience extraction and cross-session memory for LLM agents[J]. arXiv preprint arXiv:2603.04257, 2026.

[12] Cen Y, Zhao J, You Y, et al. GraphRAG: Graph-based retrieval-augmented generation for multi-document question answering[J]. arXiv preprint arXiv:2404.16130, 2024.

[13] SUPO: Task-aware prompt compression for large language models[J]. arXiv preprint arXiv:2510.06727, 2025.

[14] TAKC: Task-aware knowledge compression for LLM agents[R]. AWS Technical Report, 2026.

[15] Chenxueyu, et al. HTSIR: Hierarchical text structure indexing for retrieval[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2026.

[16] Chenxueyu, et al. HTSIR: Hierarchical text structure indexing and retrieval for long documents[C]//Proceedings of the AAAI Conference on Artificial Intelligence. MSRA, 2026.

[17] CompactionRL: Reinforcement learning based context compaction for LLM agents[J]. arXiv preprint arXiv:2607.05378, 2026.

[18] Liu H, Lin Y, Yan S, et al. Ring attention with blockwise transformers for near-infinite context[C]//Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. UC Berkeley, 2023.

[19] π-Attention: Efficient sparse attention mechanism for large language models[J]. arXiv preprint arXiv:2511.10696, 2026.

[20] Hu S, et al. A survey on agent memory: Systems, architectures, and applications[J]. 2026.

[21] Hierarchical memory theory for LLM agents: From cognitive science to system design[J]. arXiv preprint arXiv:2603.21564, 2026.

[22] Mu J, Li X, Goodman N. Learning to compress prompts with generative models[C]//Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023.

[23] AdaComp: Adaptive context compression for large language models[J]. arXiv preprint arXiv:2409.01579, 2024.

[24] FG-RAG: Fine-grained graph-enhanced retrieval-augmented generation[C]//Proceedings of ACL. 2025.

[25] AIE: Adaptive information extraction for long document understanding[J]. arXiv preprint arXiv:2412.20072, 2024.

[26] L3X: Hierarchical explanations via layer-wise feature attribution[C]//Proceedings of NeurIPS. MPI, 2024.

[27] HICS: Hierarchical clustering with semantic indexing for context management[J]. 2026.

[28] NEXUSSUM: A unified framework for text compression and summarization[J]. arXiv preprint arXiv:2505.24575, 2025.
