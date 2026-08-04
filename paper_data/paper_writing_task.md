# 论文撰写任务：ShenlunForge v2 中文论文

## 任务目标
撰写一篇面向《计算机工程与设计》（北大核心+科技核心）的中文学术论文，约6000-8000字。

## 论文标题（暂定）
面向大语言模型智能体的自适应上下文压缩框架

## 投稿期刊要求
- 《计算机工程与设计》
- 中文论文，格式参照该刊模板
- 字数6000-8000字
- 需包含中英文摘要、关键词
- 图表4-5张
- 参考文献20-30篇

## 论文结构

### 摘要
- 中文摘要（200-300字）+ 英文摘要
- 关键词5个

### 第1章 引言
- LLM智能体长上下文挑战：上下文窗口限制、推理成本、"迷失在中间"问题
- 现有方法的不足：通用压缩缺乏任务针对性、固定策略无法适应多场景
- 本文贡献：提出ShenlunForge v2框架，基于申论+策论双引擎的自适应上下文压缩方法
- 三个核心创新点

### 第2章 相关工作
分3个小节：
- 2.1 上下文压缩技术（抽取式LLMLingua、生成式RECOMP/CORE、隐式ICAE/COCOM、KV缓存DynamicKV、分层AdmTree、长时记忆InfiniteICL）
- 2.2 任务驱动压缩（SUPO、TAKC、GraphRAG QFS）
- 2.3 Agent记忆管理（MemGPT、ExpeL、MemEX）

### 第3章 ShenlunForge v2框架
- 3.1 框架总览（架构图描述）
  - 双引擎设计：申论引擎（语义理解型）+ 策论引擎（结构保留型）
  - 自适应路由机制
- 3.2 申论引擎（Shenlun Strategy）
  - 审题模块：query驱动的信息重要性判断
  - 标注模块：基于LLM的关键信息标记
  - 提炼模块：从原始对话到规范要点的转化
  - 分类模块：按任务要素组织信息层次
  - 精简模块：字数约束下的信息优先级排序
- 3.3 策论引擎（Simple Strategy）
  - 适用场景：结构化信息、时间线、待办事项
  - 规则驱动的信息保留策略
- 3.4 自适应路由
  - 基于查询复杂度和内容特征的策略选择
  - 混合策略（Hybrid）：Shenlun全量 + 代码块原文叠加

### 第4章 实验与分析
- 4.1 实验设置
  - 测试场景：6种典型LLM智能体对话场景
  - 基线方法：No Compression、Truncate、Sliding Window、Simple、Shenlun
  - 评估指标：事实保留率(Fact Retention)、压缩率(Compression Ratio)、查询相关性(Query Relevance)、信息密度(Information Density)
- 4.2 多场景对比实验
  - 6场景批量实验结果（表1）
  - 分析：Shenlun在推理场景优势、Simple在结构化场景优势
- 4.3 阶段感知实验
  - 对话生命周期4阶段：探索/决策/执行/收尾
  - 结果（表2）
  - 分析：不同阶段最优策略差异
- 4.4 消融实验
  - 5变体×6场景（表3）
  - 验证各模块独立贡献
- 4.5 SOTA对比（预留位置）
  - 与LLMLingua、Selective Context等方法的对比
  - 注：此部分数据待补充，先写框架和占位文字

### 第5章 结论与展望
- 总结贡献
- 局限性
- 未来工作

## 核心实验数据

### 表1：6场景批量实验结果（消融实验数据）
| 场景 | Full Shenlun | Hybrid | Direct-Only | Analyze-Only | Simple |
|---|---|---|---|---|---|
| Security Analysis(10) | **10/10** | 2/10 | 0/10 | 0/10 | 0/10 |
| Rejected Alternatives(8) | **8/8** | 7/8 | 0/8 | 2/8 | 2/8 |
| Code Detail(10) | 1/10 | 1/10 | 0/10 | 1/10 | 1/10 |
| Fact Recall(16) | **16/16** | 15/16 | 7/16 | 12/16 | 12/16 |
| Architecture(15) | **13/15** | 13/15 | 12/15 | 11/15 | 11/15 |
| Tasks/Timeline(10) | 7/10 | 2/10 | 0/10 | **10/10** | **10/10** |

### 表2：阶段感知实验结果（Hybrid v2版）
| 阶段 | Shenlun | Hybrid v2 | Simple | Sliding |
|---|---|---|---|---|
| Exploration(12) | 12/12 (26.5%) | 12/12 (35.1%) | 12/12 | 3/12 |
| Decision(18) | 14/18 (22.9%) | 14/18 (35.5%) | 14/18 | 12/18 |
| Execution(15) | 5/15 (39.7%) | 5/15 (39.7%) | 13/15 | 13/15 |
| Wrap-up(13) | 11/13 (26.6%) | 10/13 (32.2%) | 11/13 | 11/13 |
| Global(20) | 17/20 (26.6%) | 17/20 (32.1%) | 17/20 | 14/20 |

括号内为压缩率。

### 首轮5策略单场景数据（用于表1补充或图1）
| 策略 | Tokens | 压缩率 | 事实保留 | 查询相关性 | 耗时 |
|---|---|---|---|---|---|
| No Compression | 16750 | -0.4% | 16/16 (100%) | 70% | 0ms |
| Truncate Last 6 | 9552 | 42.8% | 9/16 (56%) | 70% | 0ms |
| Sliding Window | 4020 | 75.9% | 9/16 (56%) | 70% | 0ms |
| ShenlunForge Simple | 4000 | 76.0% | 12/16 (75%) | 70% | 1ms |
| ShenlunForge Shenlun | (待查) | ~75% | (高分) | (待查) | (需LLM调用) |

## 三个核心创新点（论文中需重点阐述）

1. **审题-压缩一体化的前馈机制**：将任务理解（审题）显式建模为压缩流水线第一步，审题结果指导后续所有步骤。现有文献中尚未被系统化提出和验证。

2. **基于任务要素的语义分层归纳**：分层依据从"结构/相似性"转向"任务语义"——根据查询要求的作答要素对信息进行归类。不同于GraphRAG的图连通性分层和HTSIR的文档结构分层。

3. **双引擎自适应路由机制**：申论引擎擅长语义理解和推理，策论引擎擅长结构化信息保留。根据查询复杂度和内容特征自动选择最优策略，混合策略（Hybrid）结合两者优势。

## 框架核心设计理念
ShenlunForge框架借鉴中国传统问题解决方法论（申论+策论），将其核心思想映射到LLM上下文压缩：
- **申论四步法**→Shenlun引擎：审题(query理解)→标注(信息标记)→提炼(要点提取)→精简(字数约束)
- **策论方法论**→Simple引擎：论点明确、论据充分、论证严密→结构化信息完整保留
- **自适应路由**→混合策略：根据场景特征动态选择引擎

## 参考文献列表（至少包含以下关键文献）
1. Jiang H, et al. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2024.
2. Mu Y, et al. LongLLMLingua. 2024.
3. Fan A, et al. RAG vs Fine-tuning for LLM Context Compression (CORE). 2025. arXiv:2508.19282
4. Ge T, et al. In-Context Autoencoder for Context Compression (ICAE). ICLR 2024.
5. Chu Y, et al. COCOM: Compressing Context with Multi-Scale. 2024. arXiv:2407.09252
6. Zhang Y, et al. DynamicKV. 2024. arXiv:2412.14838
7. Laskin M, et al. AdmTree. NeurIPS 2025.
8. Wang L, et al. InfiniteICL. 2025. arXiv:2504.01707
9. Packer C, et al. MemGPT. 2023. arXiv:2310.08560
10. Zhao A, et al. ExpeL. AAAI 2024.
11. MemEX. 2026. arXiv:2603.04257
12. Cen Y, et al. GraphRAG. Microsoft Research 2024. arXiv:2404.16130
13. SUPO. ByteDance+Stanford 2025. arXiv:2510.06727
14. TAKC. AWS 2026.
15. HTSIR. MSRA, AAAI 2026.
16. Chenxueyu et al. HTSIR. AAAI 2026.
17. CompactionRL. 2026. arXiv:2607.05378
18. Ring Attention. UC Berkeley 2023.
19. π-Attention. 2026. arXiv:2511.10696
20. Hu et al. Agent Memory Survey. 2026.
21. Hierarchical Memory Theory. 2026. arXiv:2603.21564
22. RECOMP. 2023.
23. AdaComp. 2024. arXiv:2409.01579
24. FG-RAG. 2025.
25. AIE. MSRA 2024. arXiv:2412.20072
26. L3X. MPI 2024.
27. HICS. 2026.
28. NEXUSSUM. 2025. arXiv:2505.24575

## 写作要求
1. 学术论文风格，用词严谨规范
2. 中文为主，专业术语首次出现时附英文
3. 实验部分用数据说话，表格清晰
4. SOTA对比部分先预留框架和占位说明，标注"数据待补充"
5. 引言部分突出创新性和与现有方法的差异
6. 方法部分要有数学公式或伪代码描述
7. 结论部分诚实讨论局限性
8. 总字数控制在6000-8000字

## 产出文件
将完整论文写入 `/app/data/所有对话/主对话/shenlun-forge/paper_data/shenlunforge_paper_draft.md`
