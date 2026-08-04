# ShenlunForge 批量实验结果汇总

## 实验配置
- 日期：2026-08-01
- 对话数据：16条消息，模拟全栈SaaS开发会话（~16690 tokens）
- LLM后端：DeepSeek API (deepseek-chat)
- 最大输出tokens：4000
- 场景数：6个不同查询类型
- 策略数：5种压缩策略

## 策略说明
1. **No Compression (Baseline)** — 全量保留，不压缩
2. **Truncate Last 6** — 截断保留最近6条消息
3. **Sliding Window** — 滑动窗口，按token预算截断
4. **ShenlunForge Simple** — 纯规则提取（无LLM），基于消息角色+关键词打分
5. **ShenlunForge Shenlun** — LLM驱动四步流水线（审题→找点→加工→书写）

## 核心结果矩阵

| 场景 | 关键事实数 | Shenlun | Shenlun% | Simple | Simple% | Sliding | Sliding% |
|---|---|---|---|---|---|---|---|
| 1. 事实回忆 | 16 | 16/16 | 100% | 12/16 | 75% | 9/16 | 56% |
| 2. 代码细节 | 10 | 8/10 | 80% | 1/10 | 10% | 1/10 | 10% |
| 3. 架构概览 | 15 | 13/15 | 87% | 11/15 | 73% | 6/15 | 40% |
| 4. 待办时间线 | 10 | 7/10 | 70% | 10/10 | 100% | 10/10 | 100% |
| 5. 安全推理 | 10 | 8/10 | 80% | 0/10 | 0% | 0/10 | 0% |
| 6. 被否决方案 | 8 | 8/8 | 100% | 2/8 | 25% | 0/8 | 0% |

## 平均事实保留率
- **Shenlun**: 60/69 = **87%**
- **Simple**: 36/69 = **52%**
- **Sliding**: 26/69 = **38%**

## 压缩率对比
- Shenlun: 39.5% ~ 57.9% (均值 ~50%)
- Simple: ~76% 恒定
- Sliding: ~75.9% 恒定

## 耗时（Shenlun策略）
| 场景 | 耗时(ms) |
|---|---|
| Fact Recall | 10698 |
| Code Detail | 8739 |
| Architecture | 9270 |
| Tasks/Timeline | 8496 |
| Security Reasoning | 8465 |
| Rejected Alternatives | 8492 |
| **均值** | **~9027ms (~9秒)** |

## 三个关键论文发现

### 发现1: Shenlun在复杂语义任务上碾压性优势
- 安全推理：80% vs 0%（Simple/Sliding全军覆没）
- 被否决方案：100% vs 25% vs 0%
- 代码细节：80% vs 10% vs 10%
- 这些任务需要跨消息推理和语义理解，朴素方法完全无法胜任

### 发现2: Simple在结构化数据上反转胜出
- 待办/时间线：Simple 100% vs Shenlun 70%
- 原因：deadline、task名等结构化信息在对话中显式出现，规则提取即可，不需要LLM推理
- 启示：并非所有场景都需要LLM，简单场景规则方法更快更准

### 发现3: 混合策略（Hybrid）是最优解
- 简单事实提取 → Simple（毫秒级，零成本）
- 复杂推理/代码/否定 → Shenlun（~9秒，需LLM）
- 自适应路由的天然依据：可根据查询复杂度动态选择策略

## 首轮单场景数据（对比参考）
| 策略 | Tokens | Savings | Fact Retention | Duration |
|---|---|---|---|---|
| No Compression | 16750 | 0% | 16/16 (100%) | 0ms |
| Truncate Last 6 | 9552 | 43% | 9/16 (56%) | 0ms |
| Sliding Window | 4020 | 76% | 9/16 (56%) | 0ms |
| ShenlunForge Simple | 4000 | 76% | 12/16 (75%) | 1ms |
| ShenlunForge Shenlun | 4817 | 58% | 16/16 (100%) | 48933ms |

## 原始数据文件
- 单场景报告: `experiment_report_20260801_201612.md`
- 单场景JSON: `experiment_data_20260801_201612.json`
- 批量汇总报告: `batch_summary_20260801_202714.md`
- 批量JSON: `batch_data_20260801_202714.json`
- 各场景独立报告: `report_q1_fact_recall_*.md` ~ `report_q6_rejected_alternatives_*.md`
