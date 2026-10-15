# ShenlunForge

**面向 LLM Agent 的任务驱动型上下文压缩。**

对话历史越来越长。ShenlunForge 接收你准备发给模型的消息列表，返回一个更小的
列表，并且保留*当前这个问题*需要的信息——不是无脑截断，也不是通用摘要。

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create()                 # 纯规则，不需要 API key
result = forge.compress_sync(messages=history, current_query=question, max_tokens=800)

print(result.compressed_text)                 # 把这个发出去
```

[English](README.md) · 简体中文

如果这个库帮到了你，[给颗星](https://github.com/liurongchen1/shenlun-forge)是最好的回礼。

---

## 适合谁用

- **Agent 开发者**：`messages` 已经塞不进上下文窗口，或者成本明显偏高。
  在每次调用模型之前，针对即将提出的问题压缩一次。
- **只想把长对话压小的人**：不想为此再付一份 API 费用。规则策略完全本地运行，
  毫秒级，零成本。
- **中文以及中英混杂对话**：按词边界分词的方案和面向英文调优的压缩工具，
  在这类文本上表现通常很差。

## 安装

```bash
pip install shenlun-forge
```

包含规则策略和命令行工具。按需安装可选依赖：

```bash
pip install "shenlun-forge[openai]"      # 启用 acs / hcs 策略
pip install "shenlun-forge[langchain]"   # LangChain 适配器
pip install "shenlun-forge[llamaindex]"  # llama-index 消息对象兼容
pip install "shenlun-forge[embedding]"   # 本地向量检索
pip install "shenlun-forge[full]"        # 全部
```

需要 Python 3.9+。

## 30 秒上手，不需要 API key

```python
from shenlun_forge import ShenlunForge

history = [
    {"role": "user", "content": "订座系统的数据库用哪个？"},
    {"role": "assistant", "content": "数据库用 PostgreSQL，排除了 MongoDB。"},
    {"role": "user", "content": "确认邮件怎么发？"},
    {"role": "assistant", "content": "用 Resend，每种语言一套模板。"},
    # ... 后面还有 40 轮 ...
]

forge = ShenlunForge.create()          # 默认就是 simple
result = forge.compress_sync(
    messages=history,
    current_query="我们最后定了哪个数据库？",
    max_tokens=500,
)

print(result.compression_ratio)        # 0.31，省掉约 69% 的 token
print(result.compressed_text)
```

命令行同理：

```bash
shenlun-forge compress --messages chat.json --query "我们最后定了哪个数据库？"
```

**想跟着一步步做？** 完整教程（从"历史装不下"到"知道该配什么"）：
[`docs/tutorial.md`](docs/tutorial.md)

更多：快速开始 [`docs/quickstart.md`](docs/quickstart.md) ·
可跑示例 [`examples/quickstart_no_llm.py`](examples/quickstart_no_llm.py)

## 该选哪个策略

| 你的情况 | 策略 | 原因 |
|---|---|---|
| 历史很长，答案需要跨消息推理 | `acs` | 会先读问题，再决定保留什么 |
| 代码密集，标识符必须原样保留 | `hcs` ⚠️ | ACS 摘要 + 代码块逐字保留——但实测并不优于免费方案，见下文 |
| 想省 token 又不想用 API key | `simple`（默认） | 纯规则、本地、毫秒级 |
| 时间线、待办清单、配置等结构化信息 | `simple` | 规则能保住离散条目，摘要会把它们揉碎 |
| 问题和答案用词完全不重合 | `acs` | 词法匹配无法从"耗电快"跳到"BMS 固件 bug" |
| 你只需要最近 N 轮 | 不需要这个库 | 直接截断更简单 |
| 你要在大量文档里检索 | RAG，不是这个 | 解决的是另一个问题 |

一句话：**`simple` 是过滤器，`acs` 是读者。** 问题和历史用词重合时，`simple`
通常够用；答案分散且措辞与问题不同时，才需要 `acs`。

完整判断依据见 [`docs/scenarios.md`](docs/scenarios.md)。

## 接上 LLM

`acs` 和 `hcs` 需要任意 OpenAI 兼容接口——DeepSeek、OpenAI、Groq、vLLM、
Ollama 的 OpenAI 兼容层都可以。

```python
forge = ShenlunForge.create(
    strategy="acs",
    model="deepseek-chat",
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
)

result = await forge.compress(
    messages=history,
    current_query="我们在这个项目里做了哪些安全措施？",
    max_tokens=4000,
)
```

`hcs` 在其之上额外保留代码块原文——当摘要会把 `verify_password` 写成
"实现了认证"的时候，这一层很关键。

参见 [`examples/compress_before_llm.py`](examples/compress_before_llm.py) 与
[`docs/integration.md`](docs/integration.md)。

## 到底该不该压？

压缩是有损的，所以第一个问题是"需不需要压"。回答它是免费的——不需要 API key、
不调用 LLM、不需要实例：

```python
from shenlun_forge import plan_compression

plan = plan_compression(messages=history, model_window=128_000)

if plan.fits_in_window:
    send(history)                                  # 什么都不用做
else:
    result = await forge.compress(
        messages=history,
        current_query=query,
        max_tokens=plan.recommended_max_tokens,    # 用满可用窗口
    )
```

`plan.advice` 会用一句话说明判断依据。之所以建议先问这一步：在我们的实测里，
预算给得太紧会让相当一部分问题**直接答不出来**，而质量随预算持续改善——所以很少
有理由压到窗口允许的范围以下。

如果你已经有实例，`ShenlunForge.plan()` 是同样的功能，只是做成方法。

## 怎么读返回值

```python
result.compressed_text          # 发给模型的字符串
result.original_tokens          # 原始 token
result.compressed_tokens        # 压缩后 token
result.compression_ratio        # 压缩后 / 原始
result.blocks                   # 保留下来的消息及其原始位置

result.stats["passthrough"]     # True 表示没压出效果，原样返回
result.stats["degraded"]        # True 表示没有按配置真正跑起来
result.stats["degraded_reason"] # 具体原因
result.stats["llm_calls"]       # 实际发起了几次 LLM 请求
result.stats["llm_failures"]    # 其中失败几次
result.stats["over_budget"]     # True 表示结果超出了 max_tokens
result.stats["recent_truncated"]# True 表示最近的消息被裁剪过
result.stats["dropped_messages"]# 有多少条历史没进摘要
result.stats["dropped_tokens"]  # 这些被丢掉的历史大概多少 token
```

四个行为是刻意设计的，建议先知道：

- **`max_tokens` 是上限，不是目标。** 如果只有四条消息和问题相关，那就只保留
  四条。用无关内容把预算填满只会让上下文更差。
- **压不动就原样返回。** 如果历史不足 200 token，或者压缩省不到 5%，会原封不动
  返回输入，并置 `passthrough: True`。返回一个比输入*更长*的结果，比不压缩更糟。
- **最后一轮优先。** 如果最后一条消息本身就装不进预算，默认会整条保留它——丢掉
  当前正在进行的这一轮通常比略微超出预算更糟——并把 `over_budget` 置为 `True`
  如实告诉你。如果 `max_tokens` 对你是硬约束（按它核算成本，或目标模型的窗口
  正好这么大），传 `budget_overflow="truncate"`：它会截断最近的消息以守住预算，
  并置 `recent_truncated: True`。两条路都会在返回值里明示，不会默默替你决定。
- **它知道自己没装下什么。** 摘要末尾会跟一行，说明还有多少条较早历史不在里面。
  见下一节。

### 摘要会自己交代缺口

```
…（摘要正文）

【压缩说明】另有 37 条较早历史未包含在上文（约 12,480 tokens）。如果这里找不到答案，请说明缺口，不要臆测。
```

模型不知道压缩器丢过东西，它只知道你给了它一段历史。所以当答案正好在被丢掉的那部分里
时，它会说“上下文里没有相关信息”——而它其实分不清“没有”和“我没看到”。这一行把它从
“我读完了全部历史”纠正回“我只读了其中一部分，缺口在这里，可以去找”。

默认开启，语言跟随你的提问（中文问题给中文提示，英文问题给英文提示）。它占用的 token
算在 `max_tokens` 里，不会偷偷把结果撑破预算。不需要这行字就传 `dropped_notice=False`，
命令行对应 `--no-dropped-notice`；无论开关，`stats["dropped_messages"]` 与
`stats["dropped_tokens"]` 都会照常上报，你可以自己决定怎么记这笔账。

实测（同一批点开/关对照、按对话聚类）：摘要还像一份完整历史（保留历史 20–50%）时，
它值 +10pp 左右的回答质量；压到只剩 4% 时，没有可测量的影响。它没有让模型更多地说
“缺失”，也没有显著减少编答案——机制没测出来，就不要替它编一个。

## 命令行

```bash
# 纯规则，不需要 key（默认）
shenlun-forge compress --messages chat.json --query "我们最后决定了什么？"

# 从 stdin 读，输出到文件
cat chat.json | shenlun-forge compress --stdin --query "总结一下" -o out.json

# 接 LLM，凭据走环境变量
export SHENLUN_FORGE_API_KEY="sk-..."
export SHENLUN_FORGE_BASE_URL="https://api.deepseek.com/v1"
shenlun-forge compress -m chat.json -q "做了哪些安全措施？" -s acs -v

shenlun-forge strategies
shenlun-forge version

# 用你自己的对话测一测压缩到底有没有用
shenlun-forge eval --conversations ./my-chats --strategies truncate,simple
```

## 原理

LLM 流水线（`acs`）是一次四阶段压缩，参照分析师处理材料的方式：

| 阶段 | 做什么 |
|---|---|
| **审题** | 读问题，判断哪些类型的信息重要 |
| **找点** | 给每条消息打分——关键词重合、信息类型、时效、角色、信号词、信息密度 |
| **加工** | 合并重复内容、给每组提炼标签、保护标识符（函数名、常量、路径、`3天` 这类约束） |
| **书写** | 按优先级贪心填满 token 预算 |

规则路径（`simple`）复用同一套检索打分，但不调用任何 LLM，然后仅用与问题相关的
消息填充预算。两条路径都会原样保留最近若干轮对话——当前任务状态在那里。

中文按 2-gram 滑窗匹配，英文术语按在对话中的稀有度（IDF）加权，并做中英桥接，
让中文提问能找到回答它的英文标识符。

## 评测

可用 [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py) 复现：
一段中英混杂的框架开发对话（18 条消息，约 5.4k token），8 组问题，
132 个答案要点，预算 1200 token。事实保留率 = 答案要点在压缩结果中被找到的比例。

| 策略 | 命中 | 保留率 |
|------|------|--------|
| 不压缩（上界） | 129/132 | 97.7% |
| 截断保留后 6 条 | 38/132 | 28.8% |
| 滑动窗口 | 38/132 | 28.8% |
| `simple`（规则） | **111/132** | **84.1%** |
| `acs`（LLM） | 72/132 | 54.5% |
| `hcs`（LLM + 代码） | 68/132 | 51.5% |

这张表有个不太好看但必须说清楚的结论：**按"字面事实保留"算，免费的策略胜过
LLM 流水线。** 这不是 bug。`simple` 直接复制原消息片段，保留的标识符与答案要点
逐字一致；`acs` 会合并改写，字面子串匹配自然吃亏。LLM 流水线的价值在别处——
当问题的用词在原文里根本不出现时，早期内部实验显示排序是反过来的。

所以：**先用 `simple`**，当你发现它在"换一种说法提问"时明显答不出来，再上 `acs`。
分场景明细和注意事项见 [`benchmarks/README.md`](benchmarks/README.md)。

## 压缩之后，模型还答得出问题吗？

事实保留率衡量的是"用词"，不是"有没有用"。所以另做了一套问答评测：把历史压缩后，
让模型**只看压缩结果**回答问题，再拿它的回答与真实答案比对。工具在
[`benchmarks/qa_eval.py`](benchmarks/qa_eval.py)，细节与注意事项见
[`benchmarks/README.md`](benchmarks/README.md)。

**能不能测出差异，取决于你的数据里有没有差异。**

**在真实 Agent 工作负载上**（131 个测试点、12 段对话、单段 3k–69k token），
4000 token 预算下两者打平——规则策略 39.8%，粗暴截断 39.3%。因为那批数据里答案
大多就在历史末尾，"保留最近的尾巴"本来就已经接近最优。

**在 [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)**
（公开数据集，MIT 许可：每个问题约 10 万 token 历史、答案埋在大约 48 个 session
之一、96 个互相独立的提问）上，16000 token 预算下：

| 检索方式 | 正确率 |
|----------|--------|
| 粗暴截断 | 19.8% |
| `simple`（关键词匹配） | 30.2% |
| **`embedding`（语义检索，`retriever_type="embedding"`）** | **44.8%** |

三者两两差异都显著（McNemar p = 0.021、0.0094、<0.001）。**换检索器带来的提升，
超过这个库里任何一个调参维度**——超过 token 预算、超过合并策略、超过最近消息份额，
后三者在我们的实测里都只是噪声。语义检索也是唯一能让时间推理类问题拿到分数的
配置（25%，另外两种都是 0%）。

按问题类型：

| 问题类型 | 截断 | 关键词 | 语义 |
|----------|------|--------|------|
| 回忆助手说过的事实 | 19% | 25% | **62%** |
| 时间推理 | 0% | 0% | **25%** |
| 知识更新（"我现在用的 X"） | 75% | 62% | **81%** |
| 跨会话 | 12% | 19% | **38%** |
| 回忆**用户本人**说过的事实 | 12% | **69%** | 56% |

这比"压得好"给出了更有用的使用规则：

- **长历史里要找回早前说过的信息** → 用 `retriever_type="embedding"`，这是它明显胜出的场景。
- **问题就在复述用户自己的原话** → 关键词匹配仍然更好（69% vs 56%），字面重合在那里是强信号。
- **"最新值就是答案"的场景不要用压缩**：知识更新类问题截断更便宜也更好。
- **短对话** → 规则路径快得多，而且差异消失。

### 摘要末尾那句缺口提示，有用吗？

「另有 N 条较早历史未包含在上文」默认开启，也单独做了开/关对照实验——同一批点配对
比较，显著性按对话聚类（同一个对话里的测试点不是相互独立的样本）：

| 数据 | 开 | 关 | 结论 |
|---|---|---|---|
| LongMemEval，96 条独立长历史 | 11.5% | 12.5% | 打平（p=0.32）：压掉 96% 时，模型本来就看得出来上下文是残片 |
| 真实工作对话，131 点 / 12 个对话 | 46.3% | 40.9% | +5.4pp（p=0.089）；**保留 20–50% 历史的那一档 +10.4pp**（p=0.034） |
| 聊天记录，163 点 / 29 个对话 | 40.7% | 36.7% | +4.1pp（p=0.026）；同一档 +11.6pp（p=0.005） |

一句话：**摘要看起来还像一份完整历史时，这句话有用；压得只剩片段时，它没用。**
20–50% 这一档是在第一个数据集上发现的，然后在还没跑过的第二个数据集上复现，
所以它值得写下来，也仅此而已——不要在别的压缩率上引用这个数字。

还要如实说明一件事：我原本以为它能防幻觉，**实测没测出来**。开与关两种情况下，
"答了但没答到要点"的比例都在 0%–3%，模型并没有因为看不到缺口就开始编。
而且模型说"上下文里没有相关信息"的比例几乎不变（71.8% vs 68.7%、76.7% vs 75.5%、
87.5% vs 87.5%）。所以它改变的是模型怎么用手里这点材料，不是手里有多少材料——
机制没有被测出来，就不要替它编一个。细节与复现命令见
[`benchmarks/README.md`](benchmarks/README.md)。

## 成本

没有价格标签的准确率不能用来做决策。在一段 10 万 token 的历史上、预算
16000 token 实测（`benchmarks/cost_benchmark.py`）：

| 策略 | 中位延迟 | 输出 token | LLM 调用 |
|------|---------|-----------|---------|
| 截断 | **5 ms** | 15,900 | 0 |
| `simple` | **41 ms** | 6,519 | 0 |
| `embedding` | **109 ms** | 9,607 | 0 |
| `acs` | 3,298 ms | 9,579 | 1 |
| `hcs` | 3,801 ms | 6,519 | 0 |

免费路径是毫秒级，LLM 策略是秒级。注意**语义检索只比关键词匹配多约 60 毫秒**——
在一次模型调用面前可以忽略。"embedding 太贵"不是跳过它的好理由。

## 用你自己的对话来测

本文档里所有数字都来自同一份数据，而做这些实验最大的教训就是：**结论完全取决于你的
数据**。在普通工作对话上，压缩与截断打平；在"答案埋在长历史早期"的数据上，它领先
25 个百分点。没人能告诉你属于哪种情况——所以让库自己测：

```bash
shenlun-forge eval --conversations ./my-chats --strategies truncate,simple,embedding
```

```
从 20 个对话构建测试（提取答案要点需要模型调用）...
得到 84 个测试点，开始评测...

Accuracy by strategy (budget 16000 tokens, n=84 points):
  embedding             44.1%
  simple                30.6%
  truncate              21.4%

Best: embedding. ...
```

也可以用 Python：

```python
from shenlun_forge import OpenAIProvider, build_eval_set, evaluate

llm = OpenAIProvider(model="deepseek-chat", api_key="sk-...")
points = await build_eval_set(my_conversations, llm=llm)
report = await evaluate(points, strategies=["truncate", "simple"], llm=llm)
print(report.summary())
```

它会压缩每一段历史，让模型**只看压缩结果**回答，再拿回答与该问题当时真实回复里提取的
要点比对。要点只提取一次并冻结——如果在打分时才提取，每次运行都是在用不同尺子量。

信任结果前请记住两点：**在这个样本量下，小于 5 个百分点的差异是噪声**；而且独立的
统计单位是**对话**而不是问题——往已有对话里加更多轮次不会让结论更可靠。另外，
提要和判分都要调用模型，所以这会把你自己的对话内容发给服务商。

## 现状与边界

- `acs` 每次压缩要一到两次 LLM 调用。这是它的定位：需要"读懂"时才走这条贵路径。
- 降级会被如实上报，不会藏起来，但也不会直接抛错。生产环境请检查
  `stats["degraded"]`。
- `v2` 的设计文档（策略推理、阶段感知、多维质量评估）目前只是设计稿，代码里
  尚未实现。
- **不推荐 `hcs`。** 在它唯一该发挥作用的场景——含 2.6 万条命令执行与文件 diff 的
  中文会话、45 个测试点——它与截断和关键词匹配打成平手（45.9% vs 46.0% / 49.6%，
  p=1.00）。六次评测里它从未显著更好，却要多一次 LLM 调用。代码保留，推荐取消。
- **自动选检索器不可行，而且这是实测而非猜测。** `recommend_retriever()` 选中更优
  策略的概率只有 41%。更深层的原因写在 `benchmarks/README.md`：答案要么是随机分布的
  （无法预测，而"总是检索"本就是最优），要么靠近末尾（两种选择打平，没有收益可拿）。
  请自己按场景选检索器。

## 文档

- **[`docs/tutorial.md`](docs/tutorial.md) —— 端到端教程（推荐从这里开始）**
- [`docs/quickstart.md`](docs/quickstart.md) —— 从安装到第一个结果
- [`docs/scenarios.md`](docs/scenarios.md) —— 什么时候该用、什么时候不该用
- [`docs/integration.md`](docs/integration.md) —— OpenAI SDK、LangChain、自定义阶段
- [`docs/faq.md`](docs/faq.md) —— 降级、passthrough、成本、中文支持
- [`benchmarks/README.md`](benchmarks/README.md) —— 评测数字是怎么测出来的
- [`RELEASING.md`](RELEASING.md) —— 发布清单
- [`CHANGELOG.md`](CHANGELOG.md)

## 参与贡献

按价值排序：其他框架的适配器（AutoGen、CrewAI）、给 `benchmarks/` 加一层语义
判分（当前的字面子串指标测不出 LLM 流水线的价值）、代码级细节的更好处理、
中文/英文之外的表达规范化映射。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。

## 引用

```bibtex
@software{shenlunforge2026,
  title={ShenlunForge: Task-Driven Context Compression for LLM Agents},
  author={Liu, Rongchen},
  year={2026},
  url={https://github.com/liurongchen1/shenlun-forge}
}
```
