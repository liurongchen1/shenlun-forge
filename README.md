# ShenlunForge

**面向大语言模型智能体的任务驱动型上下文压缩工具**
**Task-driven context compression for LLM agents**

在把对话历史发给模型之前压一遍，只保留当前这个问题需要的信息。
Compress the message list before sending it, keeping what the current question needs.

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create()                 # 规则策略，不需要 API key
result = forge.compress_sync(messages=history, current_query=question, max_tokens=800)

print(result.compressed_text)                 # 把这个发出去
```

[![PyPI](https://img.shields.io/pypi/v/shenlun-forge.svg)](https://pypi.org/project/shenlun-forge/)
[![Python](https://img.shields.io/pypi/pyversions/shenlun-forge.svg)](https://pypi.org/project/shenlun-forge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

如果这个库帮到了你，[给颗星](https://github.com/liurongchen1/shenlun-forge)是最好的回礼。
If it saved you some tokens, [a star](https://github.com/liurongchen1/shenlun-forge) is the nicest way to say thanks.

双语 README（中英对照）· 纯中文版见 [README.zh-CN.md](README.zh-CN.md)

---

## 适合谁用 / Who this is for

- **写 Agent 的开发者**：`messages` 已经塞不进上下文窗口，或者成本偏高。在每次调用模型前，针对即将提出的问题压缩一次。
- **只想把长对话压小的人**：不想为此再付一份 API 费用。规则策略完全本地运行，毫秒级，零成本。
- **中文以及中英混杂对话**：按词边界分词的方案和面向英文调优的压缩工具，在这类文本上通常表现很差。

- **Agent developers** whose message list no longer fits the window, or costs more than it should. Compress before each call, for the question you are about to ask.
- **Anyone with a long conversation to shrink** and no interest in paying for an API to do it. The rule-based path is local, free and runs in milliseconds.
- **Chinese and mixed Chinese/English conversations**, where word-boundary tokenizers and English-tuned compressors tend to do badly.

## 安装 / Install

```bash
pip install shenlun-forge                                  # 规则策略 + CLI
pip install "shenlun-forge[openai]"                        # acs / hcs 策略（需要 LLM）
pip install "shenlun-forge[langchain]"                     # LangChain 适配器
pip install "shenlun-forge[llamaindex]"                    # llama-index 消息兼容
pip install "shenlun-forge[embedding]"                     # 本地语义检索
pip install "shenlun-forge[full]"                          # 全部
```

需要 Python 3.9+。 / Requires Python 3.9+.

## 30 秒上手，不需要 API key / 30 seconds, no API key

想先看一眼它到底在干什么——不用 key、不联网、不到一秒：

```bash
shenlun-forge demo          # 英文输出；中文加 --lang zh
```

它把同一段对话分别交给「粗暴截断」和本库，然后告诉你各自留下了哪几条消息。差别就在
这里：截断永远是那根尾巴，而这里留下来的是问题需要的那几条。

The same thing in English: `shenlun-forge demo` hands one conversation to plain
truncation and to this library, then shows which messages each one kept.
Truncation always returns the same tail; here the survivors depend on the
question.

```python
from shenlun_forge import ShenlunForge

history = [
    {"role": "user", "content": "订座系统的数据库用哪个？"},
    {"role": "assistant", "content": "数据库用 PostgreSQL，排除了 MongoDB。"},
    # ... 后面还有几十轮 ...
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

命令行同理： / Or from the shell:

```bash
shenlun-forge compress --messages chat.json --query "我们最后定了哪个数据库？"
```

**想跟着一步步做？** 完整教程（从"历史装不下"到"知道该配什么"）：
[`docs/tutorial.md`](docs/tutorial.md)

更多：快速开始 [`docs/quickstart.md`](docs/quickstart.md) ·
可跑示例 [`examples/quickstart_no_llm.py`](examples/quickstart_no_llm.py)

## 该选哪个策略 / Which strategy

| 你的情况 / Situation | 策略 | 原因 |
|---|---|---|
| 历史很长，答案需要跨消息推理 / Long history, answer needs reasoning | `acs` | 会先读问题，再决定保留什么 / reads the query first |
| 代码密集，标识符必须原样保留 / Code-heavy | `hcs` ⚠️ | 摘要 + 代码块逐字保留。**但实测并不优于免费方案，见下文** |
| 想省 token 又不想用 API key / No API key | `simple`（默认） | 纯规则、本地、毫秒级 |
| 时间线、待办、配置等结构化信息 / Structured data | `simple` | 规则能保住离散条目，摘要会把它们揉碎 |
| 问题和答案用词完全不重合 / No shared vocabulary | `acs` | 词法匹配无法从"耗电快"跳到"BMS 固件 bug" |
| 你只需要最近 N 轮 / Only the last N turns | 什么都不用 | 直接截断更简单 |
| 你要在大量文档里检索 / Retrieval over many documents | RAG，不是这个 | 解决的是另一个问题 |

一句话：**`simple` 是过滤器，`acs` 是读者。** 问题和历史用词重合时 `simple` 通常够用；
答案分散且措辞不同时才需要 `acs`。

The short version: **`simple` is a fast filter, `acs` is a reader.** If your query and
your history share words, `simple` is usually enough. If the answer is scattered and
phrased differently from the question, you want `acs`.

## 接上 LLM / With an LLM

`acs` 和 `hcs` 需要任意 OpenAI 兼容接口——DeepSeek、OpenAI、Groq、vLLM、Ollama 都行。

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

`acs` and `hcs` work with any OpenAI-compatible endpoint.
See [`examples/compress_before_llm.py`](examples/compress_before_llm.py) and
[`docs/integration.md`](docs/integration.md).

## 到底该不该压？ / Should you compress at all?

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

`plan.advice` 会用一句话说明判断依据。之所以建议先问这一步：实测里预算给得太紧会让
相当一部分问题**直接答不出来**。

Compression is lossy, so check whether you need it first. That costs nothing: no API
key, no LLM call. `plan.advice` explains the decision in one sentence.

## 怎么读返回值 / Reading the result

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

四个刻意设计的行为： / Four deliberate behaviours:

- **`max_tokens` 是上限，不是目标**：只有四条消息和问题相关，就只保留四条，不会用无关
  内容把预算填满。
- **压不动就原样返回**：省不到 5%，或历史不足 200 token，会原封不动返回输入。
- **最后一轮优先**：最后一条消息自己就装不进预算时，默认整条保留（信息完整优先）并置
  `over_budget`。如果 `max_tokens` 对你是硬约束，传 `budget_overflow="truncate"`。
- **它会告诉你它没装下什么**：摘要末尾跟一行“另有 N 条较早历史未包含在上文”。见下。

### 摘要会自己交代缺口 / The summary says what is missing

```
…（摘要正文）

【压缩说明】另有 37 条较早历史未包含在上文（约 12,480 tokens）。如果这里找不到答案，请说明缺口，不要臆测。
```

模型不知道压缩器丢了东西，它只知道你给了它一段历史：当答案正好在被丢掉的那部分里时，
它会说“上下文里没有相关信息”，而它其实无法区分“没有”和“我没看到”。这一行把它从
“读完了全部历史”纠正回“只读了其中一部分”。

默认开启，语言跟随你的提问（中文问题给中文提示，英文问题给英文提示），占用的 token
算在 `max_tokens` 里，不会偷偷把结果撑破预算。不需要就传 `dropped_notice=False`；
无论开关，`stats["dropped_messages"]` 和 `stats["dropped_tokens"]` 都会照常上报，
你可以自己决定怎么记这笔账。命令行是 `--no-dropped-notice`。

实测（同一批点开/关对照、按对话聚类）：摘要还像一份完整历史（保留 20–50%）时，
它值 +10pp 左右的回答质量；压到只剩 4% 时，它没有可测量的影响。它并没有让模型
更多地说“缺失”，也没有显著减少编答案——机制没测出来，就不要替它编一个。

A model handed a lossy summary does not know it is lossy, so it cannot tell
"the answer is not there" apart from "I was never shown it". This line tells it
which part of the conversation it never saw. On by default, written in the
language of the question, sized inside your `max_tokens` budget. Turn it off
with `dropped_notice=False` (`--no-dropped-notice` on the CLI); the gap is still
reported in `stats` either way.

## 命令行 / CLI

```bash
# 先看它到底在干什么（内置对话，不需要 key）
shenlun-forge demo            # 中文；英文用 --lang en

# 纯规则，不需要 key（默认）
shenlun-forge compress --messages chat.json --query "我们最后决定了什么？"

# 从 stdin 读，输出到文件
cat chat.json | shenlun-forge compress --stdin --query "总结一下" -o out.json

# 接 LLM（key 和地址走环境变量）
export SHENLUN_FORGE_API_KEY="sk-..."
export SHENLUN_FORGE_BASE_URL="https://api.deepseek.com/v1"
shenlun-forge compress -m chat.json -q "做了哪些安全措施？" -s acs -v

# 用你自己的对话测一测压缩到底有没有用
shenlun-forge eval --conversations ./my-chats --strategies truncate,simple

shenlun-forge strategies
shenlun-forge version
```

只设了 `DEEPSEEK_API_KEY` 时，命令行会自动使用 DeepSeek 的地址。
Setting only `DEEPSEEK_API_KEY` makes the CLI pick DeepSeek's endpoint automatically.

## 原理 / How it works

LLM 流水线（`acs`）是一次四阶段压缩，参照分析师处理材料的方式：

| 阶段 | 做什么 |
|---|---|
| **审题** | 读问题，判断哪些类型的信息重要 |
| **找点** | 给每条消息打分——关键词重合、信息类型、时效、角色、信号词、信息密度 |
| **加工** | 合并重复、给每组提炼标签、保护标识符（函数名、常量、路径、`3天` 这类约束） |
| **书写** | 按优先级贪心填满 token 预算 |

规则路径（`simple`）复用同一套检索打分，但不调用任何 LLM。两条路径都会原样保留最近
若干轮对话。中文按 2-gram 滑窗匹配，英文术语按稀有度加权，并做中英桥接。

The LLM pipeline (`acs`) is four stages: analyze, retrieve, process, format. The
rule-based path (`simple`) uses the same retrieval scoring with no LLM call at all.
Chinese is matched with 2-character n-grams and English terms are weighted by rarity
across the conversation.

## 成本 / What it costs

没有价格标签的准确率不能用来做决策。在一段 10 万 token 的历史、预算 16000 token 上
实测（`benchmarks/cost_benchmark.py`）：

| 策略 | 中位延迟 | 输出 token | LLM 调用 |
|---|---|---|---|
| 截断 / truncate | **5 ms** | 15,900 | 0 |
| `simple` | **41 ms** | 6,519 | 0 |
| `embedding` | **109 ms** | 9,607 | 0 |
| `acs` | 3,298 ms | 9,579 | 1 |
| `hcs` | 3,801 ms | 6,519 | 0 |

免费路径是毫秒级，LLM 策略是秒级。注意**语义检索只比关键词匹配多约 60 毫秒**——
在一次模型调用面前可以忽略。"embedding 太贵"不是跳过它的好理由。

The free paths are milliseconds; the LLM strategies are seconds. Semantic retrieval
costs about 60 ms more than keyword matching — negligible next to one model call.

## 压缩之后，模型还答得出问题吗？ / Does it actually help?

事实保留率衡量的是"用词"，不是"有没有用"。所以另做了一套问答评测：压缩后让模型
**只看压缩结果**回答，再与真实答案比对。

**关键发现：能不能测出差异，取决于你的数据里有没有差异。**

| 数据 | 答案位置 | 结论 |
|---|---|---|
| LongMemEval（公开，96 个独立长历史） | 随机埋在大约 48 个 session 之一 | 截断 19.8% → 关键词 30.2% → **语义 44.8%**，三者两两显著 |
| 真实工作对话（两批私有数据） | 靠近末尾 | 三种方式打平——答案就在末尾时，保留尾部本已最优 |

还有两个结论值得记住：

- **换检索器的影响超过所有调参**：预算、合并策略、最近窗口份额，实测全是噪声。
- **自动选检索器不可行**，而且是结构性的：答案要么随机（无法预测，且"总是检索"
  就已最优），要么靠末尾（两种选择打平，没有收益可拿）。

摘要末尾那句「另有 N 条较早历史未包含在上文」也单独测过（同一批点，开/关对照，
按对话聚类做显著性检验）：

| 数据 | 开 | 关 | 结论 |
|---|---|---|---|
| LongMemEval，96 条独立长历史 | 11.5% | 12.5% | 打平（p=0.32）——压掉 96% 时，模型本来就看得出来上下文是残片 |
| 真实工作对话，131 点 / 12 个对话 | 46.3% | 40.9% | +5.4pp；**保留 20–50% 历史的那一档是 +10.4pp**（p=0.034） |
| 聊天记录，163 点 / 29 个对话 | 40.7% | 36.7% | +4.1pp（p=0.026）；同样，20–50% 那一档 +11.6pp（p=0.005） |

一句话：**摘要看起来像一份完整历史时，这句话有用；压得只剩下片段时，它没用。**
另外要如实说，我原本以为它防幻觉，结果没测出来——两种情况下模型"答了但没答到点"
的比例都是 0%–3% 左右。细节和复现命令见
[`benchmarks/README.md`](benchmarks/README.md)。

The same result in one sentence: the note pays for itself when the compressed
history still reads like a history (a 20–50% ratio), and costs nothing
measurable when it does not. Its original justification — that a model invents
answers when the context is incomplete — did not survive measurement.

By question type on LongMemEval (16k budget, n=16 each):

| 问题类型 | 截断 | 关键词 | 语义 |
|---|---|---|---|
| 回忆助手说过的事实 | 19% | 25% | **62%** |
| 时间推理 | 0% | 0% | **25%** |
| 知识更新（"我现在用的 X"） | 75% | 62% | **81%** |
| 回忆**用户本人**说过的事实 | 12% | **69%** | 56% |

使用规则：

- **长历史里要找回早前说过的信息** → 用 `retriever_type="embedding"`。
- **问题在复述用户自己的原话** → 关键词匹配仍更好（69% vs 56%）。
- **"最新值就是答案"** → 干脆别压，用 `plan_compression()` 先判断。

## 用你自己的对话来测 / Measure it on your own conversations

本 README 的所有数字都来自特定数据，而最大的教训就是**结论取决于你的数据**。
所以让库自己测：

```bash
shenlun-forge eval --conversations ./my-chats --strategies truncate,simple,embedding
```

```python
from shenlun_forge import OpenAIProvider, build_eval_set, evaluate

llm = OpenAIProvider(model="deepseek-chat", api_key="sk-...")
points = await build_eval_set(my_conversations, llm=llm)
report = await evaluate(points, strategies=["truncate", "simple"], llm=llm)
print(report.summary())
```

它会压缩每段历史，让模型只看压缩结果回答，再与该问题当时真实回复里提取的要点比对。
要点只提取一次并冻结——如果在打分时才提取，每次运行就是用不同尺子量。

两点请记住：在这个样本量下**小于 5 个百分点的差异是噪声**；独立的统计单位是
**对话**而不是问题。另外提要和判分都要调用模型，**会把你的对话发给服务商**。

## 现状与边界 / Status and limitations

- `acs` 每次压缩要一到两次 LLM 调用。这是它的定位：需要"读懂"时才走这条贵路径。
- 降级会被如实上报，不会藏起来，但也不会直接抛错。生产环境请检查 `stats["degraded"]`。
- **不推荐 `hcs`**：在含 2.6 万条命令执行与文件 diff 的中文会话上（45 个测试点），
  它与截断、关键词匹配打成平手（45.9% vs 46.0% / 49.6%，p=1.00），却要多一次 LLM 调用。
  六次评测里从未显著更好。
- **自动选检索器不可行**（`recommend_retriever()` 只选中更优策略 41% 的时间）。
  请自己按场景选。原因写在 [`benchmarks/README.md`](benchmarks/README.md)。
- `v2` 的设计文档（策略推理、阶段感知、多维质量评估）只是设计稿，代码里尚未实现。

## 文档 / Documentation

- **[`docs/tutorial.md`](docs/tutorial.md) —— 端到端教程（推荐从这里开始）**
- [`docs/quickstart.md`](docs/quickstart.md) —— 从安装到第一个结果
- [`docs/scenarios.md`](docs/scenarios.md) —— 什么时候该用、什么时候不该用
- [`docs/integration.md`](docs/integration.md) —— OpenAI SDK、LangChain、自定义阶段
- [`docs/faq.md`](docs/faq.md) —— 降级、passthrough、成本、中文支持
- [`benchmarks/README.md`](benchmarks/README.md) —— 数字是怎么测出来的
- [`RELEASING.md`](RELEASING.md) —— 发布清单
- [`CHANGELOG.md`](CHANGELOG.md)

## 参与贡献 / Contributing

按价值排序：其他框架的适配器（AutoGen、CrewAI）、给 `benchmarks/` 加一层语义判分、
代码级细节的更好处理、中文/英文之外的表达规范化映射。

Most useful: adapters for other frameworks, a semantic scoring pass for `benchmarks/`,
better handling of code-level detail, and normalisation maps beyond Chinese/English.

## 许可证 / License

MIT —— 见 [LICENSE](LICENSE)。

## 引用 / Citation

```bibtex
@software{shenlunforge2026,
  title={ShenlunForge: Task-Driven Context Compression for LLM Agents},
  author={Liu, Rongchen},
  year={2026},
  url={https://github.com/liurongchen1/shenlun-forge}
}
```
