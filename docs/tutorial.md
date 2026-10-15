# 教程：把它接进你的 Agent（完整走一遍）

这份文档是一条完整路径，从「我的历史快装不下了」到「我知道该配什么参数、为什么这么配」。

每一节也能单独看，但按顺序读最省事。全程大约 20 分钟，其中大部分是在等 API。

---

## 起点：一个会超窗口的 Agent

假设你的 Agent 每轮把整个 `messages` 发给模型，现在历史到了 6 万 token，既接近窗口
上限，成本也在涨。

```python
messages = [
    {"role": "system", "content": "你是一个资深后端助手"},
    {"role": "user", "content": "我们要做一个餐厅订座系统，先聊技术选型"},
    # ... 后面还有两百多条 ...
]
current_question = "我们最后定的数据库和认证方案是什么？"
```

目标：发给模型之前压一次，只留下回答**这个问题**需要的部分。

## 第一步：先判断要不要压（免费、不需要 key）

很多人会跳过这一步直接压。别跳——**压缩是有损的，能压之前先确认你非压不可**：

```python
from shenlun_forge import plan_compression

plan = plan_compression(messages=messages, model_window=128_000)

print(plan.fits_in_window)   # False → 必须压
print(plan.recommended_max_tokens)
print(plan.advice)
```

两种结果：

- `fits_in_window=True` → **别压**，直接发。压了只会丢信息，省不下任何东西。
- `fits_in_window=False` → 需要压，`recommended_max_tokens` 就是该给的预算。

这一步不调用 LLM、不需要 API key、耗时是毫秒级，可以放在 Agent 循环最前面。

## 第二步：压一次，看看效果（还是不需要 key）

先用规则策略，零成本：

```python
from shenlun_forge import ShenlunForge

forge = ShenlunForge.create()          # 默认就是 simple
result = forge.compress_sync(
    messages=messages,
    current_query=current_question,
    max_tokens=plan.recommended_max_tokens or 16000,
)

print(f"{result.original_tokens} → {result.compressed_tokens} token"
      f"（保留 {result.compression_ratio:.0%}）")
print(result.compressed_text[:500])
```

看三件事：

1. **保留的内容对不对**——`result.blocks` 里的每条都带 `original_indices`，能追回原文。
2. **token 省了多少**——`compression_ratio` 越低压得越狠。
3. **它交代了多大的缺口**——`result.stats["dropped_messages"]` 和
   `stats["dropped_tokens"]` 是没进摘要的那部分历史。摘要末尾也会写一行把这件事
   告诉模型（默认开启，中文问题给中文提示）。

如果压出来比原文还长，说明这段对话本来就不该压，代码会直接原样返回并置
`stats["passthrough"] = True`。

## 第三步：接进 Agent 循环（完整可跑）

这是最终形态。注意它把判断、压缩、降级检查都串起来了：

```python
import asyncio
from shenlun_forge import ShenlunForge, plan_compression

forge = ShenlunForge.create(
    strategy="simple",                      # 先用免费的；需要更强再换
    # strategy="acs", model="deepseek-chat",
    # api_key="sk-...", base_url="https://api.deepseek.com/v1",
)

async def ask(messages: list, question: str, model_window: int = 128_000) -> str:
    # 1) 装得下就直接发
    plan = plan_compression(messages=messages, model_window=model_window)
    if plan.fits_in_window:
        return await call_model(messages, question)

    # 2) 压一次
    result = await forge.compress(
        messages=messages,
        current_query=question,
        max_tokens=plan.recommended_max_tokens or 16000,
    )

    # 3) 检查有没有降级（LLM 策略没配好、调用失败等），别让它在生产里悄悄发生
    if result.stats.get("degraded"):
        print(f"[warn] compression degraded: {result.stats['degraded_reason']}")

    # 4) 用压缩结果替换历史
    compressed_history = (
        [messages[0]] if messages and messages[0].get("role") == "system" else []
    ) + [{"role": "user", "content": result.compressed_text}]
    return await call_model(compressed_history, question)
```

要点：

- **`plan_compression` 放在最前面**，装得下就什么都不做。
- **降级要检查**。`simple` 永远不会 degraded；`acs`/`hcs` 没配好 key 或调用失败时会。
- **system 消息单独留**，别让它被压进正文里。

想换成 LLM 策略（历史很长、问题措辞和历史用词对不上时值得），只改构造那两行即可。

## 第四步：在你自己的数据上验证值不值

这是最容易被跳过、但最重要的一步。README 里的数字来自**别人的数据**，而实测结论是：

> 在普通工作对话上，压缩与"只保留最近的消息"打平；只有在答案远离历史末尾时，
> 语义检索才明显胜出（+25 个百分点）。**你的数据属于哪种，只能自己测。**

准备一批你自己的对话（每个文件是 `{"messages": [...]}`），然后：

```bash
shenlun-forge eval --conversations ./my-chats \
    --strategies truncate,simple,embedding \
    --budget 16000 --max-points 60
```

它会：压缩每段历史 → 让模型**只看压缩结果**回答 → 与当时真实回复的要点比对。

输出长这样：

```
Accuracy by strategy (budget 16000 tokens, n=60 points):
  embedding             44.1%
  simple                30.6%
  truncate              21.4%

Best: embedding. Treat differences under ~5 points as noise ...
```

读结果时记住两条：

- **小于 5 个百分点的差异是噪声**（这是该指标自身的重跑波动）。
- **独立的统计单位是「对话」，不是「问题」**——往已有对话里加更多轮次不会让结论更可靠，
  要加不同的对话。

注意：提要和判分都要调用模型，**这会把你的对话内容发给服务商**。

## 第五步：按结果选配置

| 评测结果 | 该怎么做 |
|---|---|
| `simple` 与 `truncate` 打平（最常见） | 用 `simple`。它免费、毫秒级，你已经拿到了能拿的全部。 |
| `embedding` 明显更好（差 10 个点以上） | 换成 `retriever_type="embedding"`。它只比 `simple` 慢约 60 毫秒。 |
| 三种都趴在地上（比如都低于 25%） | 问题多半不在压缩，在预算或任务本身。加大 `max_tokens`，或检查这些问题是否真的能从历史里答出来。 |
| `truncate` 最好 | 说明你的答案是"最新值"，压缩反而有害。直接截断，或者干脆别压。 |

想用 LLM 策略时再对比 `acs`；`hcs` 已判定不推荐（实测与免费方案打平却要多一次调用）。

## 常见坑

**1. 不检查 `degraded`。** 没配 key 时 `acs` 会退回规则模式并**照常返回结果**。
生产环境务必检查 `result.stats["degraded"]`。

**2. 什么都传同一个 `max_tokens`。** 预算越大质量越好，但成本也涨。用
`plan_compression()` 算出的可用窗口，而不是拍一个数字。

**3. 每轮都用同一个 query 压。** 压缩是**针对问题**的：同一个历史，对"数据库用什么"
和"接口怎么设计"应该压出不同结果。传入当前这一轮的真实问题。

**4. 把 `compressed_text` 当历史继续累积。** 它是给你这一轮用的字符串；下一轮请从
完整的 `messages` 重新压，否则会越压越薄。

**5. 用合成对话调参。** 合成数据的答案位置分布和真实对话完全不同，测出来的结论
未必成立——这一轮我已经在合成数据上栽过一次。

## 下一步

- 场景对照：[`scenarios.md`](scenarios.md)
- 集成细节（LangChain / LlamaIndex / 自定义组件）：[`integration.md`](integration.md)
- 常见疑问（降级、passthrough、中文支持）：[`faq.md`](faq.md)
- 这些数字怎么来的：[`../benchmarks/README.md`](../benchmarks/README.md)
