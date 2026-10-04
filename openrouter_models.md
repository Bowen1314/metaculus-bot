# 这把 Metaculus OpenRouter key 能调用的模型

来源：OpenRouter `/api/v1/models/user`（按这把 key 所属账户的限制过滤），2026-10-04 查询。共 160 个（OpenRouter 公开目录共 466 个）。价格是每百万 token 的美元价，输入 / 输出。

## openai（91 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `openai/gpt-3.5-turbo` | 16385 | 0.5 | 1.5 |
| `openai/gpt-3.5-turbo-16k` | 16385 | 3 | 4 |
| `openai/gpt-3.5-turbo-instruct` | 4095 | 1.5 | 2 |
| `openai/gpt-3.5-turbo:batch` | 16385 | 0.25 | 0.75 |
| `openai/gpt-4` | 8191 | 30 | 60 |
| `openai/gpt-4-turbo` | 128000 | 10 | 30 |
| `openai/gpt-4-turbo:batch` | 128000 | 5 | 15 |
| `openai/gpt-4.1` | 1047576 | 2 | 8 |
| `openai/gpt-4.1-mini` | 1047576 | 0.4 | 1.6 |
| `openai/gpt-4.1-mini:batch` | 1047576 | 0.2 | 0.8 |
| `openai/gpt-4.1-nano` | 1047576 | 0.1 | 0.4 |
| `openai/gpt-4.1-nano:batch` | 1047576 | 0.05 | 0.2 |
| `openai/gpt-4.1:batch` | 1047576 | 1 | 4 |
| `openai/gpt-4o` | 128000 | 2.5 | 10 |
| `openai/gpt-4o-2024-05-13` | 128000 | 5 | 15 |
| `openai/gpt-4o-2024-08-06` | 128000 | 2.5 | 10 |
| `openai/gpt-4o-2024-11-20` | 128000 | 2.5 | 10 |
| `openai/gpt-4o-mini` | 128000 | 0.15 | 0.6 |
| `openai/gpt-4o-mini-2024-07-18` | 128000 | 0.15 | 0.6 |
| `openai/gpt-4o-mini:batch` | 128000 | 0.075 | 0.3 |
| `openai/gpt-4o:batch` | 128000 | 1.25 | 5 |
| `openai/gpt-5` | 400000 | 1.25 | 10 |
| `openai/gpt-5-image` | 400000 | 10 | 10 |
| `openai/gpt-5-image-mini` | 400000 | 2.5 | 2 |
| `openai/gpt-5-mini` | 400000 | 0.25 | 2 |
| `openai/gpt-5-mini:batch` | 400000 | 0.125 | 1 |
| `openai/gpt-5-nano` | 400000 | 0.05 | 0.4 |
| `openai/gpt-5-nano:batch` | 400000 | 0.025 | 0.2 |
| `openai/gpt-5-pro` | 400000 | 15 | 120 |
| `openai/gpt-5-pro:batch` | 400000 | 7.5 | 60 |
| `openai/gpt-5.1` | 400000 | 1.25 | 10 |
| `openai/gpt-5.1:batch` | 400000 | 0.625 | 5 |
| `openai/gpt-5.2` | 400000 | 1.75 | 14 |
| `openai/gpt-5.2-pro` | 400000 | 21 | 168 |
| `openai/gpt-5.2-pro:batch` | 400000 | 10.5 | 84 |
| `openai/gpt-5.2:batch` | 400000 | 0.875 | 7 |
| `openai/gpt-5.3-codex` | 400000 | 1.75 | 14 |
| `openai/gpt-5.4` | 1050000 | 2.5 | 15 |
| `openai/gpt-5.4-image-2` | 272000 | 8 | 15 |
| `openai/gpt-5.4-mini` | 400000 | 0.75 | 4.5 |
| `openai/gpt-5.4-mini:batch` | 400000 | 0.375 | 2.25 |
| `openai/gpt-5.4-nano` | 400000 | 0.2 | 1.25 |
| `openai/gpt-5.4-nano:batch` | 400000 | 0.1 | 0.625 |
| `openai/gpt-5.4-pro` | 1050000 | 30 | 180 |
| `openai/gpt-5.4-pro:batch` | 1050000 | 15 | 90 |
| `openai/gpt-5.4:batch` | 1050000 | 1.25 | 7.5 |
| `openai/gpt-5.5` | 1050000 | 5 | 30 |
| `openai/gpt-5.5-pro` | 1050000 | 30 | 180 |
| `openai/gpt-5.5-pro:batch` | 1050000 | 15 | 90 |
| `openai/gpt-5.5:batch` | 1050000 | 2.5 | 15 |
| `openai/gpt-5.6-luna` | 1050000 | 0.2 | 1.2 |
| `openai/gpt-5.6-luna-pro` | 1050000 | 0.2 | 1.2 |
| `openai/gpt-5.6-luna-pro:batch` | 1050000 | 0.1 | 0.6 |
| `openai/gpt-5.6-luna:batch` | 1050000 | 0.1 | 0.6 |
| `openai/gpt-5.6-sol` | 1050000 | 2 | 10 |
| `openai/gpt-5.6-sol-pro` | 1050000 | 2 | 10 |
| `openai/gpt-5.6-sol-pro:batch` | 1050000 | 1 | 5 |
| `openai/gpt-5.6-sol:batch` | 1050000 | 1 | 5 |
| `openai/gpt-5.6-terra` | 1050000 | 2 | 12 |
| `openai/gpt-5.6-terra-pro` | 1050000 | 2 | 12 |
| `openai/gpt-5.6-terra-pro:batch` | 1050000 | 1 | 6 |
| `openai/gpt-5.6-terra:batch` | 1050000 | 1 | 6 |
| `openai/gpt-5:batch` | 400000 | 0.625 | 5 |
| `openai/gpt-6-astra` | 1050000 | 10 | 50 |
| `openai/gpt-6-astra-pro` | 1050000 | 10 | 50 |
| `openai/gpt-6-astra-pro:batch` | 1050000 | 5 | 25 |
| `openai/gpt-6-astra:batch` | 1050000 | 5 | 25 |
| `openai/gpt-6-luna` | 1050000 | 0.1 | 0.5 |
| `openai/gpt-6-luna-pro` | 1050000 | 0.1 | 0.5 |
| `openai/gpt-6-luna-pro:batch` | 1050000 | 0.05 | 0.25 |
| `openai/gpt-6-luna:batch` | 1050000 | 0.05 | 0.25 |
| `openai/gpt-6-sol` | 1050000 | 2 | 10 |
| `openai/gpt-6-sol-pro` | 1050000 | 2 | 10 |
| `openai/gpt-6-sol-pro:batch` | 1050000 | 1 | 5 |
| `openai/gpt-6-sol:batch` | 1050000 | 1 | 5 |
| `openai/gpt-6.1-sol` | 1050000 | 2 | 10 |
| `openai/gpt-6.1-sol-pro` | 1050000 | 2 | 10 |
| `openai/gpt-audio` | 128000 | 2.5 | 10 |
| `openai/gpt-audio-mini` | 128000 | 0.6 | 2.4 |
| `openai/gpt-chat-latest` | 400000 | 5 | 30 |
| `openai/o1` | 200000 | 15 | 60 |
| `openai/o1-pro` | 200000 | 150 | 600 |
| `openai/o3` | 200000 | 2 | 8 |
| `openai/o3-mini` | 200000 | 1.1 | 4.4 |
| `openai/o3-mini-high` | 200000 | 1.1 | 4.4 |
| `openai/o3-mini:batch` | 200000 | 0.55 | 2.2 |
| `openai/o3-pro` | 200000 | 20 | 80 |
| `openai/o3:batch` | 200000 | 1 | 4 |
| `openai/o4-mini` | 200000 | 1.1 | 4.4 |
| `openai/o4-mini-high` | 200000 | 1.1 | 4.4 |
| `openai/o4-mini:batch` | 200000 | 0.55 | 2.2 |

## anthropic（27 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `anthropic/claude-fable-5` | 1000000 | 10 | 50 |
| `anthropic/claude-fable-5.1` | 1000000 | 10 | 50 |
| `anthropic/claude-fable-5.1:batch` | 1000000 | 5 | 25 |
| `anthropic/claude-fable-5:batch` | 1000000 | 5 | 25 |
| `anthropic/claude-haiku-4.5` | 200000 | 1 | 5 |
| `anthropic/claude-haiku-4.5:batch` | 200000 | 0.5 | 2.5 |
| `anthropic/claude-opus-4.1:batch` | 200000 | 7.5 | 37.5 |
| `anthropic/claude-opus-4.5` | 200000 | 5 | 25 |
| `anthropic/claude-opus-4.5:batch` | 200000 | 2.5 | 12.5 |
| `anthropic/claude-opus-4.6` | 1000000 | 5 | 25 |
| `anthropic/claude-opus-4.6:batch` | 1000000 | 2.5 | 12.5 |
| `anthropic/claude-opus-4.7` | 1000000 | 5 | 25 |
| `anthropic/claude-opus-4.7:batch` | 1000000 | 2.5 | 12.5 |
| `anthropic/claude-opus-4.8` | 1000000 | 5 | 25 |
| `anthropic/claude-opus-4.8:batch` | 1000000 | 2.5 | 12.5 |
| `anthropic/claude-opus-5` | 1000000 | 5 | 25 |
| `anthropic/claude-opus-5.5` | 1000000 | 4 | 20 |
| `anthropic/claude-opus-5.5:batch` | 1000000 | 2 | 10 |
| `anthropic/claude-opus-5:batch` | 1000000 | 2.5 | 12.5 |
| `anthropic/claude-sonnet-4.5` | 1000000 | 3 | 15 |
| `anthropic/claude-sonnet-4.5:batch` | 1000000 | 1.5 | 7.5 |
| `anthropic/claude-sonnet-4.6` | 1000000 | 3 | 15 |
| `anthropic/claude-sonnet-4.6:batch` | 1000000 | 1.5 | 7.5 |
| `anthropic/claude-sonnet-5` | 1000000 | 2 | 10 |
| `anthropic/claude-sonnet-5.5` | 1000000 | 2 | 10 |
| `anthropic/claude-sonnet-5.5:batch` | 1000000 | 1 | 5 |
| `anthropic/claude-sonnet-5:batch` | 1000000 | 1 | 5 |

## google（23 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `google/gemini-2.5-flash` | 1048576 | 0.3 | 2.5 |
| `google/gemini-2.5-flash-lite` | 1048576 | 0.1 | 0.4 |
| `google/gemini-2.5-flash-lite:batch` | 1048576 | 0.05 | 0.2 |
| `google/gemini-2.5-pro` | 1048576 | 1.25 | 10 |
| `google/gemini-3-flash-preview` | 1048576 | 0.5 | 3 |
| `google/gemini-3-pro-image` | 131072 | 2 | 12 |
| `google/gemini-3-pro-image-preview` | 65536 | 2 | 12 |
| `google/gemini-3.1-flash-image` | 65536 | 0.5 | 3 |
| `google/gemini-3.1-flash-image-preview` | 65536 | 0.5 | 3 |
| `google/gemini-3.1-flash-lite` | 1048576 | 0.25 | 1.5 |
| `google/gemini-3.1-flash-lite-image` | 65536 | 0.25 | 1.5 |
| `google/gemini-3.1-flash-lite-preview` | 1048576 | 0.25 | 1.5 |
| `google/gemini-3.1-pro-preview` | 1048576 | 2 | 12 |
| `google/gemini-3.1-pro-preview-customtools` | 1048576 | 2 | 12 |
| `google/gemini-3.5-flash` | 1048576 | 1.5 | 9 |
| `google/gemini-3.5-flash-lite` | 1048576 | 0.3 | 2.5 |
| `google/gemini-3.6-flash` | 1048576 | 0.75 | 3.75 |
| `google/gemini-3.7-flash` | 1048576 | 0.75 | 3.75 |
| `google/gemini-3.8-flash` | 1048576 | 0.75 | 3.75 |
| `google/gemma-4-26b-a4b-it:free` | 262144 | 0 | 0 |
| `google/gemma-4-31b-it:free` | 262144 | 0 | 0 |
| `google/lyria-3-clip-preview` | 1048576 | 0 | 0 |
| `google/lyria-3-pro-preview` | 1048576 | 0 | 0 |

## openrouter（6 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `openrouter/auto` | 2000000 | -1e+06 | -1e+06 |
| `openrouter/auto-beta` | 2000000 | -1e+06 | -1e+06 |
| `openrouter/bodybuilder` | 128000 | -1e+06 | -1e+06 |
| `openrouter/free` | 200000 | 0 | 0 |
| `openrouter/fusion` | 1000000 | -1e+06 | -1e+06 |
| `openrouter/pareto-code` | 2000000 | -1e+06 | -1e+06 |

## ~openai（5 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `~openai/gpt-astra-latest` | 1050000 | 10 | 50 |
| `~openai/gpt-luna-latest` | 1050000 | 0.1 | 0.5 |
| `~openai/gpt-mini-latest` | 400000 | 0.75 | 4.5 |
| `~openai/gpt-sol-latest` | 1050000 | 2 | 10 |
| `~openai/gpt-terra-latest` | 1050000 | 2 | 12 |

## ~anthropic（4 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `~anthropic/claude-fable-latest` | 1000000 | 10 | 50 |
| `~anthropic/claude-haiku-latest` | 200000 | 1 | 5 |
| `~anthropic/claude-opus-latest` | 1000000 | 4 | 20 |
| `~anthropic/claude-sonnet-latest` | 1000000 | 2 | 10 |

## ~google（2 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `~google/gemini-flash-latest` | 1048576 | 0.75 | 3.75 |
| `~google/gemini-pro-latest` | 1048576 | 2 | 12 |

## typesafe（1 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `typesafe/jev-router` | 1000000 | -1e+06 | -1e+06 |

## nvidia（1 个）

| 模型 | 上下文 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| `nvidia/switchyard` | 1000000 | -1e+06 | -1e+06 |
