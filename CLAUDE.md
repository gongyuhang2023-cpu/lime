# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这个仓库是什么

`xushengfeng/lime` 的 fork（LLM 驱动的拼音输入法）。上游 remote 名为 `upstream`，**push 已被禁用**（URL 设成 `DISABLED_no_push_to_upstream`），同步用 `git fetch upstream && git merge upstream/master`。

上游的 `dev` 分支落后 master 68 个提交且无独有内容，**是废弃分支，不要用**。最新 tag `260205.0` 比 master 旧 5 个月，开发基于 `master`。

### 本 fork 的目标

上游的用户词机制只做到一半，这个 fork 要补的是：

1. **持久化** —— `loadUserData()`（main.ts:735）函数体是 `// todo`，只恢复词表、不恢复上下文；没有任何自动落盘。重启即失忆。
2. **用户词的真实模型打分** —— 见下方「已知设计缺口」。这是让候选排序真正由语境驱动的关键，也是这个 fork 存在的主要理由。

## 命令

```bash
# 首次准备
deno install
deno approve-scripts node-llama-cpp   # 必须！否则原生 llama.cpp 二进制不存在

# 模型放在项目的【同级目录】，不是项目内
git clone https://www.modelscope.cn/unsloth/Qwen3-0.6B-GGUF.git

# 起服务器
deno serve -A --port 5000 server.ts
deno run -A key.ts                    # 生成密钥，只需一次，写进 rime/lua/llm_pinyin.lua 的 key 变量

# 测试
deno test -A test/test_user_word.ts
deno test -A test/test_user_word.ts --filter "组词"   # 单个测试
deno test -A test/                                    # 全部

# 打字速度/排序质量基准（长句灌拼音，统计按键数、候选查找耗时）
deno run -A test/test_text.ts

# 批量导入 RIME 词库为用户词（按词频 >5000 过滤，写入 userword/preload_word.txt）
deno run -A userword/preload_word.ts rime <rime词库.dict.yaml路径>

# Web 前端（不影响输入法本身，只影响 demo/统计页）
deno run install_interface && deno run build_interface
```

## 架构

### 核心思路：模型出概率，拼音做过滤

传统输入法是「词典给候选 → n-gram 排序」。lime 反过来：

- `LIME` 持有一个**长期存活的** `LlamaContextSequence`。你已经上屏的文字就是它的上下文，`commit(text)` 把文字追加进去 —— 这就是「记忆」。
- `last_result` 保存模型**完整的 next-token 概率分布**（`topK: Infinity`，见 `init_ctx`）。
- 构造函数启动时遍历**模型全词表**，把每个 token 反解成文字再转拼音，建立 `token_pinyin_map`（token → 拼音）和 `first_pinyin_token`（拼音 → token 集合）两张索引。这是每次启动的主要开销。
- 敲拼音时 `single_ci()` 用拼音**过滤** `last_result`，按模型概率排序。

所以候选完全来自模型词表，没有传统意义上的词典解码。理解任何排序问题都要从 `last_result` 和 `filterByPinyin` 入手。

### 用户词 = 虚拟 token（ExToken）

`tokenIndex` 从「模型最大真实 token id + 1」开始。`addUserWord(w)` 给每个用户词分配一个虚拟 id，映射到它展开后的真实 token 序列（`userTokens`），并把它注册进拼音索引 —— 于是用户词能作为一等候选出现。`exTokens()` / `detoken()` 负责展开。

`checkAddUserWord()` 会拒绝两类词：tokenize 后只有 1 个 token 的（模型自己就能预测），以及含无拼音映射 token 的。

### ⚠️ 已知设计缺口：用户词的模型概率恒为 0

三处硬编码置零：

- `main.ts:333` —— `addUserWord` 新增时置 0
- `main.ts:320` —— `commit` 后置 0
- `main.ts:594` —— `addToken()` 每次重算分布后置 0

**后果**：用户词是「白名单」不是「高分词」。它进得了候选池，但排名靠「长词优先」启发式和 `config.ts` 里 `afterReSort` 的 `resortFeq`（一张 top2500 常用字频率表）决定，**与上下文无关**。

正确做法是给虚拟 token 算展开后的真实联合概率 `P(t1)·P(t2|t1)·…`。`addToken()`（main.ts:578）已经在做 ExToken 展开求值，只是算完把结果丢了填了 0。改动要配剪枝（只给 top-N 候选算），否则每个候选一次前向扛不住。

### 持久化现状

| 环节 | 状态 |
|---|---|
| 导出 | `getUserData()` (main.ts:726) + `GET /api/userdata`，返回 `{words, context}` |
| 导入 | `loadUserData()` (main.ts:735) 是 `// todo`，只恢复 `words`，**`context` 完全没恢复** |
| 启动加载 | `server.ts:64` 读 `config.userWordsPath` 纯文本词表，逐行 `addUserWord()` |
| 自动落盘 | **不存在** |

注意「记忆」是两层，持久化难度差一个量级：**用户词**（`userTokens`，纯数据，重放 `addUserWord` 即可）vs **上下文**（`sequence.contextTokens` + llama.cpp 的 KV cache，是模型中间状态，只能存文本重新 prefill，且必须配滚动窗口）。

### 上下文窗口管理

`pre_context`（默认 `"下面的内容主题多样"`）作为种子。`max_count = contextSize - 64`，超了就 `tryOmitContext()` 裁掉旧 token；`omitContext` 是个 10 秒空闲触发的 deBounce，趁空闲做大裁剪。`reset_context()` 会连用户词一起清空。

### 配置覆盖

`config.ts` 是默认配置，**复制成 `user_config.ts`**（已 gitignore）来改。注意 `config.ts` 在模块加载时就调用 `initLIME()` —— **import 它就会加载模型**，server.ts 和 preload_word.ts 都吃这个成本。

### 部署形态

Deno + Hono HTTP 服务器，RIME 前端（`rime/lua/llm_pinyin.lua`）通过 shell 调 `curl` 请求 `/candidates` 和 `/commit`；开了 HiAE 加密时还会 shell 调 `deno run hiae_payload.ts`。**所有按键都流经 localhost HTTP**，别把端口暴露出去。

RIME schema 叫 `llm`，是独立方案，**不能和其他 RIME 方案组合使用**。

## 环境约束

- `loadModel()` 里 `getLlama({ gpu: false })` —— **CPU 推理是写死的**（main.ts:64），想上 GPU 要改这里。
- 默认模型路径 `../Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf`，**在项目同级目录**，不在项目内。
- 默认 `contextSize` 4096。
- gitignore 的本地状态：`key.txt`、`user_config.ts`、`userword/preload_word.txt`、`node_modules`。
