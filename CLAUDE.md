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

# 打字速度/排序质量基准
LIME_GPU=1 LIME_BENCH_MODEL=<gguf> LIME_BENCH_TEXT=<txt> deno run -A test/test_text.ts
deno run -A test/test_text.ts cal 400        # 把上一轮结果换算成理论 cpm

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

### 用户词打分（本 fork 已修）

上游把用户词的模型概率三处硬编码为 0（注释都写着「临时」），导致按概率排序的逻辑对用户词全部失效，上游自带的 `组词` 测试在 master 上是红的。

现在：`applyUserTokenProbs()` 取首个真实 token 的概率作为估计，填进 `last_result` 后与其他候选一起过归一化。排序上，`Candidate.userWord` 标记让同等长度的用户词优先于模型即兴拼出的同音串 —— 这是用户词典本来的语义（实测冰灯概率 0.0028 远低于并等 0.466，是这一层让它赢的，光靠概率赢不了）。

`addUserWord` 按 token 序列去重；`registerUserToken` 是 `addUserWord` / `loadUserData` 共用的注册入口，**只往 `userTokens` 塞映射而不建拼音索引的话，这个词永远不会进候选集**。

### 持久化

| 环节 | 状态 |
|---|---|
| 导出 | `getUserData()` + `GET /api/userdata`，返回 `{words, context}` |
| 导入 | `loadUserData()` 本 fork 已实现（含索引重建、`tokenIndex` 推进、上下文重放），有往返测试 |
| 启动加载 | `server.ts:64` 读 `config.userWordsPath` 纯文本词表，逐行 `addUserWord()` |
| 自动落盘 | **仍不存在** —— 这是剩下的活 |

「记忆」是两层，难度差一个量级：**用户词**（纯数据，重建索引即可）vs **上下文**（`sequence.contextTokens` + llama.cpp 的 KV cache，是模型中间状态，只能把 token 序列重新喂一遍，长度受 `contextSize` 限制）。

### 候选排序与长句

最终排序在 `single_ci` 末尾：按拼音长度降序，用户词同长度优先，再过 `afterReSort`（`config.ts` 里挂了 `resortFeq`，一张 top2500 常用字频率表）。

**上游只按长度排序、完全忽略 `score`。** 敲一整句拼音时，候选表头部会被一串等长的错误整句猜测占满，正确的短词被挤到很后面（实测最惨一例：正确单字排第 54 位，`page_size` 是 5，等于翻 11 页）。

`LIME_LONG_HEAD` 限制头部允许几个长候选，之后让位给短候选。

**已否定的方案**：给长候选设「每音节平均置信度」闸门无效。扫过 0.35 / 0.50 / 0.65 三档，首选命中率只从 47.1% 升到 50.7%，但选择次数从 333 涨到 353，总成本（偏移加权）持平，长尾（54/22/13/12）**四档完全不变**。原因是那些垃圾长句的置信度高于 0.65 —— 模型不是不自信，是自信地错。别再往这个方向试。

### 上下文窗口管理

`pre_context`（默认 `"下面的内容主题多样"`）作为种子。`max_count = contextSize - 64`，超了就 `tryOmitContext()` 裁掉旧 token；`omitContext` 是个 10 秒空闲触发的 deBounce，趁空闲做大裁剪。`reset_context()` 会连用户词一起清空。

### 配置覆盖

`config.ts` 是默认配置，**复制成 `user_config.ts`**（已 gitignore）来改。注意 `config.ts` 在模块加载时就调用 `initLIME()` —— **import 它就会加载模型**，server.ts 和 preload_word.ts 都吃这个成本。

### 部署形态

Deno + Hono HTTP 服务器，RIME 前端（`rime/lua/llm_pinyin.lua`）通过 shell 调 `curl` 请求 `/candidates` 和 `/commit`；开了 HiAE 加密时还会 shell 调 `deno run hiae_payload.ts`。**所有按键都流经 localhost HTTP**，别把端口暴露出去。

RIME schema 叫 `llm`，是独立方案，**不能和其他 RIME 方案组合使用**。

### 读基准数字前必须知道的两件事

**它测的是最难的输入方式。** `test_text.ts` 用 `Intl.Segmenter` 切词后，会把连续的词块**累积到标点为止**才作为一个输入单元 —— 也就是「敲完一整个小句的拼音再选字」，不是逐词上屏。所以偏移数字比日常逐词输入要难看。

**核心指标是 `偏移加权`，不是 `非0偏移占比`。** 后者只说首选中没中，前者才是总选择成本（Σ 偏移×次数）。两者会背离：置信度闸门那一轮，首选命中从 47.1% 升到 50.7%，但选择次数从 333 涨到 353，总成本没变 —— 只看命中率会误判成有效。

## 环境约束

- `loadModel()` 里 `getLlama({ gpu: false })` —— **CPU 推理是写死的**（main.ts:64），想上 GPU 要改这里。
- 默认模型路径 `../Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf`，**在项目同级目录**，不在项目内。
- 默认 `contextSize` 4096。
- gitignore 的本地状态：`key.txt`、`user_config.ts`、`userword/preload_word.txt`、`node_modules`。
