# Mini-SGLang 带读笔记（docs/minisglang-notes.md）

> 陪读版：源码在服务器 `/storage/xuan/vlabench/code/mini-sglang`（sgl-project/mini-sglang，约 9000 行）。每个模块按「读什么 → 关键结构 → 和 HF generate 的差别」组织。文件引用都是真实的，可直接跳转对着读。读完这份，你就具备写 README 概念章节所需的全部素材。

## 0. 这份代码是什么

一个教学用 LLM 推理引擎，只保留真实 SGLang/vLLM 的骨架：**prefill/decode 两阶段调度、分页 KV cache、radix 前缀缓存、chunked prefill、overlap 调度（CPU/GPU 双流）、decode 的 CUDA graph、张量并行**。没有投机解码、没有多 LoRA、没有分布式 KV——每删掉一样东西都是为了让你能在一下午读完核心。

一条请求的生命周期（先建立全局图景再进模块）：

```
UserMsg(token ids) ──► PrefillManager.pending_list
                          │  PrefillAdder：token 预算内凑批 + prefix 匹配(cached_len)
                          ▼
                    Batch(phase="prefill")  ──engine.forward_batch──► 首个 token
                          │  非 ChunkedReq 的请求经 filter_reqs 加入 DecodeManager.running_reqs
                          │  同时 cache_req 把 prompt 的 KV 插进 radix 树
                          ▼
                    Batch(phase="decode")   ──每轮带上全部 runnable 请求──► 逐 token
                          │  EOS 或 max_tokens → remove_req + 释放页 + 尾部 KV 插树
                          ▼
                    DetokenizeMsg ──► 返回
```

## 1. `scheduler/scheduler.py` — 主循环（267 行，先读这个）

- `run_forever` → `overlap_loop`（`scheduler.py:97`）：每轮做四件事——收消息 → `_schedule_next_batch()` → 在**引擎流**上 forward → `_process_last_data(上一轮的输出)`。注意它处理的是**上一轮**的 ForwardData：CPU 处理与 GPU 计算靠两条 CUDA stream 错开一拍（`scheduler.py:66-69`），这就是 overlap scheduling，专治"GPU 等 Python"。
- `_schedule_next_batch`（`scheduler.py:248`）：`prefill_manager.schedule_next_batch(budget) or decode_manager.schedule_next_batch()`——**prefill 优先**，代码里留着 `TODO: support other policies: e.g. DECODE first`（这是真实引擎里 prefill/decode 优先级之争的最小版本）。
- `_prepare_batch`：pad_batch（CUDA graph 档位对齐）→ `cache_manager.allocate_paged` → 拼 Positions/input_mapping/write_mapping 三个 pin-memory 张量异步上卡。**调度器的输出就是这几个索引张量**，模型侧完全按索引取数——这是 paged attention 的软件面。
- `_process_last_data`（`scheduler.py:158`）：append token → 判断结束（EOS/max）→ 结束则 `cache_req(finished=True)` 释放资源，未结束且是 prefill 则把 prompt 前缀 `cache_req(finished=False)` 插入 radix 树。

## 2. `scheduler/prefill.py` — prefill 怎么凑批（162 行）

- `PrefillAdder`（`prefill.py:27`）两个预算：`token_budget = max_extend_tokens`（**一次 prefill 前向最多算多少 token**，即 chunked prefill 的档位）和 `reserved_size`（初始化为 `decode_manager.inflight_tokens`——**在飞 decode 的 KV 需求要预留**，prefill 不能把 decode 的页吃光）。
- `_try_allocate_one`：先 `cache_manager.match_req` 查前缀缓存命中 `cached_len`，只需要算 `extend_len = input_len - cached_len`。**radix cache 的收益直接体现在 prefill 的工作量上。**
- `ChunkedReq`（`prefill.py:22`）：prompt 超过预算就被切块，`can_decode=False`、`append_host` 直接 raise——**被切块的 prefill 不采样、不进 decode**，下轮继续吃预算。这是"长 prompt 不许饿短请求"的机制核心。
- 值得注意的细节：`match_req` 匹配的是 `input_ids[:input_len-1]`（`scheduler/cache.py:33`）——去掉最后一个 token，因为它的 KV 还没算。

## 3. `scheduler/decode.py` — continuous batching 的真身（39 行！）

```python
def schedule_next_batch(self):
    return Batch(reqs=sorted(self.running_reqs, ...), phase="decode")
```

连续批的实现就这一句话：**decode 批 = 当前全部 runnable 请求**。没有"等凑齐一批"、没有"同进同出"——每轮迭代自然地把上一轮 prefill 完的请求吸收进来（`filter_reqs`），把结束的踢出去（`remove_req`）。新请求的加入粒度是**一次迭代**，而不是一个 batch 的生命周期。这就是 continuous batching 的全部。

`inflight_tokens`（`decode.py:33`）给每个在飞请求多预留一页——decode 每步都可能涨一个 token，页分配不能贴着当前长度算。

## 4. `kvcache/radix_cache.py` + `scheduler/cache.py` — KV 的账本（237+146 行）

两层数据结构：

1. **页池**（`cache.py:CacheManager`）：`free_slots` 一个张量就是全部空闲页；`allocate_paged` 按请求的 `cached_len→device_len` 补页并写 page_table。`lazy_free_region` 把一轮批内的释放攒到批末统一 cat——避免分配器抖动。
2. **radix 树**（`radix_cache.py`）：key = 一页的 token ids（`_get_key_fn`），节点存 token 序列对应的页索引；`match_prefix` 树走 + `split_at` 分裂；`ref_count` 区分 protected（在用）/evictable（可驱逐）；`evict` 用**叶子节点的最小堆按时间戳弹出**（叶优先 + LRU——只能从叶子驱逐，因为前缀可能被别人引用）。

`cache.py:cache_req` 里那段区间注释（`[0, cached_len)`、`[cached_len, new_handle.cached_len)`…）是全文件最值得抄的注释：一个请求的 KV 生命被切成五段，各自归属不同主体（树/请求/尾巴）。读懂它 = 读懂生产级 KV 管理的语义。

## 5. `engine/engine.py` + `engine/graph.py` — 执行层

- `page_table` 布局：`(max_running_req+1) × max_seq_len` 的**扁平 int32 索引表**（`engine.py:69`），第 i 行 = 第 i 个在飞请求的每 token KV 页位置；dummy 请求槽填 `num_tokens`（越界读落到一个安全页——CUDA graph padding 的配合设施）。
- CUDA graph **只给 decode**（`engine.py:194` `can_use_cuda_graph`）：档位 `[1,2,4] + range(8, 256+1, 8)`（`graph.py:67`）。原因在概念上正对应我们的测量：decode 每步形状固定（batch×1 token），可以把整个前向+采样录进 graph 重放，消掉 Python/launch 开销；prefill 形状随 prompt 变，没法录。
- `forward_batch`：attn_backend.prepare_metadata 由调度器喂的索引张量构建 paged attention 元数据。

## 6. 与 HF `generate()` 的差别清单（我们基准的对照组）

| | HF `generate()`（本仓库的测量路径） | Mini-SGLang |
|---|---|---|
| 批 | 静态：一次 generate 一批，同进同出 | 连续：每轮迭代重组成批 |
| KV | 连续分配、请求结束即释放 | 分页 + radix 前缀复用 + LRU 驱逐 |
| prefill/decode | 同一个循环里顺其自然 | 显式两阶段、独立预算、可切块（chunked prefill） |
| CPU 开销 | Python 循环逐步驱动 | overlap 双流 + decode CUDA graph |
| 调度 | 无（谁来都是串行） | prefill 优先、预算制准入 |

## 7. 用到 VLA 章节：每个概念接我们的一根数据线

1. **prefill/decode 两阶段 ↔ 两种 VLA decode 形态**：OpenVLA 是教科书映射（prefill 31 ms / decode 127 ms 占 65%，逐 token 带宽墙 → INT4 decode 127→101 ms 的加速直接可用引擎侧结论）；SmolVLA 的 decode 是 10 步并行去噪、共享同一条 prefix KV（源码 `fill_kv_cache`），"chunk 几乎免费"（chunk 1→50 只 +8 ms）的根源就是没有逐 token 墙。
2. **radix 前缀缓存 ↔ KV 能否跨控制步复用**：Mini-SGLang 的 key 是 token 前缀；VLA 每步图像 token 都变 → 图像部分永远 miss，可复用的只有冻结指令前缀。我们测得 prefill 中 vision 仅 9–14 ms、语言主干占大头 → 值得做的是"指令前缀常驻"，不是图像缓存。SmolVLA 已经在步内复用 KV（10 个去噪步共享），这是引擎思想在模型内部的预演。
3. **chunked prefill + 预算 ↔ 保护在飞请求**：`reserved_size = inflight_tokens` 的设计直接翻译成多机器人场景——新机器人的 prefill 不许吃掉正在解码的机器人的 KV 预算。
4. **continuous batching ↔ 多机器人**：decode.py 的 39 行证明连续批的准入粒度是"一次迭代"。我们测得 SmolVLA batch 8 吞吐 ×6.6 且 decode 几乎不随 batch 变慢（带宽受限）——连续批在 VLA 上有真实的物理红利；且 chunk 模型的 decode 长度固定（10 步）、比变长 LLM decode 更好调度。OpenVLA 上游不支持批量生成，适合"每机器人一个流"。
5. **overlap + CUDA graph ↔ 我们的 "other" 相位**：SmolVLA 每步 ~56 ms（约 1/3 e2e）花在 Python 粘合上——这正是 overlap 调度和 decode CUDA graph 在生产引擎里消掉的那类开销。引擎技术栈对 VLA 的第一步收益不在模型，在粘合层。
