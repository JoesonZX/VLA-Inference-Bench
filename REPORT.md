# REPORT — VLA-Inference-Bench

> 顺畅故事线与结果。所有数字出自 `results/*.json`（正式协议 v2：30 次采样（个别 50 次）、warmup 5、固定同一张物理 H100、run 前 GPU 空闲检查、输入帧按 run 序号轮换）；汇总见 `results/summary.md`，图由 `plots/make_plots.py` 从 JSON 生成。英文对外版见 README。

## TL;DR（三句话结论）

1. **OpenVLA-7B 在单张 H100 上一条动作 ≈195 ms（BF16、14.1 GB 权重）；NF4 INT4 把权重压到 4.1 GB（−71%）而端到端延迟几乎不变（+3%）**——decode（占 65%、带宽受限）因 4-bit 权重流量减半而加速（127→101 ms），抵消了 prefill（算力受限）的解量化开销（31→53 ms）。量化让推理变快还是变慢，取决于负载落在带宽侧还是算力侧。
2. **对输出 action chunk 的 SmolVLA，chunk 长度几乎免费**：chunk 1→50，一步延迟只 +8 ms（168→176 ms），因为 decode 是并行的 flow-matching 去噪而非自回归；摊销后每动作成本从 168 ms 降到 3.5 ms（**47×**）。chunk 化比量化干净得多，代价在控制语义（开环执行）而非延迟。
3. **bitsandbytes INT8 在单步小 batch 的 VLA 负载上是坏交易**：默认 outlier 分解配置慢 17–34 倍（SmolVLA 3067 ms、OpenVLA 6664 ms），修正 threshold=0 后仍比基线慢 52–58%。**bitsandbytes 路线下选 NF4 而非 INT8——但 INT4 的输出偏差不可忽视且随 chunk 放大（见 4.2），部署前必须配合任务级评估。**

## 1. 问题：部署一个 VLA，"一步"要花多少？

VLA 以观测→动作闭环运行：机器人控制通常要求 10–50 Hz 的动作输出，而 7B 级 VLM 主干的每步推理动辄数百毫秒。三个系统问题被推到前台：

- **延迟**：单步端到端延迟决定可达控制频率；
- **显存**：权重+激活峰值决定一张卡装得下几个模型/服务几个机器人；
- **保真**：任何压缩都会改变输出动作，改变多少需要测量而不是假设。

两个杠杆——训练后量化（INT8/INT4）与动作分块（chunk）——分别作用在显存和吞吐上。本仓库在同一张 H100、同一组固定输入上，对两种架构的 VLA 系统测量这两个杠杆的收益与代价，并把延迟分解到相位，说明**为什么**。

## 2. 两个模型，两种"一步"

| | OpenVLA-7B | SmolVLA-450M |
|---|---|---|
| 主干 | Prismatic VLM（Llama-2-7B + 双 SigLIP） | SmolVLM2-500M（16 层）+ 动作专家 |
| 动作 | 每步 **1 个动作** = 7 个离散 token | 每步一个 **chunk（默认 50 动作）**，连续值 |
| decode | **自回归**：prefill 后 6 次单 token 前向 | **flow-matching**：10 次去噪，每次并行出整 chunk |
| KV cache | prompt（256 图像 token + 指令） | prefix（图像+语言+状态）算一次，**10 个去噪步复用** |
| 出厂精度 | BF16 | 混合（塔 BF16 + 动作头 FP32），lerobot 官方按此跑 |

相位埋点（CUDA event + 实例级 forward 补丁，详见 `models/*.py` docstring 与 LOG.md）：`vision` / `prefill_lm` / `decode`（OpenVLA=6 次前向之和；SmolVLA=10 次去噪之和）。

## 3. 方法与协议

- **硬件**：同一张物理 H100 80GB（driver 595.58.03）；每次 run 前 `nvidia-smi` 空闲检查写入 JSON meta（31 个正式 run 全部 idle=True）。共享集群无法锁频，为已知限制。
- **输入**：`lerobot/pusht` episode 0 的 8 个固定帧（96×96）+ 固定指令；fixture 进 git。SmolVLA 走 base checkpoint 的缺相机零填充路径；OpenVLA 用 `bridge_orig` 统计反归一化（数值口径一致即可，不追求语义——本仓库测推理行为，不测任务成功率）。
- **帧轮换**（协议 v2）：输入帧按 `run_idx % 8` 轮换，偏差指标覆盖全部 8 帧；v1 每步固定第 0 帧，OpenVLA 贪心解码下 30 次采样实为同一输入（v1 数据 std≈0 暴露了问题），已废弃重测。
- **采样**：warmup 5 + 30 次计时（chunk5 的 bf16 配置因瞬态干扰以 50 次复测）；报 mean±std、p50/p99；CUDA event 计相位、`torch.cuda.synchronize()` 收边。
- **可复现性**：SmolVLA 的 flow-matching 噪声由 (seed, run_idx) 派生的 CPU 生成器固定，量化 run 与 FP32 参考逐比特同噪声；OpenVLA 贪心解码确定。
- **偏差**：OpenVLA 对 BF16 参考（7 token top-1 不匹配率 + 反归一化动作 L2）；SmolVLA 对 FP32（as-shipped）参考（chunk MSE + 每动作 L2）。
- **版本**：torch 2.14.0+cu130；transformers 4.46.3（OpenVLA env）/ 4.57.1（SmolVLA env）；lerobot 0.4.4；bitsandbytes 0.50.2；timm 0.9.16（OpenVLA 远程代码硬性要求 0.9.10–0.9.16）。

## 4. 结果

### 4.1 相位分解：一步的时间花在哪

（图：`fig_openvla_precision.png` / `fig_smolvla_precision.png` / `fig_smolvla_chunk.png`；全表：`results/summary.md`）

- **OpenVLA BF16**：195±2 ms = decode 126.6（**65%**）+ prefill 30.7 + vision 9.2 + CPU 预处理 ~13.6 + 采样粘合 ~15。自回归 decode 主导，与 LLM serving 经验一致。
- **SmolVLA FP32@chunk50**：176±2 ms = decode 87.6（**50%**，10 次去硝 ≈8.8 ms/步）+ prefill 9.2 + vision 14.4 + 前后处理 ~9 + 粘合 ~56（lerobot 逐层 Python 循环，工程优化空间而非模型本质成本）。
- 两个模型 vision 塔都只占 5–8%：**视觉编码不是瓶颈，语言主干的 decode 才是**。

### 4.2 量化：买到什么，付出什么

（图：`fig_quant_tradeoff.png` / `fig_deviation.png`）

| 模型 | 精度 | e2e (ms, mean±std) | 权重 (GB) | vs 基线 | 输出偏差 |
|---|---|---|---|---|---|
| OpenVLA | BF16 | 195.0±2.0 | 14.09 | 基线 | — |
| OpenVLA | INT8 | 308.2±4.8 | 7.43 | **+58% / −47%** | token 不匹配 61%，动作 L2 0.147±0.28（相对 55%） |
| OpenVLA | INT4 | 201.3±2.0 | 4.08 | **+3% / −71%** | token 不匹配 64%，动作 L2 0.523±0.47（相对 1170%） |
| SmolVLA | FP32(出厂) | 176.2±1.9 | 0.90 | 基线 | — |
| SmolVLA | INT8 | 267.1±2.1 | 0.65 | +52% / −28% | chunk MSE 0.035（chunk50） |
| SmolVLA | INT4 | 213.9±5.7 | 0.46 | +21% / −49% | chunk MSE 0.244（chunk50） |

- **INT4 延迟"免费"的结构性原因**：decode 带宽受限（4-bit 权重流量减半 → OpenVLA decode 127→101 ms），prefill 算力受限（解量化是额外计算 → 31→53 ms），两端相抵。"量化快不快"取决于负载在带宽侧还是算力侧——VLA 单步小 batch 恰好两端都有。
- **INT8 是坏交易**：默认 threshold=6.0 的 outlier 分解让单步慢 17–34 倍；threshold=0 后仍全面慢于 INT4，且 bnb 的 int8 kernel 会把 bf16 输入 cast 到 fp16（"int8"名不副实）。
- **偏差没有免费午餐**：OpenVLA 的 token 不匹配率两档都 ~60%+（256-bin 下相邻 token 数值接近，但反归一化后 L2 不可忽略）；INT4 的动作 L2（0.52±0.47）**大于** INT8（0.15±0.28）且方差大——某些帧的动作偏离显著。SmolVLA 侧 INT4 的 chunk MSE（0.24）比 INT8（0.035）高一个数量级，且随 chunk 从 1→50 放大 ~7 倍（0.035→0.244），长 chunk 上误差累积。**结论：INT4 换显存是划算的，但输出偏差是实打实的，部署前必须配合任务级评估；INT8 在本设置下两头不讨好。**
- 注：OpenVLA INT4 相对 L2 的中位数远小于均值（部分参考动作接近零范数放大了相对值），绝对 L2 0.52 才是主读数。

### 4.3 chunk：被低估的免费午餐

（图：`fig_chunk_curve.png` / `fig_smolvla_chunk.png`）

- chunk 1→50（FP32）：e2e 168.4→176.2 ms（**+4.6%**），产出动作 ×50 → 摊销每动作 168.4→3.52 ms（**47.8×**）。INT4 下同样平缓（204→214 ms）。
- 原因：SmolVLA 的 decode 是并行 flow-matching（chunk 只是张量加宽），prefill/vision 完全不变。对比 OpenVLA：每步 1 个动作，每步全价重付 prefill+decode——这正是两种 VLA 架构在 serving 语义上的根本差异。
- 代价不在延迟在控制：chunk 期间动作开环执行（反应性下降）。延迟测量看不到这一维度，列为 limitation。

### 4.4 batch：多机器人共享一张卡

（图：`fig_batch.png`；OpenVLA 上游生成代码不支持 batch>1，见 LOG.md）

- SmolVLA BF16@chunk50：batch 1/2/4/8 的 e2e = 179/214/231/270 ms → **聚合吞吐 ×6.63**（近线性）。
- 相位解释：decode 带宽受限 → batch 增大 decode 几乎不动（86→109 ms）；vision 算力受限 → 随 batch 线性增长（14→77 ms）。**瓶颈随 batch 从 decode 侧转移到 vision 侧**，继续扩 batch 的收益递减点可直接从相位数据读出（batch≈8 时 vision 已占 29%）。

### 4.5 优化粘合层：M4 实验（profile → hoist 重排 → A/B）

> 背景：4.1 里 SmolVLA 的 "other"（粘合层）相位每步 ~56ms（约 ⅓ e2e）。M4 用"先剖析再动手"的流程验证概念章节的论点——serving 技术栈的第一笔收益在粘合层。

**剖析（T1，`benchmarks/profile_glue.py`）**：单步 wall ~211ms 中 **GPU 只忙 64.9ms、launch gap ≈146ms**；单步 **11253 次 kernel launch**、**32 次流同步（CPU 侧 52ms）**、`pow`×927。三个真靶子：flow-matching 循环的 `while tensor>=tensor` 条件（每轮强制 GPU→CPU 同步）、`embed_suffix` 每步重算时间步正弦嵌入（输入只有 10 个固定值）、`torch.tensor(列表, device=cuda)` 的同步 H2D。原设想的"缓存 attention mask"只值 0.2ms——**剖析避免了我们优化伪靶子**。

**干预（T2，`--variant hoist`）**：时间嵌入/掩码/position ids 按 (num_steps, chunk, batch) 预计算缓存、浮点循环控制、常量掩码移出去噪循环。**输出与上游逐比特一致（30 次配对 run 偏差全部为 0）**——纯粹的执行重排，零数值代价。

**A/B 结果（T3，同会话背靠背、warmup 20、30 次采样）**：

| 配置 | baseline e2e | hoist e2e | p50 对比 | 偏差 |
|---|---|---|---|---|
| fp32 chunk=1 | 181.7±18.9 | **129.7±1.9** | **171.3 → 129.7（−24%）** | 0 |
| fp32 chunk=10 | 172.5±2.1 | 162.9±14.3 | 172.4 → 170.4 | 0 |
| fp32 chunk=50 | 179.8±10.6 | 165.8±19.3 | 177.8 → 177.6 | 0 |
| bf16 chunk=50 | 177.8±2.7 | 162.1±19.6 | 178.7 → 174.4 | 与 bf16 基线同级 |

结构化解读（这是本实验最有价值的部分）：

- **粘合层优化的收益出现在"启动受限"的配置上**：chunk=1 时去噪张量极小、GPU 每个前向秒完工，CPU 侧的同步与启动间隙直接暴露为墙钟时间——删掉它们，p50 稳定 −24%（且分布从 ±19 收紧到 ±2，说明基线的抖动本身就是这些同步点）。
- **chunk=50 时收益被掩盖**：大 chunk 的 decode 内核足够大，被删掉的 CPU 工作本来就在 GPU 执行期间被异步吸收（p50 持平；hoist 的低均值来自偶发的 ~130ms 快态——即若残余启动开销也被消除的潜在水平，那是 CUDA graph（V2）的领地）。
- 测量方法论教训（详见 LOG Session 2）：H100 无 root 锁不了频率，短 run 会踩时钟斜坡；跨变体对比必须同会话背靠背 + 足够 warmup + 看中位数。
- V1（torch.compile）按剖析数据跳过：11253 次 launch 来自 lerobot 手写的 16 层 Python 循环，compile 必然大量图断裂。

**V2：CUDA graph 把去噪循环整体录进图（M4 stretch，本轮完成）**

设计依据（读 `smolvlm_with_expert.py` + M4 剖析确立的三个前提）：denoise 期间 KV 字典**只读**（纯输入）；prefix 长度恒定（固定 prompt + 固定图像尺寸 → 掩码/位置恒定）；hoist 循环已无 host 同步（可捕获）。实现：每控制步 eager 跑 embed_prefix + prefill，把新 KV **拷贝进预分配静态缓冲**（地址跨步稳定 → 图可复用），把 10 步去噪循环捕获一次，之后每步只拷贝噪声进静态输入缓冲并 `graph.replay()`。相位钩子在 replay 下看不到 decode（回放的是 kernel 不是 Python）——跨变体只比墙钟。

**三臂正式结果（同会话背靠背、warmup 20、30 次采样、安静窗口）**：

| fp32 chunk=50 | e2e (mean±std) | p50 | vs baseline | 偏差 vs fp32 参考 |
|---|---|---|---|---|
| baseline（上游） | 175.7±1.4 | 176.4 | — | — |
| hoist（粘合层重排） | 132.9±2.7 | 133.1 | −24% | **0（逐比特一致）** |
| graph（去噪循环入图） | **96.1±2.3** | 97.1 | **−45%** | **0（逐比特一致）** |

- graph 在 bf16 chunk50 同为 93.0±2.3ms；chunk 1/10/50 的 graph 延迟几乎平坦（82.5/92.2/96.1ms）——去噪的 kernel 时间本就小，剩余成本集中在 eager 段（vision 14.5 + prefill 9 + 前后处理 9 + KV 拷贝）。
- 对照 T1 的理论地板（单步 GPU kernel 忙碌 ≈65ms）：整步 96ms，距 kernel 地板约 31ms——下一步的靶子是 prefill 段的 Python 循环（同一武器可再用一次）。
- 上一节"chunk=50 时 hoist 偶发 130ms 快态"之谜解开：那是集群安静窗口的 hoist 稳态（本轮安静窗口下 hoist p50=133ms 稳定复现）。
- 摊销经济：chunk=50 每动作 3.52 → 1.92 ms（baseline → graph）。

**M4+V2 总结论**：从上游代码到 graph 化，**单步延迟 −45%（176→96ms）且输出逐比特不变**——零量化代价、零算法改动，纯执行层工程。三级阶梯各自消除一类开销：hoist 消同步与重复计算（CPU 侧），graph 消逐 kernel 启动（launch 侧）；这正是 LLM serving 引擎十年进化的微缩重演，也是概念章节（§5）"第一笔收益在粘合层/执行层"论点的完整实证。图：`fig_hoist.png`（三臂阶梯）。

## 5. 从 LLM Serving 看 VLA 推理（Mini-SGLang 概念映射，定稿）

> 源码已通读（sgl-project/mini-sglang，约 9000 行，克隆在服务器 `/storage/xuan/vlabench/code/mini-sglang`）；带读笔记见 `docs/minisglang-notes.md`（含逐文件行号引用）。README 里有本节的英文浓缩版。

概念对照表（左列 = Mini-SGLang 的真实模块，右列 = 本仓库实测）：

| Serving 概念 | Mini-SGLang 中的位置 | VLA 对应物（本文实测） |
|---|---|---|
| prefill（算力受限、按 token 预算凑批、可切块） | `scheduler/prefill.py`（PrefillAdder；ChunkedReq） | OpenVLA：每控制步全价 prefill 31 ms（16%）；SmolVLA：每 chunk 一次 9 ms |
| decode（带宽受限、逐 token） | `scheduler/decode.py`（39 行：decode 批=全部 runnable 请求） | OpenVLA：6 步自回归 127 ms（**65%**，带宽墙→INT4 降到 101 ms）；SmolVLA：10 步并行去噪 88 ms（无逐 token 墙→chunk 免费的根源） |
| 分页 KV + radix 前缀缓存 | `kvcache/radix_cache.py`（token 前缀树、页对齐、叶 LRU） | 图像 token 每步都变→跨步复用必然 miss；可缓存的只有冻结指令前缀；prefill 中 vision 仅 9–14 ms、LM 占大头→"指令前缀常驻"才值得做。SmolVLA 已在步内复用 KV（10 个去噪步共享，`fill_kv_cache`） |
| chunked prefill + 预算保护 | `PrefillAdder.reserved_size = inflight_tokens` | 新机器人的 prefill 不得吃掉正在 decode 的机器人的 KV 预算——保留逻辑原样适用 |
| continuous batching | 准入粒度 = 一次迭代（每轮重组批） | 多机器人共享单卡：batch 8 吞吐 ×6.6 且 decode 几乎不慢（带宽受限）→ 物理红利真实存在；chunk 模型 decode 长度固定（10 步）比变长 LLM 更好调度；OpenVLA 上游不支持批量生成→"每机器人一个流" |
| overlap 调度 + decode CUDA graph | `scheduler.py` 双流循环；`engine/graph.py` bs∈{1,2,4,8..256} | SmolVLA 每步 ~56 ms（约 ⅓ e2e）耗在 Python 粘合层（"other" 相位）——正是这类引擎技术消除的开销。VLA 的第一笔 serving 收益不在模型在粘合层 |

四个论点（每条都有双向支撑：引擎源码 + 本文数据）：

1. **prefill/decode 的两种 VLA 形态**。OpenVLA 是教科书形态：图像+指令前向（31 ms）后接 6 步单 token 自回归 decode（127 ms）——decode 占 65%，与 LLM 一致；于是 LLM 侧 decode 优化（paged attention、投机解码、量化带宽减半）可直接迁移，本文 INT4 使 decode 127→101 ms 正是带宽减半效应。SmolVLA 则是另一种形态：decode 不是自回归而是 10 步 flow-matching 去噪，且**全部去噪步共享一条 prefix KV cache**（源码 `fill_kv_cache=True/False` 的用法与 serving 引擎的 prefill/decode 分离完全同构）。"chunk 几乎免费"的本质：decode 并行，不存在逐 token 带宽墙。
2. **KV cache 能否跨控制步复用？** 整体不能：每步图像 token 都在变（SmolVLA 每步 `fill_kv_cache` 重建）。可复用的是**冻结前缀**（instruction/系统提示）——我们的分解显示 prefill 中 vision 仅 9–14 ms、语言主干占大头，说明值得做的是前缀缓存而非图像缓存。这是 serving 引擎 prefix caching（radix 树按 token 前缀组织、页对齐、叶 LRU）思想的 VLA 版本，收益上界可直接从本文 prefill 分解推算。
3. **continuous batching ↔ 多机器人**。Mini-SGLang 的 decode.py 用 39 行证明连续批的准入粒度是"一次迭代"。我们的 batch 数据是静态批（同进同出）；连续批解决"新机器人请求到达时不必等整批结束"。chunk 模型与它天然契合：每个请求的 decode 是固定 10 步去噪，长度完全可预测，比 LLM 的变长 decode 更好调度。OpenVLA 相反：每步 7 token 短自回归 + 上游不支持批量生成，适合"每机器人一个流"而不是凑批。
4. **调度器视角**。Mini-SGLang 的调度器在 prefill 与 decode 间分时间片（且 prefill 优先、给在飞 decode 预留页预算）；VLA 版对应问题是"谁触发重推理"：单动作模型每个控制步都全价 prefill+decode（OpenVLA 每 195 ms 一次），chunk 模型每 50 步才 prefill 一次（SmolVLA 摊销 3.5 ms/动作）。**chunk 化本质是把调度粒度从"每控制步"改成"每 chunk 周期"——这是 VLA 侧独有、LLM serving 里没有直接对应物的自由度。**

## 6. 工程发现（对复现者有用）

- lerobot 0.3.2 ↔ 0.4.4 checkpoint 格式不兼容（stats 从 config.json 移到独立 preprocessor 文件）；HF 上的 smolvla_base 是新格式，**必须 lerobot ≥ 0.4.4**；lerobot 又钉 huggingface_hub<1.0 → transformers 需 4.x（4.57.1 可用，5.x 冲突）。
- SmolVLA 出厂即混合精度（塔 bf16 + 头 fp32），层内有 `.to(q_proj.weight.dtype)` 胶水；bnb 量化必须跳过 q/k/v/o 投影，否则整型存储 dtype 毒化输入 cast。
- bnb int8 默认 threshold=6.0 在单步小 batch 负载下慢 17–34 倍；设 0 后仍劣于 INT4。
- lerobot 直接调 `module.forward(...)` 绕过 `__call__`，forward hook 不触发——计时需实例级补丁。
- 全部踩坑与修法（16 项）见 LOG.md。

## 7. Limitations

- 共享集群 GPU、无法锁频；run 前空闲检查缓解但未消除（v2 中两个配置曾现瞬态双峰延迟，复测后干净，已记录）。
- 偏差是数值层面（8 帧、固定噪声配对）；任务成功率评估（LeRobot 仿真）为 future work。
- pusht 低分辨率单相机输入 + OpenVLA 用 bridge 统计反归一化：绝对延迟不受影响，偏差数值仅在该输入分布下有意义。
- bnb int8 kernel 将 bf16 输入 cast 到 fp16 计算，"INT8" 行为并非纯 int8。
- SmolVLA base 声明三相机、实际单相机（模型零填充路径）；vision 相位含 2 个空相机槽的编码成本。
- OpenVLA 上游生成代码不支持 batch>1；batch 结论仅对 SmolVLA 成立。

## 8. 复现

```bash
# 单张 H100（全部正式数据固定同一物理卡；服务器侧脚本）
python data/fixtures.py
bash scripts/run_formal.sh        # 31 个配置（矩阵见脚本）
python plots/make_plots.py        # 7 张图 + summary.md，从 JSON 重建一切
python plots/summarize.py         # 控制台速览
```
