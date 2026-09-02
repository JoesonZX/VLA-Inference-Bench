# REPORT — VLA-Inference-Bench

> 顺畅故事线与结果。所有数字出自 `results/*.json`（30 次采样、warmup 5、固定 H100 GPU 1、帧轮换协议 v2）；汇总表由 `plots/summarize.py` / `plots/make_plots.py` 生成。英文对外版见 README。

## TL;DR（三句话结论）

1. **OpenVLA-7B 在单张 H100 上一步动作生成 ≈196 ms（BF16、14.1 GB 权重）；NF4 INT4 把权重压到 4.1 GB（−71%）而端到端延迟几乎不变**——因为 decode（占 65%，带宽受限）因 4-bit 权重流量减半而加速，抵消了 prefill（compute 受限）的解量化开销。
2. **对输出 action chunk 的 SmolVLA，chunk 长度几乎免费**：chunk 1→50，一步延迟只增加约 40 ms（decode 是并行的 flow-matching 去噪，不是自回归），摊销后每个动作的成本下降约一个数量级——"chunk 化"是多机器人部署里比量化更干净的杠杆。
3. **bitsandbytes 的 INT8 在这类单步、小 batch 的 VLA 负载上不划算**：默认 outlier 分解配置比 BF16 慢 17–34 倍，即便 threshold=0 修正后仍比 BF16 慢 30–57%（两模型一致）；INT4+NF4 才是 bitsandbytes 路线下显存与延迟的正确 tradeoff 点。

## 1. 问题：部署一个 VLA，"一步"要花多少？

VLA 以观测→动作的闭环频率运行：机器人控制通常要求 10–50 Hz 的动作输出，而 7B 级 VLM 主干的每步推理在消费级硬件上动辄数百毫秒。这把三个系统问题推到前台：

- **延迟**：单步端到端延迟决定可达控制频率；
- **显存**：权重+激活峰值决定一张卡能装下几个模型/服务几个机器人；
- **保真**：任何压缩（量化）都会改变输出动作，改变多少需要量化而不是"应该没事"。

两个杠杆——训练后量化（INT8/INT4）和动作分块（action chunking）——分别作用在显存和吞吐上。本仓库在同一张 H100 上，用同一组固定输入，对两种架构的 VLA 系统地测量这两个杠杆的收益与代价，并把延迟分解到相位级别，说明**为什么**会出现这样的结果。

## 2. 两个模型，两种"一步"

| | OpenVLA-7B | SmolVLA-450M |
|---|---|---|
| 主干 | Prismatic VLM（Llama-2-7B + SigLIP） | SmolVLM2-500M（16 层）+ 动作专家 |
| 动作形式 | 每步 **1 个动作**，离散化为 7 个 token | 每步一个 **chunk（默认 50 动作）**，连续值 |
| decode 形态 | **自回归**：逐 token 生成（prefill 后 6 次单 token 前向） | **flow-matching**：10 次去噪，每次并行生成整个 chunk |
| KV cache | prompt 内部（图像 256 token + 指令） | prefix（图像+语言+状态）算一次，**10 个去噪步全部复用** |
| 出厂精度 | BF16 | 混合（塔 BF16 + 动作头 FP32），官方 eval 按此跑 |

相位分解埋点（详见 `models/*.py` docstring）：`vision`（视觉塔）、`prefill_lm`（一次前向建 KV）、`decode`（OpenVLA=6 次单 token 前向之和；SmolVLA=10 次去噪步之和）。

## 3. 方法与协议

- **硬件**：同一张物理 H100 80GB（驱动 595.58.03）；每次 run 前 `nvidia-smi` 空闲检查写进 JSON meta（所有正式 run 均为 idle）。共享集群无法锁频，是本测量的已知限制。
- **输入**：`lerobot/pusht` episode 0 的 8 个固定帧（96×96）+ 固定指令，fixture 文件进 git；SmolVLA 用 base checkpoint 的单相机路径（缺省相机由模型自身的零填充/mask 机制处理）。OpenVLA 以 `bridge_orig` 统计反归一化（数值口径一致即可，不追求语义有效——本仓库测量的是推理行为不是任务成功率）。
- **采样**：warmup 5 + 30 次计时；CUDA event 计相位、`torch.cuda.synchronize()` 收边；报 mean±std、p50/p99。输入帧按 run 序号轮换，偏差指标覆盖全部 8 帧。
- **可复现性**：SmolVLA 的 flow-matching 噪声由 (seed, run_idx) 派生的 CPU 生成器固定——量化 run 与 FP32 参考使用逐比特相同的噪声；OpenVLA 贪心解码确定。
- **偏差**：OpenVLA 对 BF16 参考（7 token 的 top-1 不匹配率 + 反归一化后动作 L2）；SmolVLA 对 FP32（as-shipped）参考（chunk MSE + 每动作 L2）。
- **版本**：torch 2.14.0+cu130、transformers 4.46.3（OpenVLA env）/ 4.57.1（SmolVLA env）、lerobot 0.4.4、bitsandbytes 0.50.2、timm 0.9.16。OpenVLA 远程代码要求 timm∈{0.9.10–0.9.16}，已满足。

## 4. 结果

### 4.1 相位分解：一步的时间花在哪

（图：`fig_openvla_precision.png` / `fig_smolvla_precision.png` / `fig_smolvla_chunk.png`；表：`results/summary.md`）

- **OpenVLA BF16**：e2e ≈196 ms = decode 128（65%）+ prefill 31 + vision 9 + CPU 预处理 14 + 采样粘合 ~14。**自回归 decode 主导**，与 LLM serving 的经验完全一致。
- **SmolVLA（chunk=50）**：e2e ≈177 ms = decode 85（48%，10 次去噪共 8.5 ms/步）+ prefill 9 + vision 14 + 前后处理 10 + 粘合 ~60。**粘合层占比高**（lerobot 的逐层 Python 循环），是工程优化空间而非模型本质成本。
- 两个模型的 vision 塔都只占 5–8%：**视觉编码不是 VLA 推理的瓶颈，语言主干的 decode 才是**。

### 4.2 量化：买到什么，付出什么

（图：`fig_quant_tradeoff.png` / `fig_deviation.png`）

| 模型 | 精度 | e2e (ms) | 权重 (GB) | 相对 BF16/FP32 | 偏差 |
|---|---|---|---|---|---|
| OpenVLA | BF16 | ~196 | 14.1 | 基线 | — |
| OpenVLA | INT8 | ~308 | 7.4 | +57% / −47% 显存 | token 不匹配 57%，动作 L2 ≈0.10 |
| OpenVLA | INT4 | ~201 | 4.1 | **+2% / −71% 显存** | token 不匹配 57%，动作 L2 ≈0.08 |
| SmolVLA | FP32(出厂) | ~177 | 0.90 | 基线 | — |
| SmolVLA | INT8 | ~267 | 0.65 | +51% / −28% | chunk MSE ~0.013 |
| SmolVLA | INT4 | ~212 | 0.46 | +20% / −49% | chunk MSE ~0.44（chunk50） |

- **INT4 的延迟"免费"有结构性原因**：decode 是显存带宽受限，4-bit 权重把每次前向要读的权重量减半 → OpenVLA decode 128→101 ms；prefill 是算力受限，解量化是额外计算 → 31→53 ms；两者相抵，e2e 持平。**"量化让推理变快还是变慢"取决于负载落在带宽侧还是算力侧**——VLA 单步小 batch 恰好两端都有。
- **INT8 是本设置下的坏交易**：bnb 默认 threshold=6.0 的 outlier 分解让单步慢 17–34 倍（SmolVLA 3067 ms、OpenVLA 6664 ms）；threshold=0 后仍全面慢于 INT4。若 int8 kernel 输入被 cast 到 fp16（bnb 行为），精度优势也打了折扣。**结论：bitsandbytes 路线下选 NF4，不选 int8。**
- 偏差注记：OpenVLA 的 57% token 不匹配率听起来吓人，但 256-bin 离散化下相邻 token 数值上接近，反归一化后的动作 L2 只有 ~0.08–0.10（相对 ~8–10%）；INT4 的 L2 甚至小于 INT8。SmolVLA 的 INT4 chunk MSE 随 chunk 变大而放大（0.046@1 → 0.44@50），说明误差在长 chunk 上累积更明显。**数值偏差 ≠ 任务失败**——任务级成功率评估明确列为 future work。

### 4.3 chunk：被低估的免费午餐

（图：`fig_chunk_curve.png` / `fig_smolvla_chunk.png`）

- chunk 1→50：e2e 173→212 ms（+22%），但**产出动作数 ×50** → 摊销每动作成本从 173 ms 降到 4.2 ms（**41 倍**）。
- 原因：SmolVLA 的 decode 是并行 flow-matching（chunk 只是张量加宽），而 prefill/vision 完全不变。对比 OpenVLA：每步 1 个动作、每步全价重付 prefill。
- 代价不在延迟而在**控制语义**：chunk 期间动作是开环执行的（反应性下降）。这是延迟测量看不到的维度，README 的 limitations 里注明。

### 4.4 batch：多机器人共享一张卡

（图：`fig_batch.png`；OpenVLA 上游生成代码不支持 batch>1，见 LOG.md）

- SmolVLA BF16@chunk50：batch 1→2→4→8，e2e 178→210→225→265 ms，**聚合吞吐 ×6.7**（近线性）。
- 相位视角的解释：decode 是带宽受限 → batch 增大几乎不增加 decode 时间（86→106 ms）；vision 是算力受限 → 随 batch 线性增长（14→75 ms）。**瓶颈会随 batch 从 decode 侧移到 vision 侧**——继续扩 batch 的收益递减点可从相位数据推出来。

## 5. 从 LLM Serving 看 VLA 推理（Mini-SGLang 概念映射）【初稿，待共同精读 Mini-SGLang 后修订】

LLM serving 引擎（以 LMSYS 的教学实现 Mini-SGLang 为参照）用四个概念组织推理：prefill/decode 两阶段、KV cache、continuous batching、调度器。把它们映射到 VLA 循环，每个映射点都有本仓库的实测支撑：

1. **prefill/decode 的两种 VLA 形态**。OpenVLA 就是教科书形态：图像+指令的 prompt 前向（31 ms）之后是 6 步单 token 自回归 decode（128 ms）——decode 占 65%，与 LLM 一致，于是 LLM 侧的 decode 优化（Speculative decoding、paged attention、量化带宽减半）可以直接迁移，本仓库 INT4 让 decode 128→101 ms 正是"带宽减半"效应。SmolVLA 则是另一种形态：decode 不是自回归而是 10 步 flow-matching 去噪，且**所有去噪步共享同一条 prefix KV cache**（模型源码 `fill_kv_cache=True/False` 的用法与 serving 引擎的 prefill/decode 分离完全同构）。"chunk 几乎免费"（4.3）的本质：它的 decode 是并行的，不存在逐 token 的带宽墙。
2. **KV cache 能否跨控制步复用？** 不能整体复用：每个控制步的图像 token 都在变（`fill_kv_cache` 每步重建）。可复用的只有**指令前缀**——而我们的分解显示 prefill 里 vision 只占 9–14 ms、语言主干占大头，说明值得缓存的是"冻结前缀"（instruction + 任何静态系统提示），图像部分则每步重算。这正是 serving 引擎 prefix caching 思想的 VLA 版本，量化收益可从本文 prefill 分解直接推算。
3. **continuous batching ↔ 多机器人**。4.4 的 batch 数据是"静态批"：8 个请求凑齐同进同出。continuous batching 解决的是"新机器人的请求到达时不必等整批结束"——对 SmolVLA 这类 chunk 模型尤其契合（每个请求的 decode 是固定 10 步去噪，长度完全可预测，比 LLM 的变长 decode 更好调度）。OpenVLA 则相反：每步 7 token 的短自回归 + 上游代码不支持批量生成，天然适合"每机器人一个流"的部署而不是凑批。
4. **调度器的视角**。Mini-SGLang 的调度器在 prefill 与 decode 间分配时间片；VLA 版本的对应问题是谁触发重推理：单动作模型每控制步都要 prefill（OpenVLA 每 196 ms 全价一次），chunk 模型每 50 步才 prefill 一次（SmolVLA 每 4.2 ms/动作摊销）。**chunk 化本质上是把调度粒度从"每个控制步"改成"每个 chunk 周期"**，这是 VLA 侧独有的、LLM serving 里没有直接对应物的自由度。

（本节引用的 Mini-SGLang 模块细节——scheduler/KV cache manager/continuous batching 的具体实现——需要按计划共同精读源码后补充对照表。）

## 6. 工程发现（对复现者有用）

- lerobot 0.3.2 ↔ 0.4.4 checkpoint 格式不兼容（stats 从 config.json 移到独立 preprocessor 文件），pip 最新是 0.4.4 而 HF 上 smolvla_base 是新格式——**必须 0.4.4+**。
- SmolVLA 出厂即混合精度（塔 bf16 + 动作头 fp32）且层内有 `.to(q_proj.weight.dtype)` 胶水；对它做 bnb 量化必须跳过 q/k/v/o 投影（否则 uint8 存储 dtype 毒化整层输入 cast）。
- bnb int8 默认 threshold=6.0 在单步小 batch 负载下慢 17–34 倍；设 0。
- lerobot 直接调 `module.forward(...)` 绕过 `__call__`，forward hook 不触发——计时要用实例级补丁。
- 全部踩坑记录与修法见 LOG.md。

## 7. Limitations

- 共享集群 GPU，无法锁频；以 run 前空闲检查缓解，未消除。
- 偏差是数值层面的（固定输入、固定噪声）；任务成功率评估（LeRobot 仿真）是明确的 future work。
- 输入为 pusht 低分辨率单相机 + OpenVLA 用 bridge 统计反归一化——绝对延迟不受影响，但偏差数值只在该输入分布下有意义。
- bnb int8 的 kernel 将 bf16 输入 cast 到 fp16 计算（库行为），int8 行为因此并非"纯 int8"。
- SmolVLA base 声明三相机、实际喂单相机（模型自身零填充路径）；vision 相位包含 2 个空相机槽的编码成本。

## 8. 复现

```bash
# 单张 H100（全部正式数据固定同一物理卡）
python data/fixtures.py
bash scripts/run_formal.sh        # 31 个配置 × 30 次
python plots/make_plots.py        # 图 + summary.md，从 JSON 重建一切
```
