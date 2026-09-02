# VLA-Inference-Bench 项目计划

> 版本：v1.0（2026-09-03，经 grill-me 访谈两轮共 8 问定稿）
> 本文档是工作文档（中文）；仓库对外的 README 用纯英文。

---

## 0. 一句话定位

**系统测量并对比 VLA 模型（OpenVLA-7B / SmolVLA-450M）在单 GPU 上的推理性能，量化「部署 tradeoff」：INT4/INT8 和 action chunk 各自买到什么（显存、吞吐），付出什么（延迟、输出偏差）。**

README 开头三句话结论的模板（数字由实验填充，不预设）：

1. On a single H100, OpenVLA-7B per-step action generation takes `X` ms and `Y` GB in BF16; INT4 cuts memory `Z`% at a latency cost of `W`%.
2. For chunked policies (SmolVLA), generating `N` actions per step amortizes the vision+prefill cost, raising the sustainable control rate `K`× at the price of action staleness.
3. Phase decomposition shows `[vision / prefill / decode]` dominates for OpenVLA, while SmolVLA's flow-matching decode behaves differently — implying different acceleration levers.

所有图表、章节、实验都围绕这三句话取舍。测出来 INT4 比 BF16 慢（小 batch 下 bitsandbytes 解量化开销导致的真实可能），就如实写——这正是基准仓库的价值：**结果是 finding，不是失败**。

---

## 1. 访谈决策记录（为什么是这个方案）

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| D1 | OpenVLA 没有 action chunk（每步 1 个动作 = 7 个离散 token，长度不可配置） | chunk 扫描只由 SmolVLA 承载；OpenVLA 保持单动作，作为「单动作 vs chunk 架构」的对比叙事 | 原计划的「精度 × chunk」矩阵在 OpenVLA 上物理不可行 |
| D2 | 跨 GPU 数字不可比 | **所有正式数据统一在同一张学校 H100 上采集**；笔记本 CPU 只做开发调试；Colab T4 仅作 H100 不可用时的备用 | OpenVLA BF16/INT8/INT4 的对比、跨模型对比全部合法化；$30 预算几乎不动 |
| D3 | LeRobot 仿真成功率是范围黑洞 | MVP 砍掉，只做数值偏差（token 不匹配率 + 动作 L2/MSE）；README 里诚实写为 future work | 仿真评估可能吞掉 1–2 周（总预算的一半） |
| D4 | 延迟测什么输入 | LeRobot 公开数据集（pusht）固定 episode 的固定帧 + 固定 instruction，帧索引写进配置 | 可复现 + 视觉编码计时走真实分布 |
| D5 | README 语言 | 纯英文（读者：教授/面试官/主页访客） | 国际可见性 |
| D6 | 落后时先砍什么 | **先砍 batch 实验**（叙事可用延迟数据推算讨论），再压 chunk 扫描密度到 3 个点（1/10/25） | 保剖析分解 + Mini-SGLang 章节 + README（作品集灵魂） |
| D7 | 三句话结论回答什么 | 部署 tradeoff 故事（见第 0 节） | 覆盖量化矩阵 + chunk + 显存，即大部分实验 |

## 2. 范围

### 2.1 范围内
- 模型：**OpenVLA-7B**（主角，单动作架构）、**SmolVLA-450M**（低资源 + chunk 架构承载者）；pi0/pi0FAST 明确排除出 MVP
- 测量维度：
  1. **延迟**：单步端到端 + 三段分解（见 §4.1），SmolVLA 上扫 chunk 长度
  2. **显存**：权重显存 + 运行峰值（`torch.cuda.max_memory_allocated` 等）
  3. **量化**：BF16 vs INT8 vs INT4（bitsandbytes），对延迟/显存的影响 + 输出偏差
  4. **批处理**：batch = 1/2/4/8 吞吐（**时间线上排最后，第一个被砍**）
- 硬件：学校 H100（正式数据）、笔记本 CPU（开发）、Colab T4（备用）

### 2.2 明确不做（README 的 Non-Goals 一节照抄）
训练/微调、自定义 CUDA kernel、多机分布式、LeRobot 仿真成功率（future work）、跨 GPU 对比表、pi0、vLLM/TensorRT 等推理引擎接入。

## 3. 合法对比表（协议核心）

所有正式数据出自同一张 H100，因此以下对比全部合法，README 可直接引用：

| 对比 | 合法 | 说明 |
|------|------|------|
| OpenVLA BF16 vs INT8 vs INT4（延迟/显存/偏差） | ✓ | 同卡同输入 |
| SmolVLA chunk 1→50（延迟分解变化） | ✓ | |
| SmolVLA BF16 vs INT8 vs INT4 × chunk | ✓ | M2 主矩阵 |
| OpenVLA vs SmolVLA 单步延迟/显存 | ✓ | 跨模型叙事（架构差异，非「谁好」） |
| batch 1/2/4/8 吞吐（两模型） | ✓ | 排最后，可砍 |
| H100 vs T4 数字 | ✗ | 不出现在任何表格 |

## 4. 实验协议

### 4.1 三段分解计时（本项目的工程难点，M1 核心）

两个模型架构不同，分解定义分别落地（**M1 第一天先读模型源码确认埋点**，以下按 π0-style 实现的预期写，以代码为准）：

- **OpenVLA**（自回归）：
  a. `vision`：图像预处理 + 视觉编码器 forward
  b. `prefill`：图像 token + instruction 的 prompt 前向（建 KV cache）
  c. `decode`：7 个 action token 逐个自回归解码
- **SmolVLA**（π0 风格：backbone + action expert，flow-matching 出 chunk）：
  a. `vision`：预处理 + 视觉编码
  b. `prefill`：图像 + 状态 + instruction 过 backbone
  c. `decode`：action expert 的 flow-matching 去噪循环（固定步数，非自回归）

> 这个差异本身就是 Mini-SGLang 章节的素材：同一个「decode」名字，在 VLA 世界里至少有两种形态。

实现方式优先级：
1. 首选：拆开手写循环逐步 `torch.cuda.synchronize()` + 计时（最可控）
2. 回退：整段 `generate()` 计时 + 各阶段单独微基准拼接（近似分解，README 注明方法）

### 4.2 测量纪律
- 固定 seed（flow-matching 采样等随机部分用固定 generator seed）；输入帧索引写死在配置
- 每配置：warmup 5 次 + 计时 ≥30 次，报 mean ± std + p50/p99
- 每次计时用 `torch.cuda.synchronize()` 收边；`reset_peak_memory_stats()` 在 warmup 后重置再测峰值
- **H100 共享服务器污染防护**：每次正式 run 前查 `nvidia-smi` 确认无他人进程，并把 GPU 型号/driver/CUDA/torch 版本、run 开始时的利用率快照写进 JSON meta；尽量挑空闲时段；README 的 limitations 里注明无法锁频
- 输出偏差：OpenVLA 用 action token top-1 不匹配率（7 token/步）+ 解码后连续动作 L2；SmolVLA 用 chunk MSE（对 BF16 参考输出）
- batch 实验用同 episode 的 **B 个不同帧**（不用同一帧复制，避免刻意命中 cache）

### 4.3 数据与产物
- 一切结果存 `results/*.json`（schema 见下），**JSON 与生成的图都进 git**；图只由脚本从 JSON 生成，禁止手改
- HuggingFace 权重缓存到服务器本地盘，避免每次重下 16GB

```json
{
  "meta": {"model": "openvla-7b", "precision": "int4", "quant_backend": "bnb-nf4",
           "gpu": "NVIDIA H100 80GB HBM3", "driver": "...", "cuda": "...", "torch": "...",
           "timestamp": "...", "gpu_idle_check": true},
  "config": {"batch": 1, "chunk": null, "runs": 30, "warmup": 5, "seed": 0,
             "input": {"dataset": "lerobot/pusht", "episode": 5, "frames": [10, 25, 40]}},
  "latency_ms": {"mean": 0, "std": 0, "p50": 0, "p99": 0,
                 "phases": {"vision": {"mean": 0, "std": 0},
                            "prefill": {"mean": 0, "std": 0},
                            "decode":  {"mean": 0, "std": 0}}},
  "memory_gb": {"weights": 0, "peak_allocated": 0, "peak_reserved": 0},
  "deviation": {"token_mismatch_rate": null, "action_l2": null, "chunk_mse": null}
}
```

## 5. 仓库骨架（M0 搭建，之后只加肉）

```
VLA-Inference-Bench/
├── bench.py              # 唯一入口 CLI：--model --precision --batch --chunk --runs --device
├── models/
│   ├── openvla.py        # 加载器 + 分阶段 forward（暴露 step_fn 与埋点）
│   └── smolvla.py        # 同上
├── benchmarks/
│   ├── latency.py        # 三段分解计时
│   ├── memory.py
│   ├── deviation.py
│   └── batching.py       # M3，可整体砍
├── data/fixtures.py      # 固定帧索引、LeRobot 数据集下载/缓存
├── results/              # JSON（进 git）
├── plots/make_plots.py   # JSON → 图
├── docs/minisglang-notes.md  # Mini-SGLang 阅读笔记（章节的原料）
├── README.md             # 纯英文
├── PLAN.md               # 本文档
└── requirements.txt      # 锁版本
```

原则：`bench.py` 一个入口跑一切；不写抽象基类、不做插件系统、不 prematurely 配置化。

## 6. 里程碑（总预算 36h，每周 ≤6h）

### M0（第 1 周，6h）：骨架 + SmolVLA CPU 跑通
- [ ] git init、push GitHub、MIT License、README stub（含协议一节）、requirements 锁版本（2h）
- [ ] SmolVLA 笔记本 CPU 单步推理：先随机张量冒烟，再接 pusht 固定帧（2.5h）
- [ ] `python bench.py --model smolvla --device cpu` 输出延迟表 + JSON 落盘（1.5h）
- **完成标准**：命令在 CPU 上出延迟 JSON；显存维度允许 N/A（无 CUDA）
- **收尾动作：commit + README 加「What this measures」一段**

### M1（第 2–3 周，12h）：OpenVLA 上 H100 + 三段分解
- [ ] H100 会话 #1（~1.5h）：装环境、HF 登录同意 OpenVLA license、下载权重、BF16+INT4 冒烟
- [ ] 读两个模型的推理源码，确定埋点，实现三段计时（6h，最难的部分，留足余量）
- [ ] H100 会话 #2（~1.5h）：OpenVLA 三精度 + SmolVLA 的正式采样，JSON 入库
- [ ] 顺手开始读 Mini-SGLang（见 §7，每周塞 1h 碎片时间，不占块时间）
- **完成标准**：`results/` 里有 H100 上 OpenVLA×3 精度、SmolVLA 的三段分解 JSON
- **收尾动作：commit + README 加延迟剖析第一张表**

### M2（第 4–5 周，12h）：量化矩阵 + 出图
- [ ] SmolVLA 精度 × chunk 矩阵、显存曲线、偏差指标（6h，含一次 H100 会话）
- [ ] `plots/make_plots.py` 从 JSON 自动出全部图（3h）
- [ ] README Results 章节英文初稿：图 + 每图两三句解读（3h）
- **完成标准**：删掉 `results/` 后重跑脚本可完整重建所有图；Results 初稿成形
- **收尾动作：commit + README Results 定稿**

### M3（第 6 周，6h）：batch + 概念章节 + 定稿
- [ ] batch=1/2/4/8 吞吐（2h；**落后即砍，砍后用延迟数据推算讨论**）
- [ ] Mini-SGLang 章节成文（2.5h，从 docs/minisglang-notes.md 提炼）
- [ ] README 定稿：三句话结论、复现命令、limitations、future work（1.5h）
- **完成标准**：陌生人 clone 后按 README 命令可复现（同硬件下）；挂上 joesonzx.github.io

## 7. Mini-SGLang 工作流（仓库的灵魂章节，独立工作流）

1. **读什么**：LMSYS 的 Mini-SGLang 教学引擎源码，重点四个模块——scheduler（prefill/decode 两阶段调度）、KV cache 管理、continuous batching、model runner
2. **怎么读**：第 3 周起每周 1h 碎片时间，边读边在 `docs/minisglang-notes.md` 记三件事：模块做什么、关键数据结构、和 HF `generate()` 的差别
3. **写什么**（README 章节《From LLM Serving to VLA Inference》，纯英文）：
   - prefill/decode 在 VLA 循环里的对应物——并用本项目实测数据说明 OpenVLA 的自回归 decode 与 SmolVLA 的 flow-matching decode 是两种不同的「decode」
   - KV cache 跨步复用为什么难：图像 token 每步都变 → cache 只剩 instruction 前缀可复用，实测这个前缀占 prefill 的比例
   - continuous batching 对多机器人部署意味着什么：接 M3 batch 实验的数据讨论
4. **顺序纪律**：先读懂、后动笔；章节里的每个 claim 都要能指向本项目的一张图或一个数

## 8. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| H100 被他人占用/排队 | 中 | 空闲时段跑 + `nvidia-smi` 空闲检查进协议；T4 备用跑 SmolVLA；最坏 RunPod 4090（~$5–8，预算内） |
| bitsandbytes INT4/INT8 延迟反超 BF16 | 高 | 已定为 finding 而非失败（三句话结论第 1 句就是这个 tradeoff） |
| OpenVLA 三段埋点比预期难 | 中 | 回退方案：整段 generate 计时 + 阶段微基准拼接，README 注明近似 |
| SmolVLA 实现细节与预期不符（flow-matching/自回归） | 中 | M1 第一天读源码确认；分解定义跟着代码走，架构差异反而成叙事素材 |
| 第 4 周进度落后 | 中 | 砍单顺序已锁定：batch → chunk 密度（3 点）→ 其他一律不砍 |
| 36h 总预算不够 | 中 | M0/M1 的 CPU 部分可在零成本时推进；H100 会话合并（一次跑完 M1+M2 采样） |

## 9. 预算

- H100：学校免费（2 次会话 × ~1.5h，纯计算 <30 分钟，大头是下载与装环境）
- 备用金：$30 上限不动；RunPod 4090 仅在 H100 通道彻底失效时启用（预计 $5–8 封顶）

---

## 附：下一步（M0 第 1 天）

1. `git init` + GitHub 建仓 + MIT License
2. `pip install lerobot torch` 锁版本写 requirements
3. 30 行脚本：随机张量 → SmolVLA → 一个 action chunk → print 计时（先求「跑起来」，再求「测得对」）
