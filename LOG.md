# LOG — 流水账（错误与修正都在这）

> 与 REPORT.md 的分工：这里按时间顺序记录每次踩坑和修法，包括失败尝试；REPORT.md 只写顺畅的结论故事线。

## Session 2 — 2026-09-03（M4：粘合层优化）

**[M4-T1 剖析结果：先测量再动手的价值兑现]**
- 组件级（CUDA event 包裹）：`embed_prefix` 17.3ms（含 vision ~14）、`embed_suffix` **23.9ms**（10 次去噪步的动作/时间嵌入拼装）、`make_att_2d_masks` **仅 0.2ms**——原计划 V3"缓存 mask"是伪靶子，剖析避免了优化错对象。
- 算子级（torch.profiler 单步）：wall ~211ms 中 **GPU 仅忙 64.9ms，launch gap ≈146ms**；单步 **11253 次 kernel launch**、**32 次 cudaStreamSynchronize（CPU 侧 52ms）**、`pow`×927（时间步正弦嵌入每步重算）。结论：单步延迟是**启动受限**，不是算力/带宽受限。
- 三个真靶子：① `while tensor >= tensor` 循环条件（每轮迭代强制 GPU→CPU 同步 ×10）；② `embed_suffix` 里 `create_sinusoidal_pos_embedding` 每控制步重算 ×10（输入只依赖 10 个固定时间步值）；③ `torch.tensor(python列表, device=cuda)` 的同步 H2D ×10（att_masks）。
- 剖析脚本 `benchmarks/profile_glue.py`，chrome trace 存 `results/profile/`。

**[决定] V1（torch.compile）按数据跳过**：11253 次 launch 的来源是 lerobot `SmolVLMWithExpertModel.forward` 的**手写 16 层 Python 循环**（逐层 attn/cross-attn/MLP 手动调用），compile 在这里必然大量图断裂；mode=reduce-overhead（CUDA graph）需要整图捕获更不可行。V2（手写 CUDA graph capture denoise_step）留作 stretch。先做 V3（hoist：预计算+浮点循环）。

**[错误 17] hoist 首版 Euler 符号反了**——上游 `dt = -1/num_steps`（负号：从噪声 t=1 积分到动作 t=0），我缓存成 `+1/n`：chunk MSE 17.3（积分方向反了输出全是垃圾）。修法：时间步序列生成用 `+1/n` 递减，Euler 步长用 `-1/n`。修复后 **deviation 全 0（与上游逐比特一致）**——预计算 fp64 正弦嵌入 + python 浮点时间步 + 函数式 Euler，数值上完全等价。

**[错误 18] create_sinusoidal_pos_embedding 要 torch.device 对象**，传字符串报 `'str' object has no attribute 'type'`。

**[发现 B：H100 时钟斜坡测量陷阱]** hoist 冒烟两次运行 decode 相差 40ms（89 vs 131ms），两组各自 std 都很紧——不是噪声而是**时钟档不同**：前一次运行跟着两次失败尝试（GPU 已热/boost），后一次冷启动（低时钟稳态）。教训进协议：**变体 A/B 必须 baseline 与 hoist 同会话背靠背跑，warmup ≥20**。

**[发现 C：相位事件和的归因假象]** hoist 后 decode_step 事件和（97.7ms）反而高于基线（87.6ms），但 e2e 降 36ms——基线里 while 条件的同步间隙发生在 forward 窗口**之间**（不计入任何相位），hoist 后 GPU 连续执行、间隙计入窗口**之内**。相位数据用于结构分析，**跨变体对比只看墙钟 e2e**。

**[M4-T2 V3 hoist 实现要点]**（models/smolvla.py `_sample_actions_hoist`）
- 时间步正弦嵌入：按 (num_steps, chunk, bsize, dtype) 缓存，进程内只算一次
- att/pad masks 与 suffix 2D 掩码、position_ids、prefix_pad_2d_masks：控制步内算一次，10 个去噪步复用
- `while tensor` → `for i in range(num_steps)`（python 浮点），Euler 步长 python 标量
- 保留上游语义：`suffix_out.to(action_out_proj.weight.dtype)`（混合精度出厂态必需，uniform 低精度下为 no-op）

**[M4-T3 A/B 正式结果（同会话背靠背、warmup 20、30 次）]**

| 配置 | baseline | hoist | p50 对比 | 偏差 |
|---|---|---|---|---|
| fp32 chunk1 | 181.7±18.9 | 129.7±1.9 | 171.3→129.7（−24%） | 0 |
| fp32 chunk10 | 172.5±2.1 | 162.9±14.3 | 172.4→170.4 | 0 |
| fp32 chunk50 | 179.8±10.6 | 165.8±19.3 | 177.8→177.6 | 0 |
| bf16 chunk50 | 177.8±2.7 | 162.1±19.6 | 178.7→174.4 | bf16 同级 |

解读：粘合层收益在**启动受限**配置（chunk=1）稳定兑现 −24% 且方差 ±19→±2（基线的抖动本身就是同步点）；chunk=50 时被删的 CPU 工作原本被 GPU 异步执行吸收（p50 持平，hoist 均值低是偶发 ~130ms 快态——CUDA graph 的潜在空间）。此结论写入 REPORT §4.5。

**[V2 CUDA graph：设计、实现与结果]**
- 三个可捕获前提（读源码确立）：denoze 期间 KV 字典只读不写（纯输入）；prefix 长度恒定（固定 prompt+图像尺寸 → 掩码/位置/时间嵌入全恒定）；hoist 循环无 host 同步。
- 实现（models/smolvla.py `_sample_actions_graph`）：每步 eager prefill → 新 KV `copy_` 进预分配静态缓冲（地址跨步稳定）→ 10 步去噪循环 `torch.cuda.CUDAGraph` 捕获一次（side stream 预热 3 次后捕获）→ 每步拷噪声进静态输入缓冲、`replay()`、克隆静态输出。capture 必须在 `no_grad` 而非 `inference_mode` 下（step() 按变体切换上下文）。
- **首测即通**，冒烟 93.9±1.6ms。
- 正式三臂（安静窗口、warmup 20、30 次）：baseline 175.7±1.4 / hoist 132.9±2.7 / **graph 96.1±2.3**（p50 97.1，−45%）；graph 在 bf16 c50 = 93.0±2.3，chunk 1/10/50 平坦（82.5/92.2/96.1）。**全部偏差 = 0（逐比特一致）**。
- 顺带解谜：上一轮 A/B 的"chunk50 偶发 130ms 快态"= 安静窗口的 hoist 稳态（本轮 p50 133 稳定复现）。
- 测量注意：replay 不走 Python → 相位钩子看不到 decode（JSON 里 decode_step 缺失是预期）；跨变体只比墙钟 e2e。
- 出图坑：make_plots 在 graph 数据上 KeyError（stacked 直接索引 decode 相位）→ `.get(ph, 0)`；精度矩阵图/散点图排除非 baseline 变体；三臂图两个百分比标签重叠 → 纵向错开 + margins 0.2。
- 剩余空间：96ms 距 T1 的 kernel 地板（~65ms）还差 ~31ms，集中在 eager 的 prefill 段 Python 循环——同一武器（graph 化 prefill）可再用，列为 future work。

## Session 1 — 2026-09-03（夜间自主运行）

**[00:xx] 环境盘点**
- 本地 Windows：无 Python（Microsoft Store 占位符），git 2.49、OpenSSH 可用 → 代码在本地写、scp 上服务器跑，不在本地装 Python。
- 服务器 `xuan@208.64.254.171`（hostname sn4622121353）：8× H100 80GB，driver 595.58.03，python 3.10.12，128 CPU / 2TB RAM，可访问 HF。
- GPU 占用快照：GPU 0（43MiB）、1/2/5（10MiB）空闲；3/4 有他人 python（36.9GB each）；6/7 是 vLLM worker（75GB each）。
- **协议决定：所有正式数据固定在物理 GPU 1 采集。**

**[错误 1] `python3 -m venv` 失败** — Debian 缺 `python3-venv`（ensurepip 不可用），无 root 装不了。
→ **修法**：改用 Miniconda 装到 `/storage/xuan/vlabench/miniconda3`（用户指示存储一律走 `/storage/xuan`，家目录 95% 满本来也不合适）。

**[错误 2] `conda create` 报 `CondaToSNonInteractiveError`** — conda 26.x 要求先接受 Anaconda 默认渠道 ToS。
→ **修法**：`conda tos accept --override-channels --channel .../pkgs/main` 和 `.../pkgs/r` 各跑一次，之后正常。

**[00:xx] 环境与下载（全部成功）**
- conda env `smolvla`：python 3.11 + torch 2.14.0+cu130 + lerobot 0.3.2 + bitsandbytes 0.50.2 + matplotlib。
- conda env `openvla`：python 3.11 + torch 2.14.0+cu130 + transformers 4.46.3 + bnb 0.50.2 + accelerate 等。
- 权重缓存 `/storage/xuan/vlabench/hf_cache`（HF_HOME）：openvla-7b 15GB、lerobot/smolvla_base 873MB、lerobot/pusht 7.5MB。
- HuggingFace 三项全部**匿名可下**（200），不需要 token。

**[错误 3] timm 版本硬检查** — OpenVLA 远程代码 `modeling_prismatic.py` 在 `__init__` 里硬性要求 timm ∈ {0.9.10, 0.9.11, 0.9.12, 0.9.16}，否则 NotImplementedError；pip 默认装的是最新版。
→ **修法**：`pip install timm==0.9.16`。transformers 4.40.1/tokenizers 0.19.1 只是警告不阻断，先用 4.46.3 实测，若 generate 崩再降级（见后续记录）。

**[00:xx] OpenVLA 源码阅读（埋点设计依据）**
- 模块树：`model.vision_backbone`（SigLIP 塔）→ `model.projector` → `model.language_model`（LlamaForCausalLM）。
- forward 分支：多模态分支（prefill）跑一次 vision_backbone + projector，把 256 个图像 patch embedding 拼进 inputs_embeds 过 LM；`input_ids.shape[1]==1` 分支（decode）只过 LM。**视觉只在 prefill 跑一次**。
- `predict_action` 逐行复制进 adapter（补 29871 空 token → generate(max_new_tokens=7) → `vocab_size - token_id - 1` 反查 bin_centers → q01/q99 反归一化）。保留其 quirk：append 29871 时不扩展 attention_mask（与上游行为一致）。
- **发现**：`prepare_inputs_for_generation` 对 batch>1 直接 raise——**OpenVLA 上游代码不支持批量生成**。batch 实验只能对 SmolVLA 做，OpenVLA 报告里作为 limitation 写明。
- 三段分解：`vision_backbone` / `vision_projector` / `prefill_lm`（多模态前向）/ `decode`（6 次 seq=1 前向，第 7 个 token 由 prefill 前向产出）。

**[00:xx] SmolVLA 源码阅读（lerobot 0.3.2）**
- `SmolVLAPolicy.model = VLAFlowMatching`：`vlm_with_expert`（SmolVLM2 塔 + 动作专家）、`state_proj`、`action_in/out_proj`、`action_time_mlp_*`。
- **重要发现**：`sample_actions` = prefill（`fill_kv_cache=True` 算 prefix KV cache）+ `config.num_steps`（默认10）次去噪步（`fill_kv_cache=False` 复用 cache，只过 expert 跑 action 后缀）。**模型内部就是 prefill + KV cache + decode 结构**，直接成为概念章节素材。
- chunk 覆盖：`policy.config.chunk_size` 同时控制噪声形状与后缀长度，可安全扫描。
- 可复现性：`predict_action_chunk(batch, noise=...)` 支持显式噪声；用 CPU generator 以 (seed, run_idx) 派生固定噪声，量化 run 与 BF16 参考逐比特同噪声。
- batch：原生支持（bsize 取自 state.shape[0]）。
- 量化：手动递归替换 nn.Linear → bnb Linear8bitLt / Linear4bit(nf4)（lerobot 不走 transformers 的量化加载路径）。

**[决定] dtype 策略**：SmolVLA BF16 用 `model.to(torch.bfloat16)`（部署视角）；若 flow-matching 循环里 fp32 噪声与 bf16 权重相乘报 dtype 错误，则噪声转成参数 dtype（代码已按参数 dtype 生成噪声）并记录。实测见 Session 1 冒烟部分。

**[错误 4] smolvla 环境缺 transformers** — lerobot 的 pip 包不把 transformers 列为强依赖。
→ **修法**：`pip install transformers accelerate`（装到 5.16.1，太新）。

**[错误 5] SmolVLM processor 要 num2words** — `ImportError: Package num2words is required`。
→ **修法**：`pip install num2words`。

**[错误 6] lerobot 0.3.2 读不懂 smolvla_base 新 checkpoint 格式** — 新格式把归一化统计量放在独立的 `policy_preprocessor/postprocessor` 文件里，0.3.2 期望 stats 嵌在 config.json → `normalize_inputs` 断言 stats 为 inf。
→ **修法**：升级 `pip install -U lerobot`（0.3.2 → 0.4.4），改用官方新 API：`SmolVLAPolicy.from_pretrained` + `make_pre_post_processors(config, model_id)` 三段式（preprocess → policy → postprocess）。API 面不变（predict_action_chunk / fill_kv_cache 都在）。

**[错误 7] lerobot 0.4.4 与 transformers 5.16 冲突** — lerobot 钉 huggingface_hub<1.0，transformers 5.x 需要 hub>=1.5 → `cannot import name 'is_offline_mode'`。
→ **修法**：smolvla 环境降级 `transformers==4.57.1`（与 hub 0.35.3 兼容，SmolVLM 处理器可用）。**教训：推理仓库的依赖矩阵要么按 checkpoint 时代整套钉版本，要么准备好来回试。**

**[错误 8] `config.image_features[0]` KeyError** — 0.4.4 里 image_features 是 dict 不是 list。
→ **修法**：`next(iter(...))`。

**[错误 9] scp 目标路径写成 `models/../`** — 新适配器落到了仓库根目录而不是 `models/`，服务器上跑的一直是旧版（旧版"按首个参数 dtype 批量 cast"在 fp32 模式把 state 误 cast 成 bf16 → state_proj fp32 报 dtype 错）。
→ **修法**：显式传 `...:/models/smolvla.py`；删掉根目录杂散文件。**教训：scp 后应立即 grep 验证远端文件内容。**

**[发现 A] SmolVLA 是混合精度加载** — 塔级参数（SmolVLM2 预训练权重）bf16，动作头投影（checkpoint 微调部分）fp32。lerobot 官方 eval 就这么跑；层内还有 `.to(dtype=q_proj.weight.dtype)` 胶水。因此：
- **fp32 基线 = as-shipped 混合精度（塔 bf16 + 头 fp32），噪声 fp32** —— 这才是"官方口径"基线；
- 低精度模式 = 全模型统一 bf16/量化 + 复刻 `sample_actions`/`denoise_step`（去掉上游硬编码的 `suffix_out.to(float32)`，Euler 步显式 cast 防 in-place 类型提升错误）。复刻代码在 models/smolvla.py，与上游逐行对照过。

**[错误 10] fp32 噪声被误 cast 成 bf16** — `next(parameters()).dtype` 在 fp32 模式返回塔的 bf16，噪声被错误降精度 → action_in_proj(fp32) 报错。
→ **修法**：仅低精度模式 cast 噪声；fp32 恒为 fp32。

**[错误 11] 相位钩子只打出 vision，prefill/decode 缺失** — lerobot 源码直接调 `vlm_with_expert.forward(...)`（绕过 `nn.Module.__call__`），forward hook 不触发。
→ **修法**：对 vlm_with_expert 用实例级 forward 猴补丁计时（按 `fill_kv_cache` 分类 prefill_lm/decode_step），vision 塔仍用常规 hook。

**[错误 12] int4 报 `baddbmm_cuda not implemented for 'Byte'`** — lerobot 层内胶水读 `q_proj.weight.dtype`，bnb Params4bit 的存储 dtype 是 uint8，整层输入被 cast 成 uint8。
→ **修法**：量化排除 q/k/v/o 四种注意力投影（保持 bf16），MLP 与其余 Linear 全量化；int8 同样排除（Int8Params dtype 是 int8，同病）。排除集对 int8/int4 一致，保证可比。

**[错误 13] bnb int8 默认 threshold=6.0 病态慢** — SmolVLA int8 首测 3067ms/步（bf16 的 17 倍），OpenVLA int8 首测 6664ms/步（34 倍）：outlier 分解把每层切成大量小 matmul。
→ **修法**：`threshold=0.0`（SmolVLA）/`llm_int8_threshold=0.0`（OpenVLA）。修后 SmolVLA int8 = 266ms，OpenVLA int8 = 399ms。**这本身是报告素材：bnb int8 默认配置在单步小 batch VLA 负载下不可用。**

**[错误 14] OpenVLA int4 的 conv 层 dtype 半精度** — 量化加载默认非量化模块 fp16，pixel_values 是 bf16 → conv 报 c10::Half vs BFloat16。
→ **修法**：所有精度统一传 `torch_dtype=torch.bfloat16`。

**[冒烟结果矩阵（H100, GPU1, chunk50, batch1, 3 次粗采样）]**

| 模型 | 精度 | e2e ms | 权重 GB |
|---|---|---|---|
| OpenVLA-7B | BF16 | 198 | 14.09 |
| OpenVLA-7B | INT4 | 241 | 4.08 |
| OpenVLA-7B | INT8 | 399 | 7.43 |
| SmolVLA | FP32(混合) | 207 | 0.90 |
| SmolVLA | BF16 | 178 | 0.89 |
| SmolVLA | INT4 | 214 | 0.46 |
| SmolVLA | INT8 | 266 | 0.65 |

**[错误 15] OpenVLA bf16 掉进 else: raise** — 修错误 14 时把 if/elif 链改断（`if int8 / elif int4 / else raise`，bf16 无分支直接抛 ValueError）→ v1 正式跑 OpenVLA 三连挂。
→ **修法**：else 分支只在非 bf16 时 raise。**教训：改完分支逻辑应该三个精度各冒烟 30 秒再挂后台。**

**[错误 16] 偏差指标建立在单一输入上（协议级 bug）** — batch=1 时每步都用 fixture 第 0 帧；OpenVLA 贪心解码是确定性的 → 30 次"采样"实为同一输入重复 30 次（v1 数据 action_l2_std≈0 暴露了这一点）。
→ **修法**：按 `run_idx % 8` 轮换输入帧（batch 行也保持互不相同），v2 全量重跑。参考输出与量化 run 的 run_idx 对齐，配对比较依然成立。

**[注意] bnb int8 会把 bf16 输入 cast 到 fp16**（`MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization`）——int8 kernel 只吃 fp16 输入，属预期行为，报告 limitations 里注明。

**[v2 收尾记录]**
- v2 全量重跑完成（31 配置，全部 idle=True）。两个配置出现瞬态双峰延迟（bf16_chunk5 p50=224/max=230 且两次复现；int4_chunk25 ±26.5）→ 均为共享集群瞬态干扰：chunk5 以 50 次采样复测后干净（177.1±2.8），int4_chunk25 复测 209.9±2.8。
- 期间顺手修了：deviation 函数按 run_idx 配对取 min(N)（50 次 run 对 30 次参考的形状不匹配）；make_plots 输出目录笔误（图写到了仓库根）。
- 出图两轮视觉审查后修正：堆叠图图例与 wall-e2e 样例冲突（移到图外下方）、INT8 顶部标记裁切（margins 12%）、堆叠不含 CPU 前后处理导致黑线悬空（加入 cpu_pre/cpu_post 段，堆叠= e2e）、散点图 SmolVLA FP32/BF16 标注重叠（错开 va）+ 无图例（补）。
- 最终产物：7 张图 + results/summary.md + 31 个 JSON + 6 个参考 .pt，全部回同步到本地仓库并提交。

**[Session 1 最终数据快照（协议 v2，详见 REPORT.md）]**

| 模型 | 精度 | e2e ms | 权重 GB | 偏差 |
|---|---|---|---|---|
| OpenVLA-7B | BF16 | 195.0±2.0 | 14.09 | — |
| OpenVLA-7B | INT8 | 308.2±4.8 | 7.43 | L2 0.147 |
| OpenVLA-7B | INT4 | 201.3±2.0 | 4.08 | L2 0.523 |
| SmolVLA | FP32(出厂) | 176.2±1.9 | 0.90 | — |
| SmolVLA | BF16 | 178.8±2.9 | 0.89 | MSE 0.0024 |
| SmolVLA | INT8 | 267.1±2.1 | 0.65 | MSE 0.035 |
| SmolVLA | INT4 | 213.9±5.7 | 0.46 | MSE 0.244 |
| SmolVLA bf16 batch 1→8 | | 179→270 | | 吞吐 ×6.63 |
| SmolVLA fp32 chunk 1→50 | | 168.4→176.2 | | 摊销 47.8× |



