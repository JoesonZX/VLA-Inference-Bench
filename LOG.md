# LOG — 流水账（错误与修正都在这）

> 与 REPORT.md 的分工：这里按时间顺序记录每次踩坑和修法，包括失败尝试；REPORT.md 只写顺畅的结论故事线。

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
