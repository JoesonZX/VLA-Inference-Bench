# REPORT — VLA-Inference-Bench

> 顺畅故事线与结果（每张图、每个数都以 results/*.json 为唯一来源）。状态：骨架，数字待正式采样填充。

## 1. 问题：部署一个 VLA，一步要花多少？

（叙事：控制频率要求 vs 单步端到延迟；显存决定单卡能塞几个模型/几个机器人；量化与 chunk 是两大杠杆。）

## 2. 方法

- 固定输入（lerobot/pusht episode 0 的 8 帧 + 固定指令），固定噪声 seed；warmup 5 + 30 次采样；同一张 H100 物理卡，run 前空闲检查写进 JSON。
- 相位分解：CUDA event + forward hook（每模型的具体埋点定义见 models/*.py docstring 与 LOG.md）。
- 偏差：对同一输入集合的 BF16 参考输出算 token 不匹配率 / 动作 L2 / chunk MSE。

## 3. 两个模型的"一步"长什么样（相位分解）

（fig_phase_breakdown_openvla.png / fig_phase_breakdown_smolvla.png）

要点预告：OpenVLA 的 decode 是 6 次逐 token 自回归；SmolVLA 的 decode 是 10 次 flow-matching 去噪、全部复用同一条 prefix KV cache —— 同名"decode"，两种形态。

## 4. 量化买到什么、付出什么

（fig_quant_tradeoff.png / fig_deviation.png）

## 5. chunk 长度的杠杆

（fig_chunk_curve.png）

## 6. 从 LLM Serving 看 VLA 推理（Mini-SGLang 概念映射）

（待 M3：prefill/decode、KV cache 跨步复用、continuous batching ↔ 多机器人；用本仓库实测数据支撑每个 claim。）

## 7. Limitations
