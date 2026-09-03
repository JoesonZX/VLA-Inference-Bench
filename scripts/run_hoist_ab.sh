#!/bin/bash
# M4-T3: formal A/B for the hoist variant. Baseline and hoist run back-to-back
# in one session (same clock state), warmup 20 to settle H100 clock ramp.
set -x
cd /storage/xuan/vlabench/code/VLA-Inference-Bench
export HF_HOME=/storage/xuan/vlabench/hf_cache
export CUDA_VISIBLE_DEVICES=1
SMOL=/storage/xuan/vlabench/miniconda3/envs/smolvla/bin/python

for chunk in 50 10 1; do
  $SMOL bench.py --model smolvla --precision fp32 --chunk $chunk --runs 30 --warmup 20 --save-ref
  $SMOL bench.py --model smolvla --precision fp32 --chunk $chunk --variant hoist --runs 30 --warmup 20 --deviation-ref results/ref_smolvla_fp32_chunk$chunk.pt
done

$SMOL bench.py --model smolvla --precision bf16 --chunk 50 --runs 30 --warmup 20 --deviation-ref results/ref_smolvla_fp32_chunk50.pt
$SMOL bench.py --model smolvla --precision bf16 --chunk 50 --variant hoist --runs 30 --warmup 20 --deviation-ref results/ref_smolvla_fp32_chunk50.pt

echo M4_FORMAL_DONE
