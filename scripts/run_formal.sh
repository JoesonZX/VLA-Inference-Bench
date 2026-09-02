#!/bin/bash
# Formal sampling on the fixed physical GPU 1. Sequential; every JSON carries its own idle check.
set -x
cd /storage/xuan/vlabench/code/VLA-Inference-Bench
export HF_HOME=/storage/xuan/vlabench/hf_cache
export CUDA_VISIBLE_DEVICES=1
SMOL=/storage/xuan/vlabench/miniconda3/envs/smolvla/bin/python
OVLA=/storage/xuan/vlabench/miniconda3/envs/openvla/bin/python

# ---- SmolVLA: precision x chunk matrix (fp32 doubles as the deviation reference) ----
for chunk in 50 25 10 5 1; do
  $SMOL bench.py --model smolvla --precision fp32 --chunk $chunk --runs 30 --warmup 5 --save-ref
  $SMOL bench.py --model smolvla --precision bf16 --chunk $chunk --runs 30 --warmup 5 --deviation-ref results/ref_smolvla_fp32_chunk$chunk.pt
  $SMOL bench.py --model smolvla --precision int4 --chunk $chunk --runs 30 --warmup 5 --deviation-ref results/ref_smolvla_fp32_chunk$chunk.pt
  $SMOL bench.py --model smolvla --precision int8 --chunk $chunk --runs 30 --warmup 5 --deviation-ref results/ref_smolvla_fp32_chunk$chunk.pt
done

# ---- SmolVLA: batch sweep (bf16, chunk 50) ----
for b in 2 4 8; do
  $SMOL bench.py --model smolvla --precision bf16 --chunk 50 --batch $b --runs 30 --warmup 5
done

# ---- OpenVLA: precision sweep (bf16 doubles as the reference) ----
$OVLA bench.py --model openvla --precision bf16 --runs 30 --warmup 5 --save-ref
$OVLA bench.py --model openvla --precision int4 --runs 30 --warmup 5 --deviation-ref results/ref_openvla_bf16_chunkna.pt
$OVLA bench.py --model openvla --precision int8 --runs 30 --warmup 5 --deviation-ref results/ref_openvla_bf16_chunkna.pt

echo ALL_FORMAL_RUNS_DONE
