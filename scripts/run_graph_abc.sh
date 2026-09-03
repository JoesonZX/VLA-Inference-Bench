#!/bin/bash
# V2 formal: three-arm comparison (baseline / hoist / graph) back-to-back in one
# session, warmup 20 (clock settling), plus graph across precisions and chunks.
set -x
cd /storage/xuan/vlabench/code/VLA-Inference-Bench
export HF_HOME=/storage/xuan/vlabench/hf_cache
export CUDA_VISIBLE_DEVICES=1
SMOL=/storage/xuan/vlabench/miniconda3/envs/smolvla/bin/python

# headline three-arm pair, fp32 chunk 50
$SMOL bench.py --model smolvla --precision fp32 --chunk 50 --runs 30 --warmup 20 --save-ref
$SMOL bench.py --model smolvla --precision fp32 --chunk 50 --variant hoist --runs 30 --warmup 20 --deviation-ref results/ref_smolvla_fp32_chunk50.pt
$SMOL bench.py --model smolvla --precision fp32 --chunk 50 --variant graph --runs 30 --warmup 20 --deviation-ref results/ref_smolvla_fp32_chunk50.pt

# graph across chunks (fp32)
for chunk in 10 1; do
  $SMOL bench.py --model smolvla --precision fp32 --chunk $chunk --variant graph --runs 30 --warmup 20 --deviation-ref results/ref_smolvla_fp32_chunk$chunk.pt
done

# graph at bf16
$SMOL bench.py --model smolvla --precision bf16 --chunk 50 --variant graph --runs 30 --warmup 20 --deviation-ref results/ref_smolvla_fp32_chunk50.pt

echo V2_FORMAL_DONE
