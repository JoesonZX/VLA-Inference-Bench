#!/bin/bash
# OpenVLA-only rerun after the bf16 if/elif chain fix.
set -x
cd /storage/xuan/vlabench/code/VLA-Inference-Bench
export HF_HOME=/storage/xuan/vlabench/hf_cache
export CUDA_VISIBLE_DEVICES=1
OVLA=/storage/xuan/vlabench/miniconda3/envs/openvla/bin/python

$OVLA bench.py --model openvla --precision bf16 --runs 30 --warmup 5 --save-ref
$OVLA bench.py --model openvla --precision int4 --runs 30 --warmup 5 --deviation-ref results/ref_openvla_bf16_chunkna.pt
$OVLA bench.py --model openvla --precision int8 --runs 30 --warmup 5 --deviation-ref results/ref_openvla_bf16_chunkna.pt

echo OPENVLA_RERUN_DONE
