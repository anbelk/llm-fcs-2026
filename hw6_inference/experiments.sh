#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_FLASHINFER_SAMPLER=0

model=(--served-model-name Qwen/Qwen3-8B)
grid=("${model[@]}" --max-num-seqs 32 --max-num-batched-tokens 8192)
rates=(10 30 60 90 120 150 180 210 240)
eagle='{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","revision":"08610ffa01dd9f16731fe8f627b85905b6aa51c4","num_speculative_tokens":3}'
ngram='{"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_min":3,"prompt_lookup_max":5}'
draft='{"method":"draft_model","model":"Qwen/Qwen3-0.6B","revision":"c1899de289a04d12100db370d81485cdf75e47ca","num_speculative_tokens":5}'

bash run.sh outputs/grid-baseline bench -- "${model[@]}"
for seqs in 8 32; do
  for tokens in 2048 8192; do
    bash run.sh "outputs/grid-$seqs-$tokens" bench -- "${model[@]}" --max-num-seqs "$seqs" --max-num-batched-tokens "$tokens"
  done
done

bash run.sh outputs/fp8-benchmark bench -- "${grid[@]}" --quantization fp8
bash run.sh outputs/eagle-benchmark bench -- "${grid[@]}" --speculative-config "$eagle"
bash run.sh outputs/control-load load "${rates[@]}" -- "${grid[@]}"
bash run.sh outputs/winner-load load "${rates[@]}" -- "${grid[@]}" --speculative-config "$eagle"

bash run.sh outputs/ngram-benchmark bench -- "${grid[@]}" --speculative-config "$ngram"
VLLM_USE_V2_MODEL_RUNNER=0 bash run.sh outputs/spec-benchmark bench -- "${grid[@]}" --speculative-config "$draft"
VLLM_USE_V2_MODEL_RUNNER=0 bash run.sh outputs/spec-load load 60 120 240 -- "${grid[@]}" --speculative-config "$draft"
