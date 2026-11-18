# ДЗ 5. RL для вызова функций

[Отчёт](outputs/report.pdf) · [Эксперименты](EXPERIMENTS.md)

## Окружение

```bash
cd hw5_rl
python3 -m venv .venv
.venv/bin/pip install uv
curl -fL 'https://wheels.vllm.ai/0b3ba88f165976e77ca5e6a7a3f5bba4562b80af/vllm-0.22.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' -o vllm-0.22.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
printf 'numpy==2.2.6\n' > overrides.txt
.venv/bin/uv pip install --python .venv/bin/python --torch-backend=cu129 \
  --override overrides.txt ./vllm-0.22.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
  torch==2.11.0+cu129 trl==1.16.0 transformers==5.9.0 peft==0.18.1 datasets==4.7.0 \
  accelerate==1.10.1 flash-linear-attention==0.4.1 bfcl-eval==2026.3.23 \
  soundfile==0.14.0 scipy==1.16.2 typer==0.29.2 ninja==1.13.2
export PATH="$PWD/.venv/bin:$PATH"
export HF_HOME="$PWD/hf-cache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1

mkdir -p headers && cd headers
apt-get download libpython3.12-dev
dpkg-deb -x libpython3.12-dev_*.deb .
cd ..
export CPATH="$PWD/headers/usr/include/python3.12:$PWD/headers/usr/include"
```

## Данные

```bash
curl -fL https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data.json -o toolace.json
python prepare.py toolace.json data --seed 42
```

## Baseline и три коротких запуска

```bash
python evaluate.py Qwen/Qwen3.5-2B --output outputs/reproduction/base-bfcl

for run in grpo-strict cispo-strict cispo-partial; do
  python train.py "$run" --output "outputs/reproduction/$run"
  python evaluate.py "outputs/reproduction/$run/merged" --output "outputs/reproduction/$run-bfcl"
done
```

## Длинный GRPO

```bash
.venv/bin/pip install --no-deps --target speed-wheels/causal-conv1d \
  'https://wheels.astral.sh/artifacts/6772c983a278ca264d2af8178e26a9ad6af6c5c3ae826f5734c287f42949b5bf/causal_conv1d-1.7.0+cu.12.9.torch.2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl#sha256=6772c983a278ca264d2af8178e26a9ad6af6c5c3ae826f5734c287f42949b5bf'
export PYTHONPATH="$PWD/speed-wheels/causal-conv1d${PYTHONPATH:+:$PYTHONPATH}"
python train.py grpo-partial-long --output outputs/reproduction/grpo-partial-long
unset PYTHONPATH
python evaluate.py outputs/reproduction/grpo-partial-long/merged --output outputs/reproduction/grpo-partial-long-bfcl
```
