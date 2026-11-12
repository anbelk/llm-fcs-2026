# ДЗ5: RL для function calling

[Ноутбук](rl_hw.ipynb) · [Отчёт PDF](report.pdf) · [Результаты](outputs/results.csv)

GRPO + LoRA для Qwen3.5-2B на ToolACE. В ноутбуке — EDA, награды, три конфигурации, результаты и разбор ошибок. Исполняемый код находится в `rl_hw.py`.

В `outputs/` сохранены метрики, журнал loss/reward/KL, график и отдельные предсказания для baseline и трёх обученных моделей. Настройки, seed и версии библиотек — в `run_info.json`. Данные и веса сохранены отдельно.

Окружение задано в `Dockerfile`. Обучение выполнялось на A100 80 GB с BF16.

```bash
python3 rl_hw.py prepare
python3 rl_hw.py run --output data/runs/grpo
```
