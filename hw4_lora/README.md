# ДЗ4: Full FT и LoRA

`lora_hw.ipynb` содержит воспроизводимый код и реальные результаты семи вариантов Qwen2.5-1.5B-Instruct на xLAM и MMLU.

Файлы в `outputs/`:

- `results.csv`: метрики, число обучаемых параметров, время и пик allocated GPU-памяти.
- `manual_review.csv`: ответы трёх основных моделей на десять одинаковых запросов и эталоны xLAM.
- `manual_analysis.csv`: сравнительный разбор всех десяти запросов.
- `training_history.csv`: история loss и validation.

Для повторения нужны `requirements.txt`, Jupyter и выделенная CUDA GPU с BF16. Пик Full FT в этих настройках составил 38,69 GiB allocated. Перед загрузкой xLAM нужны принятые условия Hugging Face и вход через `hf auth login`. Веса сохранены на сервере отдельно.
