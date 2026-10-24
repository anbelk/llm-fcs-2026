# ДЗ 3. SFT для function calling

[Отчет](EXPERIMENTS.md) · [Выполненный ноутбук](sft_function_calling_hw.ipynb) · [Метрики](outputs/eval_results.csv)

В ноутбуке сохранены письменные ответы, проверка маски, график loss, сравнение десяти ответов до и после SFT, тестовые метрики и пять сценариев агента.

Для повторного запуска нужны NVIDIA GPU, зависимости из первой ячейки, `smart_home.py`, `chat_ui.py` и файлы курса `data/train.jsonl`, `data/validation.jsonl`, `data/test_cases.jsonl`. Запускать из папки `hw3`. Данные и веса в комплект сдачи не входят. Код сохраняет веса и токенизатор в `outputs/full_sft`; в текущем локальном checkout этой папки нет.

[Актуальное задание и ссылки на исходные файлы](https://docs.google.com/document/d/1A3bwvZSsbFaCHYBvtfB4dIPuM4TXUW4cbQTRB3AKqy4/edit).
