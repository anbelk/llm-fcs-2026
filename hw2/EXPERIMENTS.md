# Эксперименты

## Постановка

Ускорение обучения модели Qwen3 из шаблона на одной NVIDIA A100 80 GB. Модель, токенизатор, данные и разделение на обучение и валидацию не менялись. Длина последовательности — 512; BF16, оптимизатор, learning rate, scheduler и seed оставлены как в шаблоне.

Бейзлайн и итоговые конфигурации обучаются с нуля по 300 секунд. Главная метрика — полезные токены в секунду из `summary.json`; первые 10 шагов исключаются из замера по правилам шаблона. Память — пиковая выделенная память во время обучения. Поиск начинался с 90-секундных запусков, затем использовались проверки на 120 и 180 секунд.

## Результаты

Бейзлайн: SDPA, batch 32, accumulation 2; остальные оптимизации выключены. В таблицах B/GA — batch size / gradient accumulation. FA2 — FlashAttention 2, PF — padding-free, P — packing BFD, C — torch.compile, AC — activation checkpointing, L — Liger. Для packing режим padding-free включается автоматически.

L-CE означает только fused linear cross-entropy, L-noCE — RoPE, RMSNorm и SwiGLU без fused loss. C-auto — режим `max-autotune-no-cudagraphs`; обычный C — `default`. Activation offloading во всех экспериментах выключен.

### Первый этап

| Вариант | B/GA | Время, с | Токенов/с | Память, GiB | Val loss ↓ | Шаги |
|---|---:|---:|---:|---:|---:|---:|
| [Бейзлайн](results/00_baseline_300/summary.json) | 32/2 | 300.9 | 7002 | 41.2 | 6.62 | 366 |
| [SDPA](results/01_batch16_ga4_90/summary.json) | 16/4 | 90.3 | 7991 | 24.2 | 8.07 | 124 |
| [SDPA + AC](results/03_checkpointing_90/summary.json) | 32/2 | 90.5 | 5953 | 17.4 | 8.24 | 92 |
| [FA2](results/04_fa2_90/summary.json) | 32/2 | 90.4 | 7620 | 39.3 | 8.09 | 118 |
| [FA2 + PF](results/06_fa2_padding_free_90/summary.json) | 32/2 | 90.2 | 19291 | 15.1 | 6.88 | 303 |
| [FA2 + P](results/07_fa2_packing_90/summary.json) | 32/2 | 90.2 | 22316 | 40.2 | 8.22 | 62 |
| [FA2 + PF + L](results/08_fa2_padding_free_liger_90/summary.json) | 32/2 | 90.1 | 17271 | 12.3 | 7.15 | 257 |
| [FA2 + P + C](results/09_best_compile_90/summary.json) | 32/2 | 90.8 | 30500 | 27.3 | 9.55 | 32 |
| [FA2 + P + C](results/final_1_fa2_packing_compile/summary.json) | 32/2 | 300.7 | 30018 | 27.4 | 5.99 | 267 |
| [FA2 + P](results/final_2_fa2_packing/summary.json) | 32/2 | 301.4 | 22153 | 40.2 | 6.66 | 208 |
| [SDPA](results/retry_batch64_expandable_90/summary.json) | 64/1 | 90.8 | 6450 | 73.5 | 8.19 | 100 |
| [SDPA + L](results/retry_liger_cuda118_90/summary.json) | 32/2 | 90.2 | 7143 | 26.8 | 8.17 | 104 |

### Подбор сочетаний и размера batch

| Вариант | B/GA | Время, с | Токенов/с | Память, GiB | Val loss ↓ | Шаги |
|---|---:|---:|---:|---:|---:|---:|
| [FA2 + P + C](results/s2_00_pack_b16_ga4_compile/summary.json) | 16/4 | 120.1 | 29651 | 17.4 | 8.23 | 61 |
| [FA2 + P + C](results/s2_01_pack_b64_ga1_compile/summary.json) | 64/1 | 120.2 | 30993 | 45.7 | 8.13 | 77 |
| [FA2 + P + C](results/s2_02_pack_b96_ga1_compile/summary.json) | 96/1 | 120.8 | 30605 | 65.4 | 8.50 | 45 |
| [FA2 + P + L](results/s2_03_pack_liger_b32/summary.json) | 32/1 | 120.5 | 25769 | 24.1 | 7.17 | 187 |
| [FA2 + P + L](results/s2_04_pack_liger_b64/summary.json) | 64/1 | 120.6 | 27132 | 41.9 | 7.96 | 100 |
| [FA2 + P + C + L](results/s2_05_pack_liger_b32_compile/summary.json) | 32/1 | 120.1 | 31586 | 25.4 | 10.81 | 115 |
| [FA2 + P + C + L](results/s2_06_pack_liger_b64_compile/summary.json) | 64/1 | 120.6 | 33515 | 41.5 | 10.84 | 78 |
| [FA2 + PF + C](results/s2_07_pf_b128_compile/summary.json) | 128/1 | 120.4 | 29312 | 22.3 | 7.08 | 214 |
| [FA2 + PF + C](results/s2_08_pf_b256_compile/summary.json) | 256/1 | 120.5 | 30593 | 37.1 | 7.91 | 111 |
| [FA2 + P + C](results/s2_09_pack_b8_ga8_compile/summary.json) | 8/8 | 120.7 | 26518 | 12.4 | 8.16 | 72 |
| [FA2 + P + C](results/s2_10_pack_b32_ga1_compile/summary.json) | 32/1 | 120.1 | 30173 | 25.6 | 7.47 | 169 |
| [FA2 + P + C + L](results/s2_final_1_pack_liger_b64_compile/summary.json) | 64/1 | 300.9 | 33559 | 41.5 | 10.81 | 295 |

### Проверка Liger и финальной конфигурации

| Вариант | B/GA | Время, с | Токенов/с | Память, GiB | Val loss ↓ | Шаги |
|---|---:|---:|---:|---:|---:|---:|
| [FA2 + P + C-auto + L-noCE (прогретый кэш)](results/s3_autotune_cached_pack_liger_noce_b64/summary.json) | 64/1 | 300.4 | 32814 | 45.6 | 5.70 | 293 |
| [FA2 + P + C-auto + L-noCE (холодный кэш)](results/s3_autotune_pack_liger_noce_b64/summary.json) | 64/1 | 469.7 | — | 45.7 | 11.23 | 2 |
| [FA2 + P + C-auto + L-noCE (итоговый)](results/s3_final_autotune_pack_liger_noce_b64/summary.json) | 64/1 | 300.4 | 32753 | 45.6 | 5.76 | 288 |
| [FA2 + P + C + L-noCE](results/s3_final_default_pack_liger_noce_b64/summary.json) | 64/1 | 300.3 | 31751 | 45.6 | 5.79 | 284 |
| [FA2 + P + C + L-CE](results/s3_probe_liger_ce_b64/summary.json) | 64/1 | 180.7 | 32057 | 41.5 | 10.82 | 111 |
| [FA2 + P + C + L-noCE](results/s3_probe_liger_noce_b64/summary.json) | 64/1 | 180.4 | 31870 | 45.7 | 7.72 | 132 |

Первый запуск batch 64 завершился нехваткой памяти; повтор с `expandable_segments` прошёл успешно. Первый Liger-запуск потребовал выбора CUDA-совместимого `ptxas`. Ещё один прогон полного Liger с compile остановлен после повторения плато loss. Эти три запуска не имеют итоговых метрик и в таблицы не включены.

## Итоговая конфигурация

Выбраны **FA2 + packing BFD + torch.compile max-autotune + Liger без fused loss**, batch 64, accumulation 1. Получено **32 753 токена/с**, validation loss **5.76**, пиковая память **45.6 GiB**. Ускорение относительно бейзлайна:

**S = 32 753 / 7 002 ≈ 4.68×.**

Код — в [train.py](train.py), результат контрольного запуска — в [summary.json](summary.json). Все исходные метрики сохранены в [results](results).

Финал использует предварительно прогретый кэш компилятора, но новые веса и состояние оптимизатора. Холодный autotune занял около 470 секунд и успел выполнить только два шага — это не финальный пятиминутный замер. Время подготовки кэша не входит в заявленное ускорение. Вариант с обычным compile дал 31 751 токен/с; при единственном запуске в чистом контейнере max-autotune может не уложиться в бюджет.

## Выводы

- Batch 16 с accumulation 4 оказался быстрее бейзлайна: 7 991 против 7 002 токенов/с, при этом память снизилась с 41.2 до 24.2 GiB. Увеличение batch до 64 без других оптимизаций не помогло.
- Checkpointing снизил память до 17.4 GiB, но скорость упала до 5 953 токенов/с. Экономия памяти достигается за счёт повторных вычислений при backward.
- FA2 сам по себе дал 7 620 токенов/с, а вместе с padding-free — 19 291. Удаление вычислений над padding оказалось важнее одной замены attention.
- Packing с compile на 300-секундных запусках дал 30 018 токенов/с против 22 153 без compile. Сочетание оптимизаций ускорило обучение примерно на 36%.
- Полный Liger с compile оказался быстрым, но loss застрял около 10.8. Отключение fused linear cross-entropy восстановило нормальное обучение, поэтому финал использует остальные ядра Liger.
- Max-autotune добавил около 3% к обычному compile. Два полных запуска с готовым кэшем дали 32 814 и 32 753 токена/с; в итог взят контрольный повтор.

Packing увеличивает число полезных токенов на шаг, поэтому шаги и loss коротких запусков нельзя сравнивать без учёта объёма обработанного текста. За одинаковое время быстрые конфигурации успевают обучиться на большем числе токенов.

## Кольцевые коммуникации

В [communications.ipynb](communications.ipynb) реализованы `ring_reduce_scatter`, `ring_all_gather` и `ring_all_reduce`; код также сохранён в [collectives.py](collectives.py). Reduce-scatter накапливает сумму блоков при передаче по кольцу, all-gather собирает блоки на каждом процессе, all-reduce выполняет обе операции подряд.

Проверки на CPU с Gloo сравнивают результат с `torch.distributed` на 2 и 4 процессах и трёх размерах блока. Отдельно проверяется, что входные тензоры не изменяются. Выводы проверок сохранены в ноутбуке.
