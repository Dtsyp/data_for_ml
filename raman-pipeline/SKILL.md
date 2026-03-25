---
name: raman-pipeline
description: End-to-end orchestration of all 4 agents for Raman spectroscopy ML pipeline
license: MIT
compatibility: opencode
metadata:
  category: orchestration
  version: 1.0.0
---

# Raman Pipeline

Оркестрирует все 4 агента в единый end-to-end пайплайн для Raman-спектроскопии.

## Запуск

```bash
.venv/bin/python run_pipeline.py
```

Или с параметрами:
```bash
.venv/bin/python run_pipeline.py \
    --classes "polymer,mineral,organic" \
    --strategy balanced \
    --seed 50 \
    --iterations 5 \
    --batch 20 \
    --confidence-threshold 0.7
```

## Шаги пайплайна

### Шаг 1: Сбор данных (Spectrum Collector)
- Собирает данные из HuggingFace/Kaggle + web scraping
- Выход: `data/raw/combined.parquet`

### Шаг 2: Чистка данных (Data Detective)
- Детектирует проблемы
- ❗ HITL: пользователь выбирает стратегию
- Выход: `data/cleaned/cleaned.parquet`

### Шаг 3: Авторазметка (Spectrum Labeler)
- Mistral API размечает спектры
- ❗ HITL: пользователь правит неуверенные метки
- Выход: `data/labeled/labeled.parquet`

### Шаг 4: Active Learning
- Цикл AL: entropy vs random
- Выход: `data/active/learning_curve.png`

### Шаг 5: Обучение модели
- Финальная модель на полном размеченном датасете
- Выход: `models/final_model.pkl`, `models/metrics.json`

### Шаг 6: Генерация отчёта
- Объединяет все метрики в финальный отчёт
- Выход: `reports/final_report.md`

## HITL точки

| # | Момент | Что проверяет человек | Файл |
|---|--------|----------------------|------|
| 1 | После сбора | Какие источники использовать | (интерактивный выбор) |
| 2 | После детекции | Какую стратегию чистки | (интерактивный выбор) |
| 3 | После разметки | Правка неуверенных меток | review_queue.csv |
| 4 | Перед AL | Настройки цикла | (интерактивный выбор) |

## Правила
- Всегда используй `.venv/bin/python`
- HITL точки обязательны — не пропускай
- Сохраняй промежуточные результаты в parquet
- Генерируй отчёты после каждого шага
