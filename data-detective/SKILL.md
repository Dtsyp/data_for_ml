---
name: data-detective
description: Detect and fix data quality issues in spectroscopy datasets
license: MIT
compatibility: opencode
metadata:
  category: data-quality
  version: 1.0.0
---

# Data Detective

Автоматически выявляет и устраняет проблемы качества данных в спектральных датасетах.

## Workflow

1. **Загрузить** очищенный датасет из `data/raw/combined.parquet`
2. **Детекция проблем**: missing values, duplicates, outliers, class imbalance
3. **Визуализация** каждой проблемы
4. ❗ **HITL**: показать отчёт пользователю, пользователь выбирает стратегию чистки
5. **Применить** выбранную стратегию
6. **Сравнение** до/после
7. **Сохранить** очищенный датасет

## Типы проблем

1. **Missing values** — NaN в spectrum, пустые label
2. **Duplicates** — идентичные спектры (по корреляции > 0.99)
3. **Outliers** — аномальная длина спектра или интенсивность (IQR метод)
4. **Class imbalance** — перекос распределения классов
5. **Low quality** — спектры с низким SNR (signal-to-noise ratio)

## Стратегии чистки

| Strategy | Missing | Duplicates | Outliers | Low SNR |
|----------|---------|------------|----------|---------|
| aggressive | удалить строки | удалить все | удалить (IQR) | удалить |
| conservative | заполнить средним | оставить | оставить | оставить |
| balanced | удалить строки | удалить дубли | clip (z>3) | удалить |

## Скрипты

### detective.py
```bash
.venv/bin/python data-detective/scripts/detective.py --input data/raw/combined.parquet --output-dir data/detective
```
Выход:
- `problems.json` — обнаруженные проблемы
- `missing_values.png`, `class_balance.png`, `outliers.png`, `duplicates.png`

### cleaner.py
```bash
.venv/bin/python data-detective/scripts/cleaner.py --input data/raw/combined.parquet --output data/cleaned/cleaned.parquet --strategy balanced
```

### compare.py
```bash
.venv/bin/python data-detective/scripts/compare.py --before data/raw/combined.parquet --after data/cleaned/cleaned.parquet --output data/detective/comparison.md
```

## Правила
- Всегда используй `.venv/bin/python`
- Всегда показывай отчёт ПЕРЕД чисткой
- Пользователь выбирает стратегию
