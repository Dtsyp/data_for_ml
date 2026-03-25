---
name: spectrum-collector
description: Collect Raman spectroscopy datasets from multiple sources and unify schema
license: MIT
compatibility: opencode
metadata:
  category: data-engineering
  version: 1.0.0
---

# Spectrum Collector

Собирает данные Raman-спектроскопии из нескольких источников и объединяет в единый датасет.

## Workflow

1. **Поиск датасетов** из 4 источников:
   - HuggingFace Hub — open datasets
   - Kaggle — community datasets
   - DuckDuckGo — web search для нишевых репозиториев
   - Google Scholar — академические статьи с датасетами
2. **Web scraping**: извлечение данных с найденных страниц (RRUFF и др.)
3. **Показать таблицу** найденных датасетов пользователю
4. ❗ **HITL**: пользователь выбирает какие датасеты скачать
5. **Скачать** выбранные датасеты
6. **Унификация схемы**: привести все к единому формату
7. **EDA**: анализ и визуализации
8. **Отчёт**: сгенерировать markdown отчёт

## Унифицированная схема

Все данные приводятся к формату:
- `spectrum` — массив значений интенсивности (list[float])
- `wavenumber` — массив волновых чисел (list[float]) или диапазон
- `label` — класс материала (str)
- `source` — откуда получены данные (str, например "hf:raman_dataset" или "scrape:rruff.info")
- `collected_at` — ISO timestamp сбора (str)

## Скрипты

### search_datasets.py
```bash
# Поиск по всем источникам
.venv/bin/python spectrum-collector/scripts/search_datasets.py --query "raman spectroscopy" --limit 10

# Поиск по конкретным источникам
.venv/bin/python spectrum-collector/scripts/search_datasets.py --query "raman spectroscopy" --sources hf,kaggle

# Доступные источники: hf, kaggle, web, scholar, all (по умолчанию)
```
Ищет датасеты на HuggingFace Hub, Kaggle, DuckDuckGo и Google Scholar. Выводит таблицу: source, name, downloads.
Также содержит функцию `scrape_url()` для извлечения данных с найденных веб-страниц.

### unify_schema.py
```bash
.venv/bin/python spectrum-collector/scripts/unify_schema.py --input data/raw/source1.csv --output data/raw/unified_source1.parquet --spectrum-col intensity --wavenumber-col wavenumber --label-col material --source-name "kaggle:raman_minerals"
```
Приводит датасет к единой схеме.

### eda_analysis.py
```bash
.venv/bin/python spectrum-collector/scripts/eda_analysis.py --input data/raw/combined.parquet --output-dir data/eda
```
Генерирует:
- class_distribution.png — распределение классов
- spectrum_examples.png — примеры спектров по классам
- spectrum_stats.png — статистика длин спектров
- eda_results.json — метрики

### generate_report.py
```bash
.venv/bin/python spectrum-collector/scripts/generate_report.py --input data/raw/combined.parquet --eda-dir data/eda --output data/eda/REPORT.md
```

## Правила
- Всегда используй `.venv/bin/python`
- Все выходные файлы — в текущей рабочей директории (data/)
- Формат обмена — parquet
- ВСЕГДА спрашивай пользователя перед скачиванием
