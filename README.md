# Raman Spectroscopy ML Pipeline

ML-пайплайн для классификации материалов по Raman-спектрам. Реализован как набор Claude Code Skills с 4 агентами и human-in-the-loop.

## 1. Описание задачи и датасета

- **Модальность:** Спектральные данные (Raman-спектроскопия)
- **ML-задача:** Классификация материалов по Raman-спектрам
- **Классы:** polymer, mineral, organic, inorganic
- **Объём:** ~300 спектров
- **Источники:** HuggingFace Hub, Kaggle, DuckDuckGo, Google Scholar + web scraping
- **Схема данных:** `spectrum` (list[float]), `wavenumber` (list[float]), `label` (str), `source` (str), `collected_at` (str)

## 2. Архитектура: 4 агента

```
Spectrum Collector → Data Detective → Spectrum Labeler → Active Learner
   (сбор данных)     (чистка)          (авторазметка)      (AL-отбор)
                                           │
                                      ❗ HITL точка
                                     (ручная правка)
```

| Агент | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| **Spectrum Collector** | Сбор из 4 источников (HF, Kaggle, Web, Scholar) + scraping, унификация, EDA | search, scrape, unify, eda |
| **Data Detective** | Детекция проблем, 3 стратегии чистки | detect_issues, fix, compare |
| **Spectrum Labeler** | Авторазметка через Mistral API, confidence scoring | auto_label, generate_spec, export_labelstudio |
| **Active Learner** | AL-цикл: entropy vs random, learning curves | fit, query, evaluate, report |

## 3. Human-in-the-Loop

**Основная HITL точка:** после авторазметки (Step 3)

1. Агент размечает спектры через Mistral API с confidence score
2. Примеры с `confidence < 0.7` → `data/labeled/review_queue.csv`
3. Человек открывает CSV, проверяет метки, заполняет `corrected_label`
4. Сохраняет как `review_queue_corrected.csv`
5. Пайплайн загружает исправления и объединяет с уверенными метками

**Дополнительные HITL точки:**
- Выбор стратегии чистки (Step 2)
- Подтверждение классов и задачи (Step 3)
- Подтверждение настроек AL (Step 4)

## 4. Запуск

### Установка

```bash
# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Установить зависимости
pip install -r requirements.txt

# Настроить API ключи
cp .env.example .env
# Заполнить MISTRAL_API_KEY в .env
```

### Запуск пайплайна

```bash
# Полный пайплайн
python run_pipeline.py

# С параметрами
python run_pipeline.py --config config.yaml

# Пропустить шаги (если данные уже собраны)
python run_pipeline.py --skip-collection --skip-labeling
```

### Запуск отдельных агентов

```bash
# 1. Поиск датасетов
.venv/bin/python spectrum-collector/scripts/search_datasets.py --query "raman spectroscopy"

# 2. Детекция проблем
.venv/bin/python data-detective/scripts/detective.py --input data/raw/combined.parquet

# 3. Чистка данных
.venv/bin/python data-detective/scripts/cleaner.py --input data/raw/combined.parquet --output data/cleaned/cleaned.parquet --strategy balanced

# 4. Авторазметка
.venv/bin/python spectrum-labeler/scripts/auto_labeler.py --input data/cleaned/cleaned.parquet --output data/labeled/labeled.parquet --classes "polymer,mineral,organic,inorganic"

# 5. Active Learning
.venv/bin/python active-learner/scripts/al_agent.py --input data/labeled/labeled.parquet --output-dir data/active

# 6. Визуализация AL
.venv/bin/python active-learner/scripts/visualize.py --entropy data/active/history_entropy.json --random data/active/history_random.json
```

## 5. Структура проекта

```
├── spectrum-collector/          # Skill 1: сбор данных
│   ├── SKILL.md
│   └── scripts/
│       ├── search_datasets.py
│       ├── unify_schema.py
│       ├── eda_analysis.py
│       └── generate_report.py
├── data-detective/              # Skill 2: качество данных
│   ├── SKILL.md
│   └── scripts/
│       ├── detective.py
│       ├── cleaner.py
│       └── compare.py
├── spectrum-labeler/            # Skill 3: авторазметка
│   ├── SKILL.md
│   └── scripts/
│       └── auto_labeler.py
├── active-learner/              # Skill 4: Active Learning
│   ├── SKILL.md
│   └── scripts/
│       ├── al_agent.py
│       └── visualize.py
├── raman-pipeline/              # Skill 5: оркестрация
│   └── SKILL.md
├── agent/                       # Оригинальный Mistral-агент для поиска
│   ├── main.py
│   ├── loop.py
│   ├── prompts.py
│   ├── ui.py
│   └── tools/
├── run_pipeline.py              # Единая точка входа
├── config.yaml                  # Конфигурация
├── requirements.txt
├── PIPELINE_GUIDE.md            # Универсальный гайд
├── data/
│   ├── raw/                     # Сырые данные
│   ├── cleaned/                 # Очищенные данные
│   ├── labeled/                 # Размеченные данные
│   └── active/                  # Результаты AL
├── models/                      # Обученная модель
└── reports/                     # Отчёты
    ├── quality_report.md
    ├── annotation_report.md
    ├── al_report.md
    └── final_report.md
```

## Data Card

| Поле | Значение |
|------|---------|
| Название | Raman Spectroscopy Materials Dataset |
| Модальность | Спектральные данные (1D signal) |
| Размер | ~300 спектров, 500 точек каждый |
| Классы | polymer, mineral, organic, inorganic |
| Формат | Parquet (pandas DataFrame) |
| Схема | spectrum, wavenumber, label, source, collected_at, confidence |
| Разметка | Автоматическая (Mistral API) + ручная проверка (HITL) |
| Лицензия | MIT |

## Технологии

- **LLM:** Mistral API (авторазметка спектров)
- **ML:** scikit-learn (LogisticRegression, PCA)
- **Оркестрация:** Claude Code Skills
- **Данные:** pandas, parquet
- **Визуализация:** matplotlib
