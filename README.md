# Universal ML Data Pipeline

Универсальный ML-пайплайн для подготовки данных. Работает с любой темой — агент сам спрашивает задачу, классы и поисковый запрос. Запускается одной командой.

## 1. Описание задачи и датасета

- **Модальность:** Текст
- **ML-задача:** Задаётся интерактивно (например: sentiment classification, topic detection)
- **Классы:** Задаются интерактивно (например: positive, negative, neutral)
- **Объём:** ~300 примеров (демо-режим) или реальные датасеты
- **Источники:** HuggingFace Hub, Kaggle, DuckDuckGo, Google Scholar + web scraping
- **Схема данных:** `text` (str), `label` (str), `source` (str), `collected_at` (str)

## 2. Архитектура: 4 агента

```
DataCollectionAgent → DataQualityAgent → AnnotationAgent → ActiveLearningAgent
    (сбор данных)       (чистка)           (авторазметка)      (AL-отбор)
                                               │
                                          ❗ HITL точка
                                         (ручная правка)
```

| Агент | Файл | Методы |
|-------|------|--------|
| `DataCollectionAgent` | `agents/data_collection_agent.py` | `scrape()`, `fetch_api()`, `load_dataset()`, `merge()`, `run()` |
| `DataQualityAgent` | `agents/data_quality_agent.py` | `detect_issues()`, `fix()`, `compare()` |
| `AnnotationAgent` | `agents/annotation_agent.py` | `auto_label()`, `generate_spec()`, `check_quality()`, `export_to_labelstudio()` |
| `ActiveLearningAgent` | `agents/al_agent.py` | `fit()`, `query()`, `evaluate()`, `report()`, `run_cycle()` |

### Технический контракт

```python
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.al_agent import ActiveLearningAgent

# Сбор данных
collector = DataCollectionAgent(config='config.yaml')
df = collector.run(sources=[{'type': 'hf_dataset', 'name': 'imdb'}])

# Чистка
quality = DataQualityAgent()
issues = quality.detect_issues(df)
df_clean = quality.fix(df, strategy={'missing': 'drop', 'duplicates': 'drop', 'outliers': 'clip_iqr'})

# Авторазметка
annotator = AnnotationAgent(modality='text')
df_labeled = annotator.auto_label(df_clean)
metrics = annotator.check_quality(df_labeled)

# Active Learning
learner = ActiveLearningAgent(model='logreg')
history = learner.run_cycle(df_labeled, df_labeled, strategy='entropy', n_iterations=5, batch_size=20)
```

## 3. Human-in-the-Loop

**Основная HITL точка:** после авторазметки (Step 3)

1. Агент размечает данные с confidence score
2. Примеры с `confidence < 0.7` → `data/labeled/review_queue.csv`
3. Человек открывает CSV, проверяет метки, заполняет `corrected_label`
4. Сохраняет как `review_queue_corrected.csv`
5. Пайплайн загружает исправления и объединяет с уверенными метками

**Дополнительные HITL точки:**
- Выбор стратегии чистки (Step 2)
- Подтверждение классов и задачи (Step 3)
- Подтверждение настроек AL (Step 4)

## 4. Запуск

```bash
python run_pipeline.py
```

Агент автоматически:
1. Создаёт виртуальное окружение (`.venv`)
2. Устанавливает все зависимости из `requirements.txt`
3. Проверяет `.env` (копирует из `.env.example` если нет)
4. Создаёт директории (`data/`, `models/`, `reports/`)
5. Спрашивает задачу, классы, поисковый запрос
6. Ищет датасеты по 4 источникам
7. Запускает полный пайплайн с HITL-точками

**Единственное требование:** заполнить `MISTRAL_API_KEY` в файле `.env` (для авторазметки через Mistral API).

## 5. Структура проекта

```
agents/                              # 4 агента (задания 1-4)
├── data_collection_agent.py         # Задание 1: сбор данных
├── data_quality_agent.py            # Задание 2: чистка данных
├── annotation_agent.py              # Задание 3: авторазметка
└── al_agent.py                      # Задание 4: Active Learning

run_pipeline.py                      # Единая точка входа (оркестрация)
config.yaml                          # Конфигурация параметров
requirements.txt                     # Зависимости
notebooks/eda.ipynb                  # EDA ноутбук

data/raw/                            # Сырые данные
data/cleaned/                        # Очищенные данные
data/labeled/                        # Размеченные данные + review_queue.csv
data/active/                         # Результаты AL + learning_curve.png
models/                              # Обученная модель + metrics.json
reports/                             # quality_report.md, annotation_report.md, al_report.md, final_report.md
```

## Data Card

| Поле | Значение |
|------|---------|
| Модальность | Текст |
| Формат | Parquet (pandas DataFrame) |
| Схема | text, label, source, collected_at, confidence |
| Разметка | Автоматическая (Mistral API) + ручная проверка (HITL) |
| Лицензия | MIT |
