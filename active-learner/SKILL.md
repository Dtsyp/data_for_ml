---
name: active-learner
description: Active Learning agent for intelligent sample selection in spectroscopy data
license: MIT
compatibility: opencode
metadata:
  category: active-learning
  version: 1.0.0
---

# Active Learner

Умный отбор данных для разметки с помощью Active Learning. Сравнивает стратегии entropy vs random.

## Workflow

1. ❗ **HITL**: пользователь подтверждает настройки (seed size, iterations, batch size)
2. **Разделение данных**: seed (50) + pool (остаток) + test (30%)
3. **Цикл AL**: 5 итераций по 20 примеров
4. **Сравнение**: entropy vs random на одном графике
5. **Отчёт**: сколько примеров сэкономлено

## Скрипты

### al_agent.py
```bash
.venv/bin/python active-learner/scripts/al_agent.py \
    --input data/labeled/labeled.parquet \
    --output-dir data/active \
    --seed-size 50 \
    --iterations 5 \
    --batch-size 20 \
    --strategy entropy
```

Выход:
- `data/active/history_entropy.json` — история обучения (entropy)
- `data/active/history_random.json` — история обучения (random)
- `data/active/final_model.pkl` — финальная модель

### visualize.py
```bash
.venv/bin/python active-learner/scripts/visualize.py \
    --entropy data/active/history_entropy.json \
    --random data/active/history_random.json \
    --output data/active/learning_curve.png
```

Выход:
- `data/active/learning_curve.png` — кривые обучения
- `data/active/REPORT.md` — текстовый отчёт

## Архитектура AL цикла

```
Seed (50 labeled) → Train model → Predict on pool
                                       ↓
                              Select top-K uncertain
                                       ↓
                              Add to labeled set
                                       ↓
                              Retrain → Evaluate
                                       ↓
                              Repeat 5 times
```

## Стратегии отбора

- **entropy**: H(p) = -Σ(p·log(p)) — максимальная неопределённость
- **margin**: 1 - (p_max - p_second) — минимальная разница между топ-2
- **random**: случайный отбор (baseline)

## Feature extraction для спектров
- PCA (50 компонент) на raw spectrum data
- StandardScaler перед PCA
- Модель: LogisticRegression (default) или SVM

## Правила
- Всегда используй `.venv/bin/python`
- Всегда запускай оба стратегии (entropy + random) для сравнения
- Спрашивай пользователя перед началом цикла
