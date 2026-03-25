---
name: spectrum-labeler
description: Auto-label Raman spectra using Mistral LLM and export for manual review
license: MIT
compatibility: opencode
metadata:
  category: annotation
  version: 1.0.0
---

# Spectrum Labeler

Автоматически размечает Raman-спектры с помощью Mistral API. Генерирует спецификацию разметки, оценивает качество и экспортирует задачи для ручной доразметки.

## Workflow

1. ❗ **HITL**: пользователь подтверждает задачу, классы и инструкции для разметки
2. **Авторазметка**: Mistral API классифицирует спектры по характерным пикам
3. **Оценка качества**: confidence scores, распределение меток
4. **Экспорт неуверенных**: примеры с confidence < threshold → `review_queue.csv`
5. ❗ **HITL**: пользователь проверяет и правит метки в `review_queue.csv`
6. **Объединение**: уверенные + исправленные → финальный датасет
7. **Спецификация**: генерация annotation_spec.md
8. **LabelStudio export**: JSON для импорта в LabelStudio

## Скрипты

### auto_labeler.py
```bash
.venv/bin/python spectrum-labeler/scripts/auto_labeler.py \
    --input data/cleaned/cleaned.parquet \
    --output data/labeled/labeled.parquet \
    --classes "polymer,mineral,organic,inorganic" \
    --task "Classify Raman spectrum material type" \
    --confidence-threshold 0.7
```

Выход:
- `data/labeled/labeled.parquet` — размеченный датасет (с confidence)
- `data/labeled/review_queue.csv` — примеры для ручной проверки
- `data/labeled/spec.md` — спецификация разметки
- `data/labeled/quality.json` — метрики качества
- `data/labeled/labelstudio.json` — экспорт для LabelStudio

## Формат review_queue.csv
```csv
index,spectrum_peaks,current_label,confidence,corrected_label
0,"[520, 1580, 2700]",polymer,0.45,
1,"[464, 128]",mineral,0.62,
```
Пользователь заполняет колонку `corrected_label` и сохраняет как `review_queue_corrected.csv`.

## Правила
- Всегда используй `.venv/bin/python`
- Mistral API ключ из `.env` (MISTRAL_API_KEY)
- ВСЕГДА спрашивай подтверждение классов и задачи перед разметкой
- Не размечай больше 500 примеров за раз (лимиты API)
