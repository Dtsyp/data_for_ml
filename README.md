# Raman/SERS Dataset Search Agent

AI-агент для автоматического поиска и сбора датасетов по Raman/SERS спектроскопии для машинного обучения.

## Что делает агент

1. **Ищет датасеты** на Kaggle, HuggingFace, Google и DuckDuckGo по заданному запросу
2. **Показывает результаты** в виде таблицы с описаниями
3. **Human-in-the-loop** — вы выбираете какие датасеты скачать
4. **Скачивает данные** из выбранных источников (Kaggle API, HuggingFace Hub, прямые URL, веб-скрапинг)

## Установка

```bash
pip install -r requirements.txt
```

### Переменные окружения

Скопируйте `.env.example` в `.env` и заполните:

```bash
cp .env.example .env
```

| Переменная | Обязательна | Описание |
|------------|-------------|----------|
| `MISTRAL_API_KEY` | Да | API ключ Mistral (бесплатно: https://console.mistral.ai/) |
| `KAGGLE_USERNAME` | Нет | Логин Kaggle (для поиска на Kaggle) |
| `KAGGLE_KEY` | Нет | API ключ Kaggle |
| `HF_TOKEN` | Нет | Токен HuggingFace (для приватных датасетов) |

## Использование

### Как standalone агент

```bash
# Поиск по умолчанию (Raman/SERS)
python -m agent

# Поиск по конкретному запросу
python -m agent "SERS nanoparticles bacteria detection dataset"

# С указанием директории для скачивания
python -m agent "Raman mineral classification" --download-dir ./my_data
```

### Как Claude Code skill

В Claude Code выполните:
```
/find-datasets SERS bacteria detection
```

## Архитектура

```
User Query
    │
    ▼
Mistral LLM (system prompt + tools)
    │
    ├─► search_kaggle()
    ├─► search_huggingface()
    ├─► search_web()         (DuckDuckGo)
    └─► search_google()      (Google scraping)
        │
        ▼ (результаты)
    ┌───────────────┐
    │ scrape_url()  │ ← исследование найденных страниц
    └───────────────┘
        │
        ▼
    present_datasets() → Таблица в терминале
        │
        ▼
    User выбирает (1, 3, 5)
        │
        ▼
    download_dataset() → ./downloads/
```

## Инструменты агента

| Tool | Описание |
|------|----------|
| `search_kaggle` | Поиск по Kaggle Datasets API |
| `search_huggingface` | Поиск по HuggingFace Hub API |
| `search_web` | Поиск через DuckDuckGo |
| `search_google` | Поиск через Google (веб-скрапинг) |
| `scrape_url` | Парсинг веб-страниц (BeautifulSoup) |
| `present_datasets` | Показ результатов + выбор пользователя (human-in-the-loop) |
| `download_dataset` | Скачивание (Kaggle/HuggingFace/URL) |

## Структура проекта

```
├── agent/
│   ├── main.py            # Точка входа CLI
│   ├── loop.py            # Агентский цикл (Mistral API + tool_use)
│   ├── prompts.py         # Системный промпт
│   ├── ui.py              # Rich UI для терминала
│   └── tools/
│       ├── kaggle_search.py
│       ├── huggingface_search.py
│       ├── web_search.py
│       ├── google_search.py
│       ├── web_scrape.py
│       └── present_datasets.py
├── .claude/commands/
│   └── find-datasets.md   # Claude Code slash command (skill)
├── .env.example            # Шаблон переменных окружения
├── downloads/              # Скачанные датасеты (gitignored)
└── requirements.txt
```
