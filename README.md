# Karakalpak NER — Rule-based Numerical Entity Extraction

A Django web application that automatically extracts and normalizes
numerical expressions from Karakalpak-language text documents.

## Features

- Extracts 8 entity types: MON, PCT, DAT, TIM, CNT, FRC, ORD, APX
- Upload `.txt` or `.docx` files
- Inline text demo (no file needed)
- Download results as JSON or Excel
- Analysis history with delete support
- Django admin panel
- CLI test command
- ~30 MB total — fits in 100 MB server

## Entity types

| Tag | Meaning         | Example input           | Output          |
|-----|-----------------|-------------------------|-----------------|
| MON | Money           | bes jüz müng so'm       | 500,000 UZS     |
| PCT | Percent         | jigirma bes protsent    | 25.0%           |
| DAT | Date            | 2024-jılı 15-may        | 15.05.2024      |
| TIM | Time            | saat on eki             | 12:00           |
| CNT | Count           | dört müng gektar        | 4,000 gektar    |
| FRC | Fraction        | bir butın bes           | 1.5             |
| ORD | Ordinal         | üshinshi                | 3-o'rinchi      |
| APX | Approximate     | bir neshe               | ~3–5            |

## Setup

### 1. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Apply migrations

```bash
python manage.py migrate
```

### 4. Create admin user (optional)

```bash
python manage.py createsuperuser
```

### 5. Run the server

```bash
python manage.py runserver
```

Open your browser at: **http://127.0.0.1:8000**

Admin panel: **http://127.0.0.1:8000/admin**

---

## Test the NER engine from terminal

```bash
# Run with built-in sample text
python manage.py test_ner

# Analyze your own text
python manage.py test_ner --text "Kompaniya bes jüz müng so'm daromad aldi."

# Analyze a file
python manage.py test_ner --file mydocument.txt

# Filter by entity type
python manage.py test_ner --types MON,PCT,DAT
```

---

## API endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Home / upload page |
| POST | `/analyze/` | Upload and analyze file |
| GET | `/result/<id>/` | View results |
| GET | `/result/<id>/?type=MON` | Filter results by type |
| GET | `/history/` | Analysis history |
| POST | `/delete/<id>/` | Delete an analysis |
| GET | `/download/json/<id>/` | Download JSON |
| GET | `/download/excel/<id>/` | Download Excel |
| GET | `/api/stats/` | JSON stats |
| POST | `/api/analyze/` | Analyze raw text (JSON body) |

### POST /api/analyze/ example

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -H "X-CSRFToken: <token>" \
  -d '{"text": "Kompaniya bes jüz müng so'\''m daromad aldi."}'
```

Response:
```json
{
  "token_count": 6,
  "count": 1,
  "items": [
    {
      "type": "MON",
      "raw": "bes jüz müng so'm",
      "formatted": "500,000 UZS",
      "value": 500000.0,
      "unit": "UZS"
    }
  ]
}
```

---

## Project structure

```
karakalpak_ner/
├── manage.py
├── requirements.txt
├── README.md
│
├── karakalpak_ner/          # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── ml/                      # NER logic (no PyTorch, pure Python)
│   ├── rule_ner.py          # Patterns, extraction, normalization
│   └── output_writer.py     # JSON + Excel export
│
├── ner/                     # Django app
│   ├── models.py            # Analysis + Result ORM models
│   ├── views.py             # All views
│   ├── urls.py              # URL routing
│   ├── forms.py             # Upload form
│   ├── admin.py             # Admin panel config
│   ├── templates/ner/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── result.html
│   │   └── history.html
│   └── management/commands/
│       └── test_ner.py      # CLI test command
│
└── media/                   # Created automatically
    ├── uploads/             # Uploaded files
    └── outputs/             # Generated JSON/Excel
```

---

## Deployment (free tier)

### Railway / Render
1. Push to GitHub
2. Connect repo on Railway or Render
3. Set environment variable: `DJANGO_SETTINGS_MODULE=karakalpak_ner.settings`
4. Build command: `pip install -r requirements.txt && python manage.py migrate`
5. Start command: `python manage.py runserver 0.0.0.0:$PORT`

Total disk usage: ~30–40 MB (well within 100 MB limit).

---

## Storage usage

| Component | Size |
|-----------|------|
| Django | ~10 MB |
| openpyxl | ~4 MB |
| python-docx | ~3 MB |
| Your NER code | ~0.1 MB |
| SQLite DB | ~1 MB |
| **Total** | **~18–25 MB** |
