# Inkshift

Translates text in scanned PDFs and image-based documents. Extracts text via OCR, translates it using OpenAI, erases the original text, and redraws the translation at the exact same position — producing a translated PDF that preserves the original layout.

Available as a CLI (`main.py`) and as a web app (`app.py`) with drag-and-drop upload and live progress.

## Features

- Supports scanned PDFs, JPG, PNG, TIFF, BMP, and WebP inputs
- OCR via PaddleOCR (handles skewed/rotated text)
- Translation via OpenAI `gpt-4.1-mini` — any target language
- Smart text detection: numbers, codes, grades, and abbreviations are left untouched
- Multi-column layout detection (splits lines at column gaps)
- Background sampled from page margins — clean erase without blurred ghosts
- Font size derived directly from OCR bounding box height (no guessing)
- Bold detection via ink density
- Graphic/stamp detection: text inside seals or logos is skipped to avoid destroying them
- Outputs a multi-page PDF at the original DPI
- Web UI: password-protected upload, live SSE progress, download translated PDF
- Cross-platform font auto-detection (Linux, macOS, Windows)

## Pipeline

```
Input (PDF / image)
       │
       ▼
 PageRenderer        Renders each page to a high-res PNG (PyMuPDF / Pillow)
       │
       ▼
 OCRProcessor        Extracts words with bounding boxes (PaddleOCR)
       │
       ▼
 build_lines()       Groups words into lines, splits text vs numeric tokens
       │
       ▼
 Translator          Batch-translates all text lines in one API call (OpenAI)
       │
       ▼
 BackgroundEraser    Samples page background from margins
       │
       ▼
 TextRenderer        Erases original text, draws translation at original position
       │
       ▼
 PDFAssembler        Combines translated page images into a single output PDF
```

## Requirements

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Install

```bash
pip install -r requirements.txt
```

### System font

The renderer looks for DejaVu Sans (Linux) or Arial (macOS/Windows) automatically, and falls back to `FONT_REGULAR` / `FONT_BOLD` in `.env` if neither is found. On Debian/Ubuntu:

```bash
apt-get install fonts-dejavu-core
```

## Setup

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-proj-...
APP_PASSWORD=your_password_here       # web app login password
FLASK_SECRET_KEY=any_long_random_string

# Optional: override font paths if auto-detection fails
# FONT_REGULAR=/path/to/font.ttf
# FONT_BOLD=/path/to/font-bold.ttf
```

## Usage

### CLI

```bash
# Translate to French (default)
python main.py sample.pdf

# Translate to German, custom output path
python main.py scan.pdf -l German -o translated_german.pdf

# Translate a JPEG image to Dutch
python main.py scan.jpg -l Dutch -o out.pdf

# Higher DPI for poor-quality scans
python main.py doc.pdf -l French --dpi 300
```

#### CLI options

| Option | Default | Description |
|---|---|---|
| `input` | *(required)* | PDF, JPG, PNG, TIFF, BMP, or WebP file |
| `-o`, `--output` | `translated_output.pdf` | Output PDF path |
| `-l`, `--language` | `French` | Target language (any language supported by GPT) |
| `--dpi` | `150` | Render resolution; use 200–300 for blurry or low-quality scans |

### Web app

```bash
python app.py
```

Serves the Inkshift landing page at `/`, the upload UI at `/app` (behind the `APP_PASSWORD` login), and a live SSE progress stream during translation. Uploads and outputs from the previous job are cleaned up automatically each time a new translation starts.

For production, run behind gunicorn:

```bash
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

(Use a single worker unless job state is moved out of in-process memory — `job_queues` / `job_outputs` are not shared across processes.)

## Project structure

```
documenttrans/
├── main.py                     # CLI entry point
├── app.py                      # Flask web app (upload, progress, download)
├── requirements.txt
├── .env.example
├── templates/                  # landing.html, login.html, index.html
├── static/                     # app.js, style.css, landing.css, favicon.svg
├── pipeline/
│   ├── page_renderer.py        # PDF/image → high-res PIL pages
│   ├── ocr.py                  # PaddleOCR wrapper
│   ├── layout.py               # words → lines, text/number classification
│   ├── eraser.py               # background sampling + bbox fill
│   ├── translator.py           # OpenAI batch translation
│   ├── renderer.py             # font sizing, text colour, bold detection, draw
│   └── assembler.py            # PIL pages → multi-page PDF
├── uploads/                    # web app: per-job input + intermediate pages (auto-cleaned)
└── outputs/                    # intermediate page PNGs / web app output PDFs (auto-created)
```

## How text is placed

1. PaddleOCR returns a bounding box for each detected word.
2. Words on the same horizontal line are grouped; words separated by a large horizontal gap are split into separate columns.
3. Pure numbers, codes (no vowels, all-caps), and grade letters (`A`, `B+`, etc.) are excluded from translation — their pixels are never touched.
4. The remaining text words are sent to OpenAI as a JSON batch. Each line gets a numeric ID so responses can be matched back even if the model reorders them.
5. For each translated line, the background colour is sampled from the strips of pixels immediately above and below the original text — giving an accurate local colour even on tinted or shadowed pages.
6. The original text bbox is filled with that colour, then the translated text is drawn at the same position using a font size derived from the original character height.

## Notes

- Intermediate PNG files are written to `outputs/pages/` (CLI) or `uploads/<job_id>/pages/` (web app) and are not deleted automatically for the CLI; the web app clears the previous job's uploads/outputs each time a new translation starts.
- If the OpenAI response cannot be parsed as JSON, the original text is kept for that line (no silent data loss).
- Text that sits inside a stamp, seal, or logo is detected by checking whether surrounding pixels are dark, and is skipped entirely.
