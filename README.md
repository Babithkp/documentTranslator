# Document Translator

Translates text in scanned PDFs and image-based documents. Extracts text via OCR, translates it using OpenAI, erases the original text, and redraws the translation at the exact same position — producing a translated PDF that preserves the original layout.

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

### Python dependencies

```
pymupdf
paddleocr
paddlepaddle
openai
pillow
numpy
python-dotenv
```

Install with:

```bash
pip install pymupdf paddleocr paddlepaddle openai pillow numpy python-dotenv
```

### System font

The renderer uses DejaVuSans (supports all Latin-extended characters for French, German, Dutch, etc.). On Debian/Ubuntu:

```bash
apt-get install fonts-dejavu-core
```

## Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

## Usage

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

### CLI options

| Option | Default | Description |
|---|---|---|
| `input` | *(required)* | PDF, JPG, PNG, TIFF, BMP, or WebP file |
| `-o`, `--output` | `translated_output.pdf` | Output PDF path |
| `-l`, `--language` | `French` | Target language (any language supported by GPT) |
| `--dpi` | `150` | Render resolution; use 200–300 for blurry or low-quality scans |

## Project structure

```
documenttrans/
├── main.py                     # CLI entry point
├── pipeline/
│   ├── page_renderer.py        # PDF/image → high-res PIL pages
│   ├── ocr.py                  # PaddleOCR wrapper
│   ├── layout.py               # words → lines, text/number classification
│   ├── eraser.py               # background sampling + bbox fill
│   ├── translator.py           # OpenAI batch translation
│   ├── renderer.py             # font sizing, text colour, bold detection, draw
│   └── assembler.py            # PIL pages → multi-page PDF
└── outputs/                    # intermediate page PNGs (auto-created)
```

## How text is placed

1. PaddleOCR returns a bounding box for each detected word.
2. Words on the same horizontal line are grouped; words separated by a large horizontal gap are split into separate columns.
3. Pure numbers, codes (no vowels, all-caps), and grade letters (`A`, `B+`, etc.) are excluded from translation — their pixels are never touched.
4. The remaining text words are sent to OpenAI as a JSON batch. Each line gets a numeric ID so responses can be matched back even if the model reorders them.
5. For each translated line, the background colour is sampled from the strips of pixels immediately above and below the original text — giving an accurate local colour even on tinted or shadowed pages.
6. The original text bbox is filled with that colour, then the translated text is drawn at the same position using a font size derived from the original character height.

## Notes

- Intermediate PNG files are written to `outputs/pages/` and are not deleted automatically.
- If the OpenAI response cannot be parsed as JSON, the original text is kept for that line (no silent data loss).
- Text that sits inside a stamp, seal, or logo is detected by checking whether surrounding pixels are dark, and is skipped entirely.
