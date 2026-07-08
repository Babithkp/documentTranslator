"""
Scanned-PDF / image translator.

Pipeline per page:
  1. Render page → high-res image.
  2. OCR every word.
  3. Group words into lines; split each line into text tokens vs numbers/codes.
  4. Batch-translate all text lines in one API call.
  5. For each translated line: fill ONLY the original text-word bbox, then
     draw the translated text at that exact position.
  6. Numbers, codes, and other non-text regions are never touched.
  7. Assemble all pages into an output PDF.

Usage:
  python main.py sample.pdf
  python main.py scan.jpg  -l German  -o out.pdf
  python main.py doc.pdf   -l Dutch   --dpi 200
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

from pipeline.page_renderer import PageRenderer
from pipeline.ocr import OCRProcessor
from pipeline.layout import build_lines
from pipeline.eraser import BackgroundEraser
from pipeline.translator import Translator
from pipeline.renderer import TextRenderer
from pipeline.assembler import PDFAssembler

load_dotenv()

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def translate_document(
    input_path: str,
    output_path: str,
    target_language: str = "French",
    dpi: int = 150,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set in .env or environment.")

    page_renderer = PageRenderer(dpi=dpi)
    ocr = OCRProcessor()
    eraser = BackgroundEraser()
    translator = Translator(api_key=api_key, target_language=target_language)
    text_renderer = TextRenderer(FONT_REGULAR, FONT_BOLD)
    assembler = PDFAssembler()

    print(f"Rendering: {input_path}")
    pages = page_renderer.render(input_path)
    print(f"  {len(pages)} page(s) at {dpi} DPI")

    output_image_paths = []

    for page in pages:
        idx = page["page_index"]
        label = f"Page {idx + 1}/{len(pages)}"

        print(f"\n{label}: OCR …")
        words = ocr.extract(page["image_path"])
        print(f"{label}: {len(words)} words detected")

        if not words:
            output_image_paths.append(page["image_path"])
            continue

        # Line-level split: each entry has text-only tokens + their bbox
        lines = build_lines(words)
        print(f"{label}: {len(lines)} text lines (numbers/codes excluded)")

        if not lines:
            output_image_paths.append(page["image_path"])
            continue

        # Batch translate all lines in one call (each line keeps its own id)
        payload = [{"id": i, "text": line["text"]} for i, line in enumerate(lines)]

        print(f"{label}: translating {len(payload)} lines → {target_language} …")
        translated = translator.translate_blocks(payload)
        id_map = {t["id"]: t["translated"] for t in translated}

        for i, line in enumerate(lines):
            line["translated"] = id_map.get(i, "")

        # Load original image and sample background once from the page margins
        image = Image.open(page["image_path"]).convert("RGB")
        bg_color = eraser.sample_background(image)
        print(f"{label}: background colour → RGB{bg_color}")

        # Overlay translated text at exact original positions
        image = text_renderer.overlay_lines(image, lines, bg_color)

        out_path = page["image_path"].replace(".png", "_translated.png")
        image.save(out_path)
        output_image_paths.append(out_path)
        print(f"{label}: saved → {out_path}")

    print(f"\nAssembling {len(output_image_paths)} page(s) …")
    assembler.assemble(output_image_paths, output_path, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate text in scanned PDFs and images."
    )
    parser.add_argument("input", help="PDF, JPG, PNG, TIFF …")
    parser.add_argument(
        "-o", "--output", default="translated_output.pdf",
        help="Output PDF (default: translated_output.pdf)",
    )
    parser.add_argument(
        "-l", "--language", default="French",
        help="Target language (default: French)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Render DPI (default: 150; try 200-300 for poor scans)",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"File not found: {args.input}")

    translate_document(
        input_path=args.input,
        output_path=args.output,
        target_language=args.language,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

