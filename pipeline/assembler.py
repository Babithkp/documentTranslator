"""Assemble translated page images into a single output PDF."""

from PIL import Image


class PDFAssembler:
    def assemble(self, image_paths: list, output_path: str, dpi: int = 150) -> None:
        if not image_paths:
            raise ValueError("No pages to assemble.")

        pages = [Image.open(p).convert("RGB") for p in image_paths]
        first, rest = pages[0], pages[1:]

        first.save(
            output_path,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=dpi,
        )
        print(f"Saved {len(pages)}-page PDF → {output_path}")
