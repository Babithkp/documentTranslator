import os
import fitz
from PIL import Image


class PageRenderer:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def render(self, input_path: str, output_dir: str = "outputs/pages") -> list:
        os.makedirs(output_dir, exist_ok=True)
        ext = os.path.splitext(input_path)[1].lower()

        if ext == ".pdf":
            return self._render_pdf(input_path, output_dir)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            return self._render_image(input_path, output_dir)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _render_pdf(self, pdf_path: str, output_dir: str) -> list:
        doc = fitz.open(pdf_path)
        pages = []

        for i in range(len(doc)):
            page = doc.load_page(i)
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            path = os.path.join(output_dir, f"page_{i + 1}.png")
            pix.save(path)

            pages.append({
                "page_index": i,
                "image_path": path,
                "width": pix.width,
                "height": pix.height,
                "dpi": self.dpi,
            })

        doc.close()
        return pages

    def _render_image(self, image_path: str, output_dir: str) -> list:
        img = Image.open(image_path).convert("RGB")
        path = os.path.join(output_dir, "page_1.png")
        img.save(path)

        return [{
            "page_index": 0,
            "image_path": path,
            "width": img.width,
            "height": img.height,
            "dpi": self.dpi,
        }]
