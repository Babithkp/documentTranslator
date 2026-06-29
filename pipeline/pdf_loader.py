import fitz
import os


class PDFLoader:
    def __init__(self, dpi=120):
        self.dpi = dpi

    def render_pages(self, pdf_path, output_dir="outputs/pages"):
        os.makedirs(output_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        pages = []

        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=self.dpi)

            path = f"{output_dir}/page_{i+1}.png"
            pix.save(path)

            pages.append({
                "page_index": i,
                "image_path": path,
                "pdf_width": page.rect.width,
                "pdf_height": page.rect.height,
                "img_width": pix.width,
                "img_height": pix.height,
                "scale_x": page.rect.width / pix.width,
                "scale_y": page.rect.height / pix.height
            })

        doc.close()
        return pages