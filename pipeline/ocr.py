import numpy as np
from PIL import Image, ImageEnhance
from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(
        self,
        lang: str = "en",
        min_confidence: float = 0.25,
        contrast: float = 1.8,
    ):
        self.min_confidence = min_confidence
        # Mild contrast boost applied only for detection: it makes faint /
        # low-ink text (light italics, worn scans) visible to the detector
        # without harming already-dark text.  The rendered output still uses
        # the untouched original image.  1.0 disables it.
        self.contrast = contrast
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract(self, image_path: str) -> list:
        image = Image.open(image_path).convert("RGB")
        if self.contrast and self.contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(self.contrast)
        # PaddleOCR expects BGR; contrast does not change geometry, so the
        # returned bboxes still map onto the original image.
        arr = np.array(image)[:, :, ::-1]
        result = self.ocr.ocr(arr, cls=True)

        if not result or not result[0]:
            return []

        words = []
        for line in result[0]:
            bbox_pts = line[0]
            text = line[1][0]
            conf = float(line[1][1])

            if conf < self.min_confidence:
                continue

            # Drop low-confidence noise dominated by one repeated character
            # (e.g. a faint line the contrast boost turned into "nnn nn nnnn").
            if _looks_like_noise(text, conf):
                continue

            x1 = min(p[0] for p in bbox_pts)
            y1 = min(p[1] for p in bbox_pts)
            x2 = max(p[0] for p in bbox_pts)
            y2 = max(p[1] for p in bbox_pts)

            words.append({
                "text": text,
                "confidence": conf,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
            })

        return words


def _looks_like_noise(text: str, conf: float) -> bool:
    """
    True for low-confidence OCR output that is almost certainly garbage.

    The signal is generic, not tied to any document: a long string whose
    single most-common letter dominates it (real words never repeat one letter
    that heavily) read at low confidence.  Short tokens and confident reads are
    always kept.
    """
    letters = [c.lower() for c in text if c.isalpha()]
    if conf >= 0.7 or len(letters) < 8:
        return False
    top = max((letters.count(c) for c in set(letters)), default=0)
    return top / len(letters) > 0.45
