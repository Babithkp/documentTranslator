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

        # Pass 1: the untouched original.  Already-legible text is read here at
        # full confidence and treated as authoritative.
        base = self._detect(image)

        # Pass 2 (optional): a contrast-boosted copy recovers faint / low-ink
        # text (light italics, worn scans) the detector skips on the original.
        # The boost can, however, blow a normal line into garbage, so its
        # detections are only ADDED where pass 1 found nothing — never allowed
        # to replace a clean read.  This keeps the recall gain without the
        # corruption a single boosted pass caused.
        if self.contrast and self.contrast != 1.0:
            boosted = self._detect(ImageEnhance.Contrast(image).enhance(self.contrast))
            for d in boosted:
                if not any(_boxes_overlap(b, d) for b in base):
                    base.append(d)

        return base

    def _detect(self, image: Image.Image) -> list:
        # PaddleOCR expects BGR; the contrast boost does not change geometry, so
        # the returned bboxes still map onto the original image either way.
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


def _boxes_overlap(a: dict, b: dict, min_ratio: float = 0.3) -> bool:
    """True when box b overlaps box a by more than min_ratio of b's own area.

    Used to tell whether a contrast-boosted detection sits on top of a region
    the original pass already read (keep the original) or in empty space
    (recovered faint text worth adding).
    """
    ix = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    iy = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    if ix <= 0 or iy <= 0:
        return False
    inter = ix * iy
    area_b = max(1.0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / area_b > min_ratio


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
