from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(self, lang: str = "en", min_confidence: float = 0.4):
        self.min_confidence = min_confidence
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract(self, image_path: str) -> list:
        result = self.ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return []

        words = []
        for line in result[0]:
            bbox_pts = line[0]
            text = line[1][0]
            conf = float(line[1][1])

            if conf < self.min_confidence:
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
