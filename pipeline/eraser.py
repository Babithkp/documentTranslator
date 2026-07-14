"""
Erase text regions by sampling the page background colour and flood-filling.

Replaces: background_sampler + the cv2.inpaint approach in image_writer.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw


class BackgroundEraser:
    def __init__(self, padding: int = 3):
        self.padding = padding

    def sample_background(self, image: Image.Image) -> tuple:
        """Sample background colour from the four page margins."""
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]
        m = max(10, int(min(h, w) * 0.04))

        strips = np.concatenate([
            arr[:m, :].reshape(-1, 3),
            arr[-m:, :].reshape(-1, 3),
            arr[:, :m].reshape(-1, 3),
            arr[:, -m:].reshape(-1, 3),
        ])

        # Keep only bright pixels — these are paper, not text or shadows
        brightness = strips.mean(axis=1)
        bright = strips[brightness > 160]

        if len(bright) == 0:
            return (245, 242, 230)

        median = np.median(bright, axis=0).astype(int)
        return (int(median[0]), int(median[1]), int(median[2]))

    def erase(self, image: Image.Image, bboxes: list, bg_color: tuple = None) -> Image.Image:
        """Fill every bbox region with the background colour."""
        if bg_color is None:
            bg_color = self.sample_background(image)

        result = image.copy().convert("RGB")
        draw = ImageDraw.Draw(result)
        p = self.padding

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            draw.rectangle(
                [
                    max(0, x1 - p),
                    max(0, y1 - p),
                    min(result.width - 1, x2 + p),
                    min(result.height - 1, y2 + p),
                ],
                fill=bg_color,
            )

        return result

    def find_missed_text_boxes(
        self,
        image: Image.Image,
        ocr_boxes: list,
        median_line_h: float,
    ) -> list:
        """
        Find dark text-shaped ink that OCR did not detect.

        OCR misses faint or low-contrast lines; those leftovers would otherwise
        stay in the output as untranslated English.  We build an ink mask of the
        whole page, blank out everything OCR already found (so numbers and codes
        it detected are preserved), then return the bounding boxes of the
        remaining text-shaped blobs.

        Table gridlines, underlines, and box borders are deliberately preserved:
        they are too thin to be text (caught by the min-height/width guard) or
        hollow outlines (caught by the fill-ratio guard).  Dense photographic or
        logo regions are preserved by the same fill-ratio and size guards.
        """
        if not median_line_h or median_line_h <= 0:
            return []

        gray = np.array(image.convert("L"))
        h, w = gray.shape

        min_h = max(6.0, median_line_h * 0.5)
        max_h = median_line_h * 2.5

        # Ink = dark strokes on lighter paper.  Adaptive threshold copes with
        # uneven scan lighting far better than a global cutoff.
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
        )

        # Erase everything OCR already located so only *missed* ink survives.
        pad = int(median_line_h * 0.3)
        for x1, y1, x2, y2 in ocr_boxes:
            ax1 = max(0, int(x1) - pad)
            ay1 = max(0, int(y1) - pad)
            ax2 = min(w, int(x2) + pad)
            ay2 = min(h, int(y2) + pad)
            if ax2 > ax1 and ay2 > ay1:
                ink[ay1:ay2, ax1:ax2] = 0

        # Merge neighbouring glyphs into word/line blobs with a wide, short kernel.
        k = max(8, int(median_line_h))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        dilated = cv2.dilate(ink, kernel, iterations=1)

        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(dilated, 8)

        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
                stats[i, cv2.CC_STAT_AREA],
            )
            # Thin rules/underlines/vertical borders — preserve.
            if bh < min_h or bw < min_h:
                continue
            # Too tall for a text line — a figure, photo, or heading block.
            if bh > max_h:
                continue
            # Spans almost the whole page width — a full rule or banner.
            if bw > w * 0.9:
                continue
            # Hollow outlines (box borders) have very low fill after dilation;
            # real text blobs fill most of their bounding box once merged.
            if area / float(bw * bh) < 0.15:
                continue

            # Ink-density guard on the original pixels (not the dilated mask).
            # Printed text is dark and dense; faint watermarks carry no real ink
            # and sparse handwritten signatures fall below the text density band.
            # This is the main defence against erasing signatures/watermarks that
            # OCR also could not read.
            patch = gray[y:y + bh, x:x + bw]
            if patch.size == 0:
                continue
            if int(patch.min()) > 140:          # no genuinely dark ink → watermark
                continue
            if (patch < 120).mean() < 0.12:     # too sparse → signature / stray marks
                continue

            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])

        return boxes
