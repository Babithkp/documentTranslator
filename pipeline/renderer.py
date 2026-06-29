"""
Overlay translated text onto the original scanned image.

Per-line:
  1. Sample text colour from the darkest pixels in the original bbox.
  2. Detect bold by ink density (dark-pixel ratio in bbox).
  3. Normalise char heights so table rows share a consistent font size.
  4. Solid-fill the original text region with the sampled page background
     colour — invisible on white/near-white pages, no blurred ghost.
  5. Wrap translated text to fit within the original bbox width, reducing
     font size only when wrapped text is too tall for the bbox.
"""

import statistics
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_MIMG = Image.new("RGB", (1, 1))
_MDRAW = ImageDraw.Draw(_MIMG)

# Ink-density threshold above which we treat text as bold.
_BOLD_THRESHOLD = 0.25


class TextRenderer:
    def __init__(self, font_regular: str, font_bold: str):
        self.font_regular = font_regular
        self.font_bold = font_bold

    def overlay_lines(
        self,
        image: Image.Image,
        lines: list,
        bg_color: tuple,
    ) -> Image.Image:
        """
        Draw each translated line over the original image in place.

        `lines` items must have:
          - translated   : translated string
          - text_bbox    : [x1, y1, x2, y2] of the original text words
          - char_height  : average pixel height of original text words
        """
        orig_arr = np.array(image.convert("RGB"))
        result   = image.copy().convert("RGB")
        draw     = ImageDraw.Draw(result)
        img_w, img_h = result.size

        bg_brightness = sum(bg_color) / 3
        global_bg = bg_color

        # ── Normalise char_heights across the page ────────────────────────
        valid_lines = [l for l in lines if l.get("text_bbox")]
        raw_heights = [l["char_height"] for l in valid_lines]
        norm_heights = _cluster_heights(raw_heights)
        height_map = {id(l): h for l, h in zip(valid_lines, norm_heights)}

        for line in lines:
            text = line.get("translated", "").strip()
            if not text or not line.get("text_bbox"):
                continue

            x1, y1, x2, y2 = map(int, line["text_bbox"])
            char_height = height_map.get(id(line), line["char_height"])

            # ── Sample original for colour + weight ───────────────────────
            text_color = _sample_text_color(orig_arr, x1, y1, x2, y2)
            bold = _is_bold(orig_arr, x1, y1, x2, y2, bg_brightness)

            # ── Fit text: wrap within bbox width, scale down if too tall ──
            # PaddleOCR bboxes include ~20 % vertical padding beyond actual ink.
            # Scaling down to 0.80 of the bbox height matches the original visual size.
            target_px = max(6, int(char_height * 0.80))
            font_path = self.font_bold if bold else self.font_regular
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            font, wrapped = _fit_text(text, target_px, bbox_w, bbox_h, font_path)

            # ── Measure full rendered block ────────────────────────────────
            line_h   = _line_height(font)
            gap      = max(1, int(line_h * 0.15))
            block_h  = len(wrapped) * line_h + (len(wrapped) - 1) * gap
            block_w  = max(
                (_MDRAW.textbbox((0, 0), l, font=font)[2] for l in wrapped),
                default=bbox_w,
            )

            # ── Skip text that sits over a stamp, seal, or graphic ───────
            # If the pixels directly above/below the bbox are dark the text
            # lives inside a graphical element — erasing it would destroy the
            # graphic, so we leave this region completely untouched.
            if _is_over_graphic(orig_arr, x1, y1, x2, y2, img_h, img_w):
                continue

            # ── Erase original: solid fill with local background colour ───
            local_bg = _sample_local_bg(orig_arr, x1, y1, x2, y2, img_h, img_w) or global_bg
            erase_x2 = min(x1 + max(block_w, bbox_w), img_w - 1)
            erase_y2 = min(y1 + max(block_h, bbox_h), img_h - 1)

            if erase_x2 > x1 and erase_y2 > y1:
                draw.rectangle([x1, y1, erase_x2, erase_y2], fill=local_bg)

            # ── Draw wrapped lines top-to-bottom ─────────────────────────
            y_cursor = y1
            for wline in wrapped:
                draw.text((x1, y_cursor), wline, font=font, fill=text_color)
                y_cursor += line_h + gap

        return result


# ── Module-level helpers ──────────────────────────────────────────────────────

def _is_over_graphic(
    arr: np.ndarray, x1: int, y1: int, x2: int, y2: int, img_h: int, img_w: int
) -> bool:
    """
    Return True when the bbox sits inside a stamp, seal, or graphic element.

    Samples thin strips directly above, below, and at the sides of the bbox.
    On clean page text those strips are bright white paper (median > 170).
    Inside a stamp, seal, or logo they contain dark ink or coloured fill and
    the median brightness drops below 170.
    """
    pad = max(6, (y2 - y1) // 2)
    strips = []
    if y1 >= pad:
        strips.append(arr[max(0, y1 - pad):y1, x1:x2])
    if y2 + pad <= img_h:
        strips.append(arr[y2:min(img_h, y2 + pad), x1:x2])
    margin = max(2, (x2 - x1) // 8)
    strips.append(arr[y1:y2, x1:x1 + margin])
    strips.append(arr[y1:y2, max(x1, x2 - margin):x2])

    pixels = np.concatenate([s.reshape(-1, 3) for s in strips if s.size > 0])
    if pixels.size == 0:
        return False
    return float(np.median(pixels.mean(axis=1))) < 170


def _sample_local_bg(
    arr: np.ndarray, x1: int, y1: int, x2: int, y2: int, img_h: int, img_w: int
) -> tuple | None:
    """
    Sample background colour from thin strips directly above and below the
    text bbox.  These pixels are background-only and give an accurate local
    colour regardless of global page tint or shadows.
    Returns None if not enough bright pixels are found (caller uses global bg).
    """
    pad = max(4, (y2 - y1) // 3)
    strips = []
    if y1 >= pad:
        strips.append(arr[max(0, y1 - pad):y1, x1:x2])
    if y2 + pad <= img_h:
        strips.append(arr[y2:min(img_h, y2 + pad), x1:x2])

    if not strips:
        return None

    pixels = np.concatenate([s.reshape(-1, 3) for s in strips if s.size > 0])
    if pixels.size == 0:
        return None

    brightness = pixels.mean(axis=1)
    bright = pixels[brightness > 170]
    if len(bright) < 5:
        bright = pixels  # fallback: use all sampled pixels

    median = np.median(bright, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list:
    """Split text into lines that each fit within max_w pixels."""
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        bb = _MDRAW.textbbox((0, 0), candidate, font=font)
        if (bb[2] - bb[0]) <= max_w:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    bb = _MDRAW.textbbox((0, 0), "Ag", font=font)
    return max(1, bb[3] - bb[1])


def _fit_text(
    text: str,
    target_px: int,
    bbox_w: int,
    bbox_h: int,
    font_path: str,
) -> tuple:
    """
    Return (font, wrapped_lines) that fit within bbox_w × bbox_h.

    Strategy: wrap at bbox_w first (no horizontal overflow ever), then
    reduce font size in steps until the wrapped block fits vertically.
    Falls back to the smallest readable size if nothing fits.
    """
    for scale in (1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40):
        size = max(6, int(target_px * scale))
        font = _load_font(font_path, size)
        wrapped = _wrap_text(text, font, bbox_w)
        if not wrapped:
            continue
        lh = _line_height(font)
        gap = max(1, int(lh * 0.15))
        total_h = len(wrapped) * lh + (len(wrapped) - 1) * gap
        if total_h <= bbox_h * 1.3:   # allow 30 % height overflow before shrinking more
            return font, wrapped
    font = _load_font(font_path, 6)
    return font, _wrap_text(text, font, bbox_w)


def _cluster_heights(heights: list) -> list:
    """
    Normalise char heights so table rows render at a consistent font size.

    All lines within ±35 % of the page-median height are "body text" and
    are set to a shared body median.  Heading/title lines (well above the
    median) keep their individual sizes.
    """
    if not heights:
        return heights

    med    = statistics.median(heights)
    result = list(heights)

    body_indices = [
        i for i, h in enumerate(heights)
        if med > 0 and abs(h - med) / med <= 0.35
    ]
    if len(body_indices) >= 2:
        body_med = statistics.median(heights[i] for i in body_indices)
        for i in body_indices:
            result[i] = body_med

    return result


def _sample_text_color(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple:
    """Return the colour of the darkest pixels inside the bbox."""
    region = arr[y1:y2, x1:x2]
    if region.size == 0:
        return (20, 20, 20)
    brightness = region.mean(axis=2)
    dark_mask  = brightness < 120
    if dark_mask.sum() == 0:
        return (20, 20, 20)
    color = region[dark_mask].mean(axis=0).astype(int)
    return (int(color[0]), int(color[1]), int(color[2]))


def _is_bold(
    arr: np.ndarray, x1: int, y1: int, x2: int, y2: int, bg_brightness: float
) -> bool:
    """
    True when ink density is high relative to the page background.
    Adaptive threshold: 55 % of background brightness.
    """
    region = arr[y1:y2, x1:x2]
    if region.size == 0:
        return False
    brightness    = region.mean(axis=2)
    ink_threshold = bg_brightness * 0.55
    dark_ratio    = (brightness < ink_threshold).sum() / brightness.size
    return dark_ratio > _BOLD_THRESHOLD


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()
