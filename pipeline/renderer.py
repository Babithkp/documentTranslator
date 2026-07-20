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

import bisect
import re
import statistics
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A bullet/list marker OCR left at the start of a line — a bullet glyph, or a
# lone "-", "*", ".", or "o" (how OCR commonly renders "•") followed by a space.
# The original "•" is destroyed inconsistently by OCR, so these artefacts read
# as stray periods; they are stripped for a clean, uniform list.
_LEADING_BULLET_RE = re.compile(r'^\s*(?:[•·▪◦‣●○∙]|[-*.oO])\s+')


def _strip_leading_bullet(text: str) -> str:
    return _LEADING_BULLET_RE.sub("", text, count=1)

_MIMG = Image.new("RGB", (1, 1))
_MDRAW = ImageDraw.Draw(_MIMG)

# Bold test: stroke width as a fraction of text height (scale-invariant).
# Stroke width is estimated from the distance transform of the ink.  Dividing
# by the line height makes the measure independent of scan resolution / font
# size, unlike raw ink density (which spikes on tightly-cropped tokens) or a
# fixed-pixel erosion (which reads every stroke as thick at high DPI).  The
# threshold sits just below genuine bold (headings/labels measure ~0.10-0.13
# stroke-width/height on these scans) and above regular body text (≤ ~0.08), so
# real bold is caught without body text or isolated codes going false-bold.
_BOLD_THRESHOLD = 0.10

# Translations (esp. German/Dutch) run longer than the English source.  Allow a
# line to use this much more than the original text width before it is wrapped
# or shrunk, so a slightly longer translation keeps the same font size as its
# neighbours instead of dropping a step and looking uneven.
_WIDTH_TOLERANCE = 1.25

# OCR bboxes are a little taller than the actual ink (ascender-to-descender plus
# padding).  Rendering at this fraction of the bbox height matches the original
# visual size closely without spilling into neighbouring lines.
_INK_HEIGHT_FACTOR = 0.90


class TextRenderer:
    def __init__(self, font_regular: str, font_bold: str):
        self.font_regular = font_regular
        self.font_bold = font_bold

    def overlay_lines(
        self,
        image: Image.Image,
        lines: list,
        bg_color: tuple,
        extra_erase_boxes: list = None,
    ) -> Image.Image:
        """
        Draw each translated line over the original image in place.

        `lines` items must have:
          - translated   : translated string
          - text_bbox    : [x1, y1, x2, y2] of the original text words
          - char_height  : average pixel height of original text words

        `extra_erase_boxes` are regions of leftover English that OCR missed
        (see BackgroundEraser.find_missed_text_boxes).  Their ink is removed too,
        but only when they do not sit inside a stamp, seal, or graphic.

        Erasing is done by inpainting: only the dark ink pixels are masked, and
        cv2 propagates the surrounding paper/watermark texture into them.  This
        removes the text without leaving the flat rectangle a solid fill would,
        so security watermarks and tinted backgrounds are preserved.
        """
        orig_arr = np.array(image.convert("RGB"))
        result   = image.copy().convert("RGB")
        img_w, img_h = result.size

        bg_brightness = sum(bg_color) / 3
        global_bg = bg_color

        # Ink pixels to remove across the whole page (one inpaint pass at the end).
        ink_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # ── Mask leftover text OCR could not read ────────────────────────────
        for box in (extra_erase_boxes or []):
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                continue
            if _is_over_graphic(orig_arr, x1, y1, x2, y2, img_h, img_w):
                continue
            _mask_ink(ink_mask, orig_arr, x1, y1, x2, y2)

        # ── Normalise char_heights across the page ────────────────────────
        valid_lines = [l for l in lines if l.get("text_bbox")]
        raw_heights = [l["char_height"] for l in valid_lines]
        norm_texts  = [l.get("text", "") for l in valid_lines]
        norm_heights = _cluster_heights(raw_heights, norm_texts)
        height_map = {id(l): h for l, h in zip(valid_lines, norm_heights)}

        # Vertical slot for each entry = distance down to the next block below.
        # A reflowed paragraph runs longer than its English source, so it is
        # fitted into this slot to avoid overlapping the following block.
        entry_tops = sorted(
            int(l["text_bbox"][1]) for l in lines if l.get("text_bbox")
        )

        # First pass: mask original ink and record where to draw each translation.
        draw_jobs = []
        for line in lines:
            text = _strip_leading_bullet(line.get("translated", "").strip())
            if not text or not line.get("text_bbox"):
                continue

            x1, y1, x2, y2 = map(int, line["text_bbox"])
            char_height = height_map.get(id(line), line["char_height"])

            # ── Sample original for colour + weight ───────────────────────
            text_color = _sample_text_color(orig_arr, x1, y1, x2, y2)
            bold = _is_bold(orig_arr, x1, y1, x2, y2, bg_brightness)

            target_px = max(6, int(char_height * _INK_HEIGHT_FACTOR))
            font_path = self.font_bold if bold else self.font_regular
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)

            # ── Skip text that sits over a stamp, seal, or graphic ───────
            # If the pixels directly above/below the bbox are dark the text
            # lives inside a graphical element — erasing it would destroy the
            # graphic, so we leave this region completely untouched.
            if _is_over_graphic(orig_arr, x1, y1, x2, y2, img_h, img_w):
                continue

            # ── Mask original ink pixels for inpainting ──────────────────
            _mask_ink(ink_mask, orig_arr, x1, y1, x2, y2)

            # Vertical room down to the next block below.  Both a reflowed
            # paragraph and a single line whose (longer) translation must wrap
            # use this so they grow downward into the empty space instead of
            # shrinking to fit one row — bounded so they never overrun the
            # following block.
            nxt = bisect.bisect_right(entry_tops, y1)
            slot_bottom = entry_tops[nxt] if nxt < len(entry_tops) else img_h
            avail_h = max(bbox_h, slot_bottom - y1 - 2)

            if line.get("reflow"):
                # Paragraph: re-wrap the whole block within its own width at a
                # uniform size, shrinking only if the (longer) translation would
                # run past the next block below.
                font, wrapped, pitch = _fit_paragraph(
                    text, target_px, bbox_w, avail_h, font_path
                )
                draw_jobs.append({
                    "x": x1, "y": y1, "wrapped": wrapped, "font": font,
                    "color": text_color, "pitch": pitch, "center_h": None,
                })
                continue

            # ── Fit line: wrap within bbox width (bounded tolerance), keep the
            #    font size and let it flow down into the slot below; shrink only
            #    when even that space is exhausted.  Capped so a line above a
            #    large gap does not balloon into many full-size rows. ──
            fit_w = min(int(bbox_w * _WIDTH_TOLERANCE), img_w - x1 - 2)
            fit_w = max(fit_w, bbox_w)
            # Floor keeps the old single-row descender slack when the next line
            # sits right below; cap stops a line above a big gap from filling it.
            line_avail = min(max(avail_h, int(bbox_h * 1.3)), bbox_h * 5)
            font, wrapped = _fit_text(text, target_px, fit_w, line_avail, font_path)
            line_h = _line_height(font)
            gap    = max(1, int(line_h * 0.15))
            block_h = len(wrapped) * line_h + (len(wrapped) - 1) * gap
            draw_jobs.append({
                "x": x1, "y": y1, "wrapped": wrapped, "font": font,
                "color": text_color, "pitch": line_h + gap, "center_h": (bbox_h, block_h),
            })

        # ── Inpaint the masked ink, preserving watermark/texture ─────────────
        if ink_mask.any():
            # Grow the mask so faint stroke halos are covered.  A wider halo is
            # safe: inpaint rebuilds the watermark from the untouched neighbours.
            ink_mask = cv2.dilate(ink_mask, np.ones((5, 5), np.uint8), iterations=1)
            bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            bgr = cv2.inpaint(bgr, ink_mask, 3, cv2.INPAINT_TELEA)
            result = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # ── Second pass: draw translations on the cleaned page ───────────────
        draw = ImageDraw.Draw(result)
        for job in draw_jobs:
            y_cursor = job["y"]
            # A single line is vertically centred on the row it replaces; a
            # paragraph starts at its top and flows down at the source pitch.
            if job["center_h"]:
                bbox_h, block_h = job["center_h"]
                y_cursor += max(0, (bbox_h - block_h) // 2)
            for wline in job["wrapped"]:
                draw.text((job["x"], y_cursor), wline, font=job["font"], fill=job["color"])
                y_cursor += job["pitch"]

        return result


# ── Module-level helpers ──────────────────────────────────────────────────────

def _mask_ink(
    mask: np.ndarray, arr: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> None:
    """
    Mark the dark ink pixels inside the bbox for inpainting.

    Only pixels clearly darker than the local paper are masked; the lighter
    watermark/tint pixels are left untouched so cv2.inpaint can rebuild the
    background texture from them instead of painting a flat patch.
    """
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(arr.shape[1], x2); y2 = min(arr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return

    region = arr[y1:y2, x1:x2]
    brightness = region.mean(axis=2)

    # Local paper brightness = the brighter half of the region.
    upper = brightness[brightness > brightness.mean()]
    bg_b = float(upper.mean()) if upper.size else float(brightness.mean())

    # Ink threshold: below paper but high enough to catch faint anti-aliased
    # stroke edges.  Capped at 175 — inside text regions the lightest watermark
    # pixels stay above ~180, so the tint survives while the ink is fully caught.
    thresh = min(bg_b * 0.68, 175.0)
    ink = brightness < thresh
    mask[y1:y2, x1:x2][ink] = 255


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
    avail_h: int,
    font_path: str,
) -> tuple:
    """
    Return (font, wrapped_lines) that fit within bbox_w × avail_h.

    Strategy: wrap at bbox_w first (no horizontal overflow ever), then
    reduce font size in steps until the wrapped block fits vertically within
    avail_h — the room down to the next block, so a longer translation wraps
    onto extra lines at full size rather than shrinking.  Falls back to the
    smallest readable size if nothing fits.
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
        if total_h <= avail_h:
            return font, wrapped
    font = _load_font(font_path, 6)
    return font, _wrap_text(text, font, bbox_w)


# A real heading is set in large type *and* carries enough text to justify it.
# A stray OCR box that is tall but holds only a short token (e.g. an isolated
# course code) is not a heading — clamping it prevents oversized single words.
_HEADING_MIN_CHARS = 10


def _fit_paragraph(
    text: str,
    target_px: int,
    width: int,
    avail_h: int,
    font_path: str,
) -> tuple:
    """
    Wrap a paragraph within `width` at the largest size (≤ target_px) whose
    reflowed height fits `avail_h`, so a longer translation never overruns the
    block that follows it.  Returns (font, wrapped_lines, line_pitch).
    """
    for scale in (1.0, 0.92, 0.85, 0.78, 0.7, 0.62, 0.55, 0.48, 0.4):
        size = max(6, int(target_px * scale))
        font = _load_font(font_path, size)
        wrapped = _wrap_text(text, font, width)
        pitch = max(1, int(_line_height(font) * 1.30))
        if len(wrapped) * pitch <= avail_h or size <= 6:
            return font, wrapped, pitch
    font = _load_font(font_path, 6)
    return font, _wrap_text(text, font, width), max(1, int(_line_height(font) * 1.30))


def _cluster_heights(heights: list, texts: list = None) -> list:
    """
    Normalise char heights so table rows render at a consistent font size.

    All lines within ±35 % of the page-median height are "body text" and
    are set to a shared body median.  Heading/title lines (well above the
    median) keep their individual sizes — but only if their text is long
    enough to be a genuine heading; a tall box holding a short token is an
    OCR artefact and is pulled back to the body size.
    """
    if not heights:
        return heights

    med    = statistics.median(heights)
    result = list(heights)

    body_indices = [
        i for i, h in enumerate(heights)
        if med > 0 and abs(h - med) / med <= 0.35
    ]
    body_med = med
    if len(body_indices) >= 2:
        body_med = statistics.median(heights[i] for i in body_indices)
        for i in body_indices:
            result[i] = body_med

    # Clamp tall outliers whose text is too short to be a real heading.
    if texts is not None:
        for i, h in enumerate(heights):
            if h > body_med * 1.35 and len(texts[i].strip()) < _HEADING_MIN_CHARS:
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
    True when the text strokes are thick (bold).

    Measures stroke thickness by how many ink pixels survive a 1px erosion:
    bold strokes keep most of their pixels, thin strokes lose most.  This is a
    ratio within the same region, so it is unaffected by how tightly OCR
    cropped the box (which made the old ink-density test misfire on isolated
    tokens).
    """
    region = arr[y1:y2, x1:x2]
    if region.size == 0:
        return False
    ink = (region.mean(axis=2) < bg_brightness * 0.6).astype(np.uint8)
    if int(ink.sum()) < 10:
        return False
    # Distance transform peaks at ~half the stroke width along each stroke's
    # centre line; average over ink pixels and double to get the mean stroke
    # width, then normalise by text height so the measure is scale-invariant.
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke_width = 2.0 * float(dt[ink > 0].mean())
    return (stroke_width / max(1, y2 - y1)) > _BOLD_THRESHOLD


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()
