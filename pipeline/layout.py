"""
Words → lines → blocks.

Each block exposes two surfaces:
  - text / bbox          : everything OCR found (for reference)
  - translatable_text    : only the alphabetic/word tokens (sent to translation)
  - translatable_bbox    : bounding box of those tokens only (erased + redrawn)
  - translatable_height  : average char height of those tokens (drives font size)

Numbers, codes, and symbols are never sent to translation and their pixel
regions are never touched.
"""

import re
import statistics


# ── Public API ────────────────────────────────────────────────────────────────

def build_lines(words: list) -> list:
    """
    Group words into lines; split each line into text vs numeric tokens.

    Returns one dict per line that has any translatable text:
      - text          : joined text words (sent to translator)
      - text_bbox     : tight bbox of those words (erased + redrawn)
      - char_height   : average word height in pixels (drives font size)
      - full_text     : full line text including numbers (for context / debug)
    """
    if not words:
        return []

    raw_lines = _group_into_lines(words)
    result = []

    for raw in raw_lines:
        merged = _merge_line(raw)
        text_words = [w for w in merged["words"] if is_text_word(w["text"])]

        if not text_words:
            continue

        t_bbox = _bbox_of(text_words, key=lambda w: tuple(w["bbox"]))
        t_height = statistics.mean(
            w["bbox"][3] - w["bbox"][1] for w in text_words
        )

        result.append({
            "text": " ".join(w["text"] for w in text_words),
            "text_bbox": t_bbox,
            "char_height": t_height,
            "full_text": merged["text"],
        })

    return result


def build_layout(words: list, dpi: int = 150) -> list:
    """Return blocks from a flat word list (output of OCRProcessor.extract)."""
    if not words:
        return []

    lines = _group_into_lines(words)
    merged = [_merge_line(line) for line in lines]
    return [_merge_block(group) for group in _group_into_blocks(merged)]


_GRADE_RE = re.compile(r'^[A-F][+-]?$')

def is_text_word(token: str) -> bool:
    """True when the token contains translatable natural language."""
    token = token.strip()
    if not token:
        return False
    # Pure numeric
    if re.fullmatch(r'[\d.,/]+', token):
        return False
    # No alphabetic content at all
    if not any(c.isalpha() for c in token):
        return False
    # Grade letters: A, B+, C-, D, F, etc.
    if _GRADE_RE.match(token):
        return False
    # All-caps abbreviations with no vowels: CCT, CCP, GPT, BSC, etc.
    alpha_only = ''.join(c for c in token if c.isalpha())
    if alpha_only and alpha_only.isupper() and not any(c in 'AEIOU' for c in alpha_only):
        return False
    # Must be at least 40 % alphabetic (filters "CS101", "P.O.3")
    alpha_ratio = sum(c.isalpha() for c in token) / len(token)
    return alpha_ratio >= 0.4


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_into_lines(words: list) -> list:
    heights = [w["y2"] - w["y1"] for w in words]
    median_h = statistics.median(heights) if heights else 12
    threshold = max(4, median_h * 0.5)

    sorted_words = sorted(words, key=lambda w: (w["y1"] + w["y2"]) / 2)
    lines: list = []

    for word in sorted_words:
        mid = (word["y1"] + word["y2"]) / 2
        placed = False
        for line in reversed(lines):
            line_mid = sum((w["y1"] + w["y2"]) / 2 for w in line) / len(line)
            if abs(mid - line_mid) <= threshold:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    # After vertical grouping, split any line that spans multiple columns.
    # A large horizontal gap between adjacent words means they belong to
    # separate columns and must not be drawn as one merged line.
    # Column gap = 7 % of content width.  This reliably splits two-column
    # layouts (column gaps are typically 15-25 % of page width) without
    # breaking single-column text with wide word spacing (2-4 % of width).
    x_max = max(w["x2"] for w in words) if words else 800
    col_gap_threshold = max(40, x_max * 0.07)

    split_lines = []
    for line in lines:
        split_lines.extend(_split_by_column_gap(line, col_gap_threshold))

    return split_lines


def _split_by_column_gap(words: list, gap_threshold: float) -> list:
    """Return sub-groups of words separated by gaps wider than gap_threshold."""
    if len(words) <= 1:
        return [words]
    sorted_words = sorted(words, key=lambda w: w["x1"])
    groups, current = [], [sorted_words[0]]
    for word in sorted_words[1:]:
        if word["x1"] - current[-1]["x2"] > gap_threshold:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    groups.append(current)
    return groups


def _merge_line(words: list) -> dict:
    words = sorted(words, key=lambda w: w["x1"])
    heights = [w["y2"] - w["y1"] for w in words]

    return {
        "text": " ".join(w["text"] for w in words),
        "bbox": _bbox_of(words, key=lambda w: (w["x1"], w["y1"], w["x2"], w["y2"])),
        "words": [
            {"text": w["text"], "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]]}
            for w in words
        ],
        "avg_word_height": sum(heights) / len(heights),
    }


def _group_into_blocks(lines: list) -> list:
    if not lines:
        return []

    lines = sorted(lines, key=lambda l: l["bbox"][1])
    avg_h = statistics.mean(l["avg_word_height"] for l in lines)
    gap_threshold = avg_h * 1.8

    groups: list = []
    current = [lines[0]]

    for line in lines[1:]:
        gap = line["bbox"][1] - current[-1]["bbox"][3]
        left_shift = abs(line["bbox"][0] - current[0]["bbox"][0])
        if gap > gap_threshold or left_shift > avg_h * 4:
            groups.append(current)
            current = [line]
        else:
            current.append(line)

    groups.append(current)
    return groups


def _merge_block(lines: list) -> dict:
    all_words = [w for l in lines for w in l["words"]]
    all_heights = [l["avg_word_height"] for l in lines]

    # Split words into text vs numeric at token level
    text_words = [w for w in all_words if is_text_word(w["text"])]

    if text_words:
        t_bbox = _bbox_of(text_words, key=lambda w: tuple(w["bbox"]))
        t_height = statistics.mean(w["bbox"][3] - w["bbox"][1] for w in text_words)
        t_text = " ".join(w["text"] for w in text_words)
    else:
        t_bbox = None
        t_height = statistics.mean(all_heights)
        t_text = ""

    return {
        # Full block (reference only — not erased or translated)
        "text": "\n".join(l["text"] for l in lines),
        "bbox": _bbox_of(all_words, key=lambda w: tuple(w["bbox"])),
        "line_count": len(lines),
        "avg_char_height": statistics.mean(all_heights),
        # Translation surface (text tokens only)
        "translatable_text": t_text,
        "translatable_bbox": t_bbox,
        "translatable_height": t_height,
    }


def _bbox_of(items: list, key) -> list:
    coords = [key(i) for i in items]
    return [
        min(c[0] for c in coords),
        min(c[1] for c in coords),
        max(c[2] for c in coords),
        max(c[3] for c in coords),
    ]
