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

    # Each visual line becomes a list of segment entries.  A line split into
    # more than one segment is columnar (a table row); a single-segment line is
    # a candidate paragraph line.
    line_segments = []
    for raw in raw_lines:
        segs = []
        for segment in _split_line_into_segments(raw):
            merged = _merge_line(segment)
            text_words = [w for w in merged["words"] if is_text_word(w["text"])]
            if not text_words:
                continue
            segs.append({
                "text": " ".join(w["text"] for w in text_words),
                "text_bbox": _bbox_of(text_words, key=lambda w: tuple(w["bbox"])),
                "char_height": statistics.mean(
                    w["bbox"][3] - w["bbox"][1] for w in text_words
                ),
                "full_text": merged["text"],
            })
        if segs:
            line_segments.append(segs)

    # The true content right edge includes numbers/codes (not just text), so a
    # short table cell is never mistaken for a full-width prose line.
    page_right = max(w["x2"] for w in words)
    return _assemble_paragraphs(line_segments, page_right)


# ── Paragraph grouping ─────────────────────────────────────────────────────────
#
# Flowing prose must be translated and re-wrapped as a whole block, not line by
# line: a per-line translation runs longer than its source line and would be
# shrunk to fit, giving the wildly uneven font sizes seen on letters/essays.
# Table rows must NOT be merged this way or the columns collapse.
#
# The two are told apart by the right margin: a wrapped-prose line runs to the
# content right edge ("full width"); a table cell / short label does not.  A
# full-width line is therefore never the *last* line of a paragraph, so a
# paragraph keeps going as long as its current line is full width and the next
# line follows at normal leading.  It ends at the first short line.
#
# Gaps are kept tight: a dropped noise line leaves a ~1.4× gap that is
# indistinguishable from a real paragraph break, so we do NOT bridge it — the
# orphaned line is instead reflowed on its own (see the lone-full-width branch),
# which avoids both merging separate paragraphs and shrinking the orphan tiny.

_PARA_FILL_RATIO = 0.80    # a line reaching this fraction of the width is "full"
_PARA_GAP_RATIO = 0.8      # max vertical gap (× line height) within a paragraph


def _assemble_paragraphs(line_segments: list, page_right: float) -> list:
    """Merge consecutive prose lines into reflowable paragraph blocks."""
    line_segments.sort(key=lambda segs: min(s["text_bbox"][1] for s in segs))

    result = []
    i, n = 0, len(line_segments)
    while i < n:
        # Columnar lines (multiple segments) are never merged into paragraphs.
        if len(line_segments[i]) != 1:
            result.extend(line_segments[i])
            i += 1
            continue

        # Grow the paragraph while its last line is full width (i.e. not a
        # paragraph-ending short line) and the next line is left-aligned/near.
        para = [line_segments[i][0]]
        j = i + 1
        while j < n and len(line_segments[j]) == 1:
            prev, cur = para[-1], line_segments[j][0]
            if not _is_full_width(prev, page_right):
                break
            ph = prev["char_height"] or 1
            gap = cur["text_bbox"][1] - prev["text_bbox"][3]
            left_aligned = abs(cur["text_bbox"][0] - prev["text_bbox"][0]) <= ph * 1.5
            if gap <= ph * _PARA_GAP_RATIO and left_aligned:
                para.append(cur)
                j += 1
            else:
                break

        # A multi-line run, or a lone full-width line, is reflowed (re-wrapped
        # at a uniform size) instead of shrunk.  A lone short line (label,
        # heading, table cell) is left as an ordinary single line.
        if len(para) >= 2 or _is_full_width(para[0], page_right):
            result.append(_make_paragraph(para))
        else:
            result.append(para[0])
        i += len(para) if len(para) >= 2 else 1

    return result


def _is_full_width(line: dict, page_right: float) -> bool:
    """True when the line runs to the content right edge (wrapped prose), as
    opposed to a short ragged table cell / label."""
    x1, x2 = line["text_bbox"][0], line["text_bbox"][2]
    avail = page_right - x1
    return avail > 0 and (x2 - x1) >= _PARA_FILL_RATIO * avail


def _make_paragraph(para: list) -> dict:
    """Collapse a run of prose lines (or one full-width line) into a reflow entry."""
    x1 = min(p["text_bbox"][0] for p in para)
    y1 = min(p["text_bbox"][1] for p in para)
    x2 = max(p["text_bbox"][2] for p in para)
    y2 = max(p["text_bbox"][3] for p in para)
    # Original line pitch keeps the reflowed lines at the source line spacing.
    pitch = (y2 - y1) / (len(para) - 1) if len(para) > 1 else para[0]["char_height"] * 1.3
    return {
        "text": " ".join(p["text"] for p in para),
        "text_bbox": [x1, y1, x2, y2],
        "char_height": statistics.median([p["char_height"] for p in para]),
        "full_text": " ".join(p["full_text"] for p in para),
        "reflow": True,
        "line_pitch": pitch,
    }


# A single inter-word space is a small fraction of the font size (~0.2–0.3 em,
# and at most ~0.6× the text height even in loosely-set or justified text).  A
# gap wider than this many text-heights therefore cannot be an ordinary space —
# it is a deliberate column / field separation.  The threshold is expressed in
# text-heights so it scales with any font size, DPI, language, or document; it
# is not tied to any particular layout.
_COLUMN_GAP_RATIO = 1.5


def _split_line_into_segments(words: list) -> list:
    """
    Split a line into segments wherever the gap between neighbouring words is
    too wide to be a normal space (see _COLUMN_GAP_RATIO).

    OCR groups a whole logical line together even when its parts are spread
    across the page.  Joining those parts with single spaces and drawing from
    the left throws the layout away.  Splitting at the wide gaps keeps each
    field/column as its own segment, redrawn at its original x-position — the
    horizontal spacing is taken from the source geometry, never invented.
    """
    if len(words) <= 1:
        return [words]

    sorted_words = sorted(words, key=lambda w: w["x1"])
    heights = [w["y2"] - w["y1"] for w in sorted_words]
    median_h = statistics.median(heights) if heights else 12
    threshold = median_h * _COLUMN_GAP_RATIO

    groups, current = [], [sorted_words[0]]
    for word in sorted_words[1:]:
        if word["x1"] - current[-1]["x2"] > threshold:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    groups.append(current)
    return groups


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
