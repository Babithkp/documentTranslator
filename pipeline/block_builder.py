import re


class BlockBuilder:
    def build_blocks(self, merged_lines, gap_threshold=18):
        merged_lines = sorted(
            merged_lines,
            key=lambda x: x["bbox"][1]
        )

        blocks = []
        current = []

        for line in merged_lines:
            if not current:
                current.append(line)
                continue

            prev = current[-1]

            if self._should_merge(prev, line, gap_threshold):
                current.append(line)
            else:
                blocks.append(self._merge_block(current))
                current = [line]

        if current:
            blocks.append(self._merge_block(current))

        return blocks

    def _should_merge(self, prev, current, gap_threshold):
        prev_box = prev["bbox"]
        curr_box = current["bbox"]

        vertical_gap = curr_box[1] - prev_box[3]

        if vertical_gap > gap_threshold:
            return False

        left_diff = abs(prev_box[0] - curr_box[0])

        if left_diff > 30:
            return False

        prev_text = prev["text"].strip()

        if self._looks_like_table_row(prev_text):
            return False

        return True

    def _looks_like_table_row(self, text):
        digits = sum(c.isdigit() for c in text)
        spaces = text.count(" ")

        if digits >= 4 and spaces >= 3:
            return True

        if re.search(r'\b[A-F][+]?\b', text):
            return True

        return False

    def _merge_block(self, lines):
        text = "\n".join(line["text"] for line in lines)

        x1 = min(line["bbox"][0] for line in lines)
        y1 = min(line["bbox"][1] for line in lines)
        x2 = max(line["bbox"][2] for line in lines)
        y2 = max(line["bbox"][3] for line in lines)

        words = []

        for line in lines:
            words.extend(line["words"])

        return {
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "words": words
        }