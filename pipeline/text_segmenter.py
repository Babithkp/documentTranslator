import re


class TextSegmenter:
    def segment(self, block, block_type):
        text = block["text"]
        words = block["words"]

        if block_type in ["footer", "code"]:
            return {
                "protected": [text],
                "translatable": "",
                "translatable_bbox": None
            }

        if block_type != "table_row":
            return {
                "protected": [],
                "translatable": text,
                "translatable_bbox": block["bbox"]
            }

        protected_words = []
        translatable_words = []

        for word in words:
            token = word["text"]

            if self._is_protected(token):
                protected_words.append(word)
            else:
                translatable_words.append(word)

        translatable_text = " ".join(
            w["text"] for w in translatable_words
        )

        bbox = self._compute_bbox(translatable_words)

        return {
            "protected": [
                w["text"] for w in protected_words
            ],
            "translatable": translatable_text,
            "translatable_bbox": bbox
        }

    def _compute_bbox(self, words):
        if not words:
            return None

        x1 = min(w["bbox"][0] for w in words)
        y1 = min(w["bbox"][1] for w in words)
        x2 = max(w["bbox"][2] for w in words)
        y2 = max(w["bbox"][3] for w in words)

        return [x1, y1, x2, y2]

    def _is_protected(self, token):
        token = token.strip()

        if re.fullmatch(r'[A-Z]{2,}\d+', token):
            return True

        if re.fullmatch(r'\d+(\.\d+)?', token):
            return True

        if re.fullmatch(r'[A-F][+]?', token):
            return True

        return False