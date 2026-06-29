import re


class BlockClassifier:
    def classify(self, block):
        text = block["text"].strip()

        if self._is_footer(text):
            return "footer"

        if self._is_table_header(text):
            return "table_header"

        if self._is_section_header(text):
            return "section_header"

        if self._is_table_row(text):
            return "table_row"

        if self._is_code(text):
            return "code"

        if self._is_paragraph(text):
            return "paragraph"

        if self._is_header(text):
            return "header"

        return "line"

    def should_translate(self, block_type):
        return block_type not in ["footer", "code"]

    def _is_footer(self, text):
        if re.search(r'page\s*:?\s*\d+\s*of\s*\d+', text, re.I):
            return True
        return False

    def _is_table_row(self, text):
        lines = text.split("\n")

        for line in lines:
            digits = sum(c.isdigit() for c in line)
            spaces = line.count(" ")

            grade = re.search(r'\b[A-F][+]?\b', line)

            if digits >= 3 and spaces >= 3:
                return True

            if grade:
                return True

        return False

    def _is_code(self, text):
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)

        return digits > letters and len(text.split()) <= 3

    def _is_paragraph(self, text):
        words = len(text.split())

        if "\n" in text:
            return True

        if words > 12:
            return True

        return False

    def _is_header(self, text):
        words = len(text.split())

        if words <= 8 and text.upper() == text:
            return True

        return False
    
    def _is_table_header(self, text):
        keywords = [
            "COURSE",
            "TITLE",
            "GRADE",
            "CREDIT"
        ]

        upper = text.upper()

        count = sum(1 for k in keywords if k in upper)

        return count >= 2

    def _is_section_header(self, text):
        keywords = [
            "SEMESTER",
            "LEVEL",
            "YEAR",
            "SESSION"
        ]

        upper = text.upper()

        for word in keywords:
            if word in upper:
                return True

        return False