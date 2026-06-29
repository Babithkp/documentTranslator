import re


class TextCleaner:
    def clean(self, text):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text