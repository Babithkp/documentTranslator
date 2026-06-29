from PIL import Image, ImageDraw, ImageFont


class TextFitter:
    
    def __init__(self, font_path):
        self.font_path = font_path
        self._dummy = Image.new("RGB", (1, 1))
        self._draw = ImageDraw.Draw(self._dummy)

    def fit(self, text, bbox, block_type="line"):
        x1, y1, x2, y2 = bbox

        box_width = int(x2 - x1)
        box_height = int(y2 - y1)

        best = None

        start_size, end_size = self._font_range(block_type)

        for font_size in range(start_size, end_size - 1, -1):
            font = ImageFont.truetype(
                self.font_path,
                font_size
            )

            lines = self._wrap_text(
                text,
                font,
                box_width
            )

            width, height = self._measure_block(
                lines,
                font
            )

            if width <= box_width and height <= box_height:
                best = {
                "lines": lines,
                "font_size": int(font_size * 0.75),
                "text_width": width,
                "text_height": height,
                "draw_x": x1,
                "draw_y": self._compute_y(
                    y1, box_height, height, block_type
                ),
                "overflow": False
            }
                break

        if best:
            return best

        # fallback: smallest font
        font = ImageFont.truetype(self.font_path, 4)
        lines = self._wrap_text(text, font, box_width)
        width, height = self._measure_block(lines, font)

        return {
            "lines": lines,
            "font_size": 4,
            "text_width": width,
            "text_height": height,
            "draw_x": x1,
            "draw_y": y1,
            "overflow": True
        }

    def _wrap_text(self, text, font, max_width):
        words = text.split()
        lines = []
        current = ""

        for word in words:
            trial = word if not current else current + " " + word
            width = self._text_width(trial, font)

            if width <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines

    def _measure_block(self, lines, font):
        max_width = 0
        total_height = 0

        for line in lines:
            bbox = self._draw.textbbox(
                (0, 0),
                line,
                font=font
            )

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            max_width = max(max_width, width)
            total_height += height + 2

        return max_width, total_height

    def _text_width(self, text, font):
        bbox = self._draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        return bbox[2] - bbox[0]

    def _compute_y(self, y1, box_height, text_height, block_type):
        if block_type == "header":
            return y1 + (box_height - text_height) / 2

        return y1
    
    def _font_range(self, block_type):
        if block_type == "header":
            return 28, 10

        if block_type == "table_header":
            return 14, 6

        if block_type == "table_row":
            return 9, 6

        if block_type == "footer":
            return 7, 4

        if block_type == "paragraph":
            return 12, 6

        return 14, 5