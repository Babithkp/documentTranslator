"""
Erase text regions by sampling the page background colour and flood-filling.

Replaces: background_sampler + the cv2.inpaint approach in image_writer.
"""

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
