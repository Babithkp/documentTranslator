import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont,ImageFilter 


class ImageWriter:
    def __init__(self, image_path, font_path):
        self.image_path = image_path
        self.font_path = font_path
        self.cv_img = cv2.imread(image_path)
        

        self.pil_img = Image.open(image_path).convert("RGB")
        self.mask = np.zeros(
            self.cv_img.shape[:2],
            dtype=np.uint8
        )
        self.pil_img = self.pil_img.filter(
            ImageFilter.GaussianBlur(radius=0.3)
        )

    def add_mask_region(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(
            self.mask,
            (x1, y1),
            (x2, y2),
            255,
            -1
        )
    
    def apply_inpainting(self):
        self.cv_img = cv2.inpaint(
            self.cv_img,
            self.mask,
            1,
            cv2.INPAINT_TELEA
        )

        self.pil_img = Image.fromarray(
            cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        )
    
    def draw_block(self, bbox, fitted_text):
        x1, y1, x2, y2 = map(int, bbox)

        draw = ImageDraw.Draw(self.pil_img)

        font = ImageFont.truetype(
            self.font_path,
            fitted_text["font_size"]
        )

        x = x1
        y = y1
        line_spacing = 2

        for line in fitted_text["lines"]:
            draw.text(
                (x, y),
                line,
                fill=(50, 50, 50),
                font=font
            )

            bbox_line = draw.textbbox(
                (x, y),
                line,
                font=font
            )

            line_height = bbox_line[3] - bbox_line[1]
            y += line_height + line_spacing

        self.cv_img = cv2.cvtColor(
            np.array(self.pil_img),
            cv2.COLOR_RGB2BGR
        )

    def save(self, output_path):
        cv2.imwrite(output_path, self.cv_img)