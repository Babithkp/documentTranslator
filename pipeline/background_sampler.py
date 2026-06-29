from PIL import Image
import numpy as np


class BackgroundSampler:
    def sample(self, image_path, bbox):
        img = Image.open(image_path).convert("RGB")
        crop = img.crop(tuple(map(int, bbox)))

        arr = np.array(crop)

        # ignore dark pixels (likely text)
        mask = np.mean(arr, axis=2) > 180

        if mask.sum() == 0:
            return (245, 240, 210)

        pixels = arr[mask]
        mean = pixels.mean(axis=0)

        return tuple(mean.astype(int))