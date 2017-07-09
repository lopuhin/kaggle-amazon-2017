import random
import math

from PIL import Image
from torchvision import transforms

import utils


class RandomTranspose:
    def __call__(self, image):
        op = random.choice([
            None,
            Image.FLIP_LEFT_RIGHT,
            Image.FLIP_TOP_BOTTOM,
            Image.ROTATE_90,
            Image.ROTATE_180,
            Image.ROTATE_270,
            Image.TRANSPOSE,
        ])
        if op is not None:
            image = image.transpose(op)
        return image


class RandomSizedCrop:
    """Random crop the given PIL.Image to a random size of (0.25 to 1.0)
    of the original size and and a random aspect ratio of 3/4 to 4/3
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_aspect=4/5, max_aspect=5/4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = transforms.Scale(self.size, interpolation=self.interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(scale(img))


default_transform = transforms.Compose([
    transforms.RandomCrop(224),
    RandomTranspose(),
])

with_scale_transform = transforms.Compose([
    RandomSizedCrop(224),
    RandomTranspose(),
])
