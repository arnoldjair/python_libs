from pathlib import Path
from typing import Tuple

import numpy as np
import PIL
import torch
from PIL import Image


def load_image(image) -> Image:
    """
    Args:
        image: either path to image or actual image: PIL, numpy of Tensor (HxWxC dims)
    Returns:
        PIL.Image: Loaded image.
    """
    if isinstance(image, str) or isinstance(image, Path):
        with open(image, "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray(image.numpy())
    elif isinstance(image, PIL.Image.Image):
        return image
    else:
        raise ValueError(f"Unsupported input image type {type(image)}")

    return image


def image_size(image) -> Tuple[float, float]:
    if isinstance(image, str) or isinstance(image, Path):
        with open(image, "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB").size
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).size
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray(image.numpy()).size
    elif isinstance(image, PIL.Image.Image):
        return image.size
    else:
        raise ValueError(f"Unsupported input image type {type(image)}")

    return None
