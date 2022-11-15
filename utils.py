import math, sys
import torch
import numpy as np

from PIL import Image
from types import ModuleType, FunctionType
from gc import get_referents

def psnr(label: torch.Tensor, outputs: torch.Tensor, max_val: float = 1.0):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label

    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        return 20 * math.log10(max_val / rmse)


def downscale_image(image: np.ndarray, factor: int = 2, resample = Image.Resampling.BOX):
    curr = image.shape
    to = (curr[0] // factor, curr[1] // factor)
    im = Image.fromarray(image).resize(to, resample=resample)

    return np.array(im)

def divide_image(image: np.ndarray, size: tuple, gray = False):
    from numpy.lib.stride_tricks import sliding_window_view
    flatspace = (image.shape[0] // size[0]) * (image.shape[1] // size[1])
    s = sliding_window_view(image, size)

    if not gray:
        return s[::size[0], ::size[1], ::size[2]].reshape((flatspace, *size))
    else:
        return s[::size[0], ::size[1]].reshape((flatspace, *size))


BLACKLIST = type, ModuleType, FunctionType

def getsize(obj, f='mb', asint=True):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)

    if f == 'kb':
        size /= 1e3
    elif f == 'mb':
        size /= 1e6
    elif f == 'gb':
        size /= 1e9
    
    return round(size) if asint else size