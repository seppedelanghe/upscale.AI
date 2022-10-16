import math
import torch
import numpy as np
from PIL import Image

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


def downscale_image(image: np.ndarray, factor: int = 2, iters: int = 2, resample = Image.BOX):
    curr = image.shape
    to = (curr[0] // factor, curr[1] // factor)
    im = Image.fromarray(image)
    
    for _ in range(iters):
        im = im.resize(to, resample=resample).resize(curr) # scale down and up again

    return np.array(im)

def divide_image(image: np.ndarray, size: tuple):
    from numpy.lib.stride_tricks import sliding_window_view
    flatspace = (image.shape[0] // size[0]) * (image.shape[1] // size[1])
    return sliding_window_view(image, size)[::size[0], ::size[1]].reshape((flatspace, *size))


