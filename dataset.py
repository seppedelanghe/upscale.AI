import math
import os
import random
import numpy as np

from PIL import Image
from PIL.ImageOps import grayscale
from torch.utils.data import Dataset
from utils import divide_image, downscale_image, getsize


class SRCNNImageDataset(Dataset):
    def __init__(self, image_dir: str, output_size: tuple, grayscale: bool = True, cache: bool = False, memlimit: int = -1, scale: int = 2):
        super().__init__()
        
        if not os.path.isdir(image_dir):
            raise Exception('Invalid image directory')

        self.output_size = output_size
        self.gray = grayscale
        self.memlimit = memlimit
        self.scale = scale
        self.cache = cache
        
        self.paths = [os.path.join(image_dir, p) for p in os.listdir(image_dir) if '.jpg' in p or '.png' in p]
        self.data = []

        self.shuffle()

    @property
    def memory(self):
        return getsize(self.data)
    
    @property
    def memory_limit(self):
        return self.memlimit != -1 and self.memory >= self.memlimit

    def shuffle(self):
        random.shuffle(self.paths)
        if self.cache:
            self.prepare()

    def prepare(self):
        self.data.clear()
        
        for p in self.paths:
            if self.memory_limit:
                break # break both loops

            image = Image.open(p)
            if self.gray:
                image = grayscale(image)

            # split larger image into subset
            images = divide_image(np.array(image), self.output_size, self.gray)

            # generate lower res. image for each subimg and add to data
            for im in images:
                lowres, original = self.make_combo(im)
                self.data.append((lowres, original))

    def make_combo(self, im: np.ndarray):
        lowres = downscale_image(im, self.scale)
        lowres = lowres.reshape((1 if self.gray else 3, self.output_size[0] // 2, self.output_size[1] // 2)).astype('float32') / 255

        original = im.reshape((1, *self.output_size) if self.gray else (self.output_size[2], self.output_size[0], self.output_size[1])).astype('float32') / 255
        self.data.append((lowres, original))

        return lowres, original

    def get_image_by_idx(self, idx):
        n = 0

        for p in self.paths:
            im = Image.open(p)
            if self.gray:
                im = grayscale(im)
            
            m = math.floor(im.size[0] // self.output_size[0] * im.size[1] // self.output_size[1])
            if idx >= n and idx < n + m:
                images = divide_image(np.array(im), self.output_size, self.gray)
                return self.make_combo(images[idx-n])

            n += m

    def __len__(self):
        if self.cache:
            return len(self.data)

        n = 0
        for p in self.paths:
            im = Image.open(p)
            n += math.floor(im.size[0] // self.output_size[0] * im.size[1] // self.output_size[1])

        return n

    def __getitem__(self, index):
        return self.data[index] if self.cache else self.get_image_by_idx(index)