import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL.ImageOps import grayscale
from torch.utils.data import Dataset
from utils import divide_image, downscale_image


class SRCNNImageDataset(Dataset):
    def __init__(self, image_dir: str, output_size: tuple, grayscale: bool = True, limit: int = -1, scale: int = 2):
        super().__init__()
        
        if not os.path.isdir(image_dir):
            raise Exception('Invalid image directory')

        self.output_size = output_size
        self.gray = grayscale
        self.limit = limit
        self.scale = scale
        
        self.paths = [os.path.join(image_dir, p) for p in os.listdir(image_dir) if '.jpg' in p or '.png' in p]

        self.data = []

        self.prepare()

    def shuffle(self):
        random.shuffle(self.paths)
        self.prepare()

    def prepare(self):
        self.data .clear()
        n = self.limit if self.limit > 0 else len(self.paths)
        for i in tqdm(range(n), total=n):
            p = self.paths[i]
            image = Image.open(p)
            if self.gray:
                image = grayscale(image)

            # split larger image into subset
            images = divide_image(np.array(image), self.output_size)

            # generate lower res. image for each subimg and add to data
            for im in images:
                lowres = downscale_image(im, self.scale)
                lowres = lowres.reshape((1, *self.output_size)).astype('float32') / 255

                original = im.reshape((1, *self.output_size)).astype('float32') / 255
                self.data.append((lowres, original))

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, index):
        return self.data[index]