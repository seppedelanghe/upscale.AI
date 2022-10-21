import os
import torch
import argparse
import numpy as np
from srcnn import SRCNN
from PIL import Image, ImageOps

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--img', dest='img', help='Path to the image to upscale', required=True)
parser.add_argument('--size', dest='size', help='Size of the images the model was trained on', default=33)
parser.add_argument('--dest', dest='dest', help='Destination to save upscaled image', default='./outputs')
parser.add_argument('--model', dest='model', help='Model to load', default='./checkpoints/best.pth')


model = SRCNN().to(device)

def upscale(subimage):
    im = np.array(subimage).astype('float32') / 255
    im = im.reshape((*im.shape, 1))

    model.eval()
    with torch.no_grad():
        im = np.transpose(im, (2, 0, 1))
        im = torch.tensor(im, dtype=torch.float).to(device)
        im = im.unsqueeze(0)
        out: torch.Tensor = model(im)
    
    # convert image to output
    out = out.cpu().detach()
    out = (out.numpy() * 255).astype('uint8').reshape(out.shape[2], out.shape[3], out.shape[1])
    out: Image = Image.fromarray(out.reshape((out.shape[0], out.shape[1])))

    return out

def main():
    args = parser.parse_args()
    if not os.path.isfile(args.img):
        raise Exception('invalid image path')

    if not os.path.isfile(args.model):
        raise Exception('invalid model path')

    model.load_state_dict(torch.load(args.model))

    image = Image.open(args.img)
    gray = ImageOps.grayscale(image)
    # up = gray.resize((gray.size[0] * 2, gray.size[1] * 2), resample=Image.Resampling.BOX)

    out = upscale(gray)

    # make output dir
    os.makedirs(args.dest, exist_ok=True)

    # save image
    gray.save(os.path.join(args.dest, 'original.png'))
    out.save(os.path.join(args.dest, 'upscaled.png'))

if __name__ == "__main__":
    main()