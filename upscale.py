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
parser.add_argument('--gray', action=argparse.BooleanOptionalAction, default=False)


model = SRCNN().to(device)

def upscale(subimage, gray: bool = False):
    im = np.array(subimage).astype('float32') / 255
    # im = im.reshape((*im.shape, 1 if gray else 3))

    model.eval()
    with torch.no_grad():
        im = np.transpose(im, (2, 0, 1))
        im = torch.tensor(im, dtype=torch.float).to(device)
        im = im.unsqueeze(0)
        out: torch.Tensor = model(im)
    
    # convert image to output
    out: torch.Tensor = out.cpu().detach().squeeze()
    out = (out.numpy() * 255).astype('uint8')
    out = out.transpose((1, 2, 0))
    out: Image = Image.fromarray(out)

    return out

def main():
    args = parser.parse_args()
    if not os.path.isfile(args.img):
        raise Exception('invalid image path')

    if not os.path.isfile(args.model):
        raise Exception('invalid model path')

    model.load_state_dict(torch.load(args.model))

    image = Image.open(args.img)
    if args.gray:
        image = ImageOps.grayscale(image)

    out = upscale(image, args.gray)

    # make output dir
    os.makedirs(args.dest, exist_ok=True)

    # save image
    image.save(os.path.join(args.dest, 'original.png'))
    out.save(os.path.join(args.dest, 'upscaled.png'))

if __name__ == "__main__":
    main()