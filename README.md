# PyTorch Image upscaling

Upscale images using a convolution neural network.
<br><br>

## Requirements
- Python >= 3.9

<br>

## Installation
```sh
# install pip packages
pip install -r requirements.txt

# install PyTorch => https://pytorch.org/
```

## Usage

__Train a model:__
```sh
# to see all options => --help

python train.py --dir /your/local/train/images/ --val /your/local/validation/images/
```

__Upscale an image:__
```sh
python upscale.py --img /your/local/image.png
```

