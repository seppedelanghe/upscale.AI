import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import matplotlib
import matplotlib.pyplot as plt

import time
from tqdm import tqdm

from srcnn import SRCNN
from dataset import SRCNNImageDataset
from utils import psnr

import wandb

matplotlib.style.use('ggplot')


# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, dest='dir', help='Directory of images for training', required=True)
parser.add_argument('--val', type=str, dest='val', help='Directory of images for validation', required=True)
parser.add_argument('--limit', type=int, dest='limit', help='Limit of images to load for training', default=-1)
parser.add_argument('--size', type=int, dest='size', help='Size of the images to train the model on', default=33)
parser.add_argument('--epochs', type=int, dest='epochs', help='Number of epochs to train the model for', default=20)
parser.add_argument('--lr', type=float, dest='lr', help='Learning rate of the optimizer', default=0.001)
parser.add_argument('--bs', type=int, dest='bs', help='Batch size', default=64)
parser.add_argument('--wab', action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

# parameters
train_dir = args.dir
validation_dir = args.val
image_limit = args.limit
img_size = args.size # in pixels => always square (65, 65)
epochs = args.epochs
lr = args.lr
batch_size = args.bs
wab = args.wab

# auto detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
train_data = SRCNNImageDataset(train_dir, (img_size, img_size), limit=image_limit)
val_data = SRCNNImageDataset(validation_dir, (img_size, img_size))

# train and validation loaders
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

if wab:
    wandb.init(project='upscaler')

# initialize the model
model = SRCNN().to(device)
print(model)

# optimizer and optim
optimizer = optim.Adam(model.parameters(), lr=lr) 
criterion = nn.MSELoss()

def train(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    for data in dataloader:
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)

        # backpropagation
        loss.backward()

        # update the parameters
        optimizer.step()

        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()

        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(len(train_data)/dataloader.batch_size)

    if wab:
        wandb.log({
            'training loss': final_loss,
            'training psnr': final_psnr,
        }, step=epoch)

    return final_loss, final_psnr


def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0

    with torch.no_grad():
        for lowres, highres in dataloader:
            lowres = lowres.to(device)
            highres = highres.to(device)
            
            outputs = model(lowres)
            loss = criterion(outputs, highres)
            # add loss of each item (total items in a batch = batch size) 
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(highres, outputs)
            running_psnr += batch_psnr

        outputs = outputs.cpu()
        save_image(outputs, f"./outputs/upscaled{epoch}.png")
        save_image(lowres, f"./outputs/lowres{epoch}.png")
        save_image(highres, f"./outputs/original{epoch}.png")

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(val_data)/dataloader.batch_size)

    if wab:
        wandb.log({
            'validation loss': final_loss,
            'validation psnr': final_psnr,
            'validation images': [
                wandb.Image(f"./outputs/lowres{epoch}.png", caption="lowres"),
                wandb.Image(f"./outputs/upscaled{epoch}.png", caption="upscaled"),
                wandb.Image(f"./outputs/original{epoch}.png", caption="original")
            ]
        }, step=epoch)

    return final_loss, final_psnr


def main():
    os.makedirs('./outputs', exist_ok=True)

    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()

    loop = tqdm(range(epochs), total=epochs, leave=True)
    for epoch in loop:
        train_epoch_loss, train_epoch_psnr = train(model, train_loader, epoch)
        val_epoch_loss, val_epoch_psnr = validate(model, val_loader, epoch)

        loop.set_postfix({
            'Train PSNR': f"{train_epoch_psnr:.3f}",
            'Val PSNR': f"{val_epoch_psnr:.3f}"
        })

        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)

        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

        if epoch % 3 == 0:
            train_data.shuffle() # load different images

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

    if not wab:
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(val_loss, color='red', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./outputs/loss.png')
        plt.show()

        # psnr plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_psnr, color='green', label='train PSNR dB')
        plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.savefig('./outputs/psnr.png')
        plt.show()

    # save the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), './model.pth')


if __name__ == "__main__":
    main()