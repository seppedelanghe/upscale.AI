import argparse

# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, dest='dir', help='Directory of images for training', required=True)
parser.add_argument('--val', type=str, dest='val', help='Directory of images for validation', required=True)
parser.add_argument('--gray', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--memlimit', type=int, dest='memlimit', help='Memory limit for loading training data', default=1024)
parser.add_argument('--cache', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--size', type=int, dest='size', help='Size of the images to train the model on', default=64)
parser.add_argument('--epochs', type=int, dest='epochs', help='Number of epochs to train the model for', default=20)
parser.add_argument('--lr', type=float, dest='lr', help='Learning rate of the optimizer', default=0.001)
parser.add_argument('--bs', type=int, dest='bs', help='Batch size', default=64)
parser.add_argument('--saverate', type=int, dest='saverate', help='Save model every n epochs', default=5)
parser.add_argument('--shufflerate', type=int, dest='shufflerate', help='Shuffle/load new data every n epochs', default=5)
parser.add_argument('--wab', action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
from tqdm import tqdm

from srcnn import SRCNN, Denoise
from dataset import SRCNNImageDataset
from utils import psnr, psnr_loss, add_noise

import wandb

matplotlib.style.use('ggplot')

# parameters
train_dir = args.dir
validation_dir = args.val
gray = args.gray
mem_limit = args.memlimit
cache = args.cache
img_size = args.size # in pixels => always square (65, 65)
epochs = args.epochs
lr = args.lr
batch_size = args.bs
saverate = args.saverate
shufflerate = args.shufflerate
wab = args.wab

# auto detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# image shape
im_shape = (img_size, img_size) if gray else (img_size, img_size, 3)

# load data
train_data = SRCNNImageDataset(train_dir, im_shape, memlimit=mem_limit, cache=cache, grayscale=gray)
val_data = SRCNNImageDataset(validation_dir, im_shape, memlimit=mem_limit, cache=cache, grayscale=gray)

# train and validation loaders
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

if wab:
    wandb.init(project='upscaler')

# initialize the models
upscaler: SRCNN = SRCNN(gray=gray).to(device)
denoiser: Denoise = Denoise(gray=gray).to(device)

# optimizer and loss
upoptim = optim.Adam(upscaler.parameters(), lr=lr)
denoise_optim = optim.Adam(denoiser.parameters(), lr=lr)

mse = nn.MSELoss() # for upscaling

def train_upscaler(dataloader, epoch):
    global upscaler
    
    upscaler.train()
    up_running_loss = 0.0
    
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for data in loop:
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        '''
            Upscaler
        '''
        upoptim.zero_grad()

        outputs: torch.Tensor = upscaler(image_data)
        loss = mse(outputs, label)

        # backpropagation
        loss.backward()
        upoptim.step()

        up_running_loss += loss.item()

    up_final_loss = up_running_loss / len(dataloader.dataset)

    if wab:
        wandb.log({
            'upscaler training loss': up_final_loss,
        }, step=epoch)

    return up_final_loss


def train_denoiser(dataloader, epoch):
    global upscaler
    global denoiser
    
    denoiser.train()
    upscaler.eval()

    denoiser_running_loss = 0.0
    
    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for data in loop:
        image_data = data[0].to(device)
        label = data[1].to(device)
        

        outputs: torch.Tensor = upscaler(image_data)

        '''
            Denoiser
        '''
        denoise_optim.zero_grad()

        outputs: torch.Tensor = denoiser(outputs.detach())
        loss = psnr_loss(outputs, label)

        # backpropagation
        loss.backward()
        denoise_optim.step()

        denoiser_running_loss += loss.item()

    denoiser_final_loss = denoiser_running_loss / len(dataloader.dataset)

    if wab:
        wandb.log({
            'denoiser training loss': denoiser_final_loss,
        }, step=epoch)

    return denoiser_final_loss


def validate(dataloader, epoch, trainingmodel='upscaler'):
    global upscaler
    global denoiser
    
    upscaler.eval()
    denoiser.eval()

    upscaler_loss = 0.0
    upscaler_psnr = 0.0

    denoiser_loss = 0.0
    denoiser_psnr = 0.0

    with torch.no_grad():
        for lowres, highres in dataloader:
            lowres = lowres.to(device)
            highres = highres.to(device)
            
            # upscaler only
            outputs = upscaler(lowres)
            loss = mse(outputs, highres)
            upscaler_loss += loss.item()
            upscaler_psnr += psnr(highres, outputs)

            # denoiser
            denoise_out = denoiser(outputs)
            loss = psnr_loss(denoise_out, highres)
            denoiser_loss += loss.item()
            denoiser_psnr += psnr(highres, denoise_out)

        outputs: torch.Tensor = outputs.cpu()
        denoise_out: torch.Tensor = denoise_out.cpu()

        if trainingmodel != 'upscaler':
            save_image(outputs, f"./outputs/{epoch}_upscaled_denoised.png")
        
        save_image(outputs, f"./outputs/{epoch}_upscaled.png")
        save_image(lowres, f"./outputs/{epoch}_lowres.png")
        save_image(highres, f"./outputs/{epoch}_original.png")

    final_up_loss = upscaler_loss/len(dataloader.dataset)
    final_up_psnr = upscaler_psnr/int(len(val_data)/dataloader.batch_size)

    final_den_loss = denoiser_loss/len(dataloader.dataset)
    final_den_psnr = denoiser_psnr/int(len(val_data)/dataloader.batch_size)

    if wab:
        if trainingmodel == 'upscaler':
            wandb.log({
                'val upscaler loss': final_up_loss,
                'val upscaler psnr': final_up_psnr,
                
                'val upscaler images': [
                    wandb.Image(f"./outputs/{epoch}_lowres.png", caption="lowres"),
                    wandb.Image(f"./outputs/{epoch}_upscaled.png", caption="upscaled"),
                    wandb.Image(f"./outputs/{epoch}_original.png", caption="original")
                ]
            }, step=epoch)
        else:
            wandb.log({
                'val denoiser loss': final_den_loss,
                'val denoiser psnr': final_den_psnr,
                
                'val denoiser images': [
                    wandb.Image(f"./outputs/{epoch}_lowres.png", caption="lowres"),
                    wandb.Image(f"./outputs/{epoch}_upscaled.png", caption="upscaled"),
                    wandb.Image(f"./outputs/{epoch}_upscaled_denoised.png", caption="upscaled denoised"),
                    wandb.Image(f"./outputs/{epoch}_original.png", caption="original")
                ]
            }, step=epoch)

    return final_up_psnr, final_den_psnr


def main():
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)

    up_val_psnr, denoise_val_psnr = [], []
    up_best_psnr, de_best_psnr = 0.0, 0.0
    start = time.time()

    print(f"\nUsing ~{train_data.memory + val_data.memory} MB of RAM.\n")
    print('\nTraining upscaler')
    loop = tqdm(range(epochs), total=epochs, leave=True)
    for epoch in loop:
        epoch_upsaler_loss = train_upscaler(train_loader, epoch)
        epoch_upscaler_psnr, epoch_denoiser_psnr = validate(val_loader, epoch)

        loop.set_postfix({
            'Upscaler Loss': f"{epoch_upsaler_loss:.8f}",
            'Upscaler PSNR': f"{epoch_upscaler_psnr:.3f}",
        })

        up_val_psnr.append(epoch_upscaler_psnr)

        if epoch % saverate == 0:
            if np.mean(epoch_upscaler_psnr) >= up_best_psnr:
                torch.save(upscaler.state_dict(), f"./checkpoints/upscaler_best.pth")
                up_best_psnr = np.mean(epoch_upscaler_psnr)

            torch.save(upscaler.state_dict(), f"./checkpoints/upscaler_{epoch}.pth")


    print('\nTraining denoiser')
    loop = tqdm(range(epochs), total=epochs, leave=True)
    for epoch in loop:
        epoch_denoiser_loss = train_denoiser(train_loader, epoch)
        epoch_upscaler_psnr, epoch_denoiser_psnr = validate(val_loader, epoch)

        loop.set_postfix({
            'Denoiser Loss': f"{epoch_denoiser_loss:.6f}",
            'Upscaler PSNR': f"{epoch_upscaler_psnr:.3f}",
            'Denoiser PSNR': f"{epoch_denoiser_psnr:.3f}",
            'RAM': f"{train_data.memory} MB"
        })

        denoise_val_psnr.append(epoch_denoiser_psnr)

        if epoch % saverate == 0:
            if np.mean(epoch_denoiser_psnr) >= de_best_psnr:
                torch.save(denoiser.state_dict(), f"./checkpoints/denoiser_best.pth")
                de_best_psnr = np.mean(epoch_denoiser_psnr)
            torch.save(denoiser.state_dict(), f"./checkpoints/denoiser_{epoch}.pth")



    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

    if not wab:
        # psnr plots
        plt.figure(figsize=(10, 7))
        plt.plot(up_val_psnr, color='orange', label='upscaler psnr')
        plt.plot(denoise_val_psnr, color='red', label='denoiser psnr')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig('./outputs/psnr.png')
        plt.show()

    # save the model to disk
    print('Saving final model...')
    torch.save(upscaler.state_dict(), './checkpoints/upscaler_final.pth')
    torch.save(denoiser.state_dict(), './checkpoints/denoiser_final.pth')


if __name__ == "__main__":
    main()