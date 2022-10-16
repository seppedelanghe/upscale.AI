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

# learning parameters
image_limit = 500
batch_size = 64 # batch size, reduce if facing OOM error
epochs = 100 # number of epochs to train the SRCNN model for
lr = 0.001 # the learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_size = 65 # in pixels => always square (65, 65)

# load data
train_data = SRCNNImageDataset('/home/seppe/Documents/merged/images/', (img_size, img_size), limit=image_limit)
val_data = SRCNNImageDataset('/home/seppe/Documents/football/test/', (img_size, img_size))

# train and validation loaders
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

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

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/loss.png')

    # psnr plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('./outputs/psnr.png')

    # save the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), './model.pth')


if __name__ == "__main__":
    main()