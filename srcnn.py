import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, size: int = 33, gray: bool = False):
        super().__init__()

        self.size = size - 1
        self.colorspace = 1 if gray else 3
    
        self.conv1 = nn.Conv2d(self.colorspace, 64, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, self.colorspace, kernel_size=2, padding=2, padding_mode='replicate')

        self.tran1 = nn.ConvTranspose2d(self.colorspace, 32, self.colorspace, stride=2)
        self.conv_final = nn.Conv2d(32, self.colorspace, kernel_size=1+self.colorspace, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.tran1(x)
        x = self.conv_final(x)
        return x