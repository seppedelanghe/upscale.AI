import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, size: int = 33):
        super().__init__()

        self.size = size - 1
    
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=2, padding=2, padding_mode='replicate')

        self.tran1 = nn.ConvTranspose2d(1, 32, 1, stride=2)
        self.conv_final = nn.Conv2d(32, 1, kernel_size=2, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.tran1(x)
        x = self.conv_final(x)
        return x