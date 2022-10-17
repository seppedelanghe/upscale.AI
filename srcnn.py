import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, size: int = 33):
        super().__init__()

        self.size = size - 1
    
        self.conv1 = nn.Conv2d(1, self.size * 3, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(self.size * 3, self.size, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(self.size, 1, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x