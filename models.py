import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels=100, 
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ) 

    def forward(self, z):
        z = self.conv(z)
        return z
    
class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*36*36, 32)
        self.fc2 = nn.Linear(32, 2)
        # throw a sigmoid in here

    def forward(self, image):
        # print(image.shape)
        image = F.relu(self.conv1(image))
        image = self.pool(image)
        image = torch.reshape(image, (-1, 8*36*36))
        image = F.relu(self.fc1(image))
        image = self.fc2(image)
        return image