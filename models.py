import torch
import torch.nn as nn
import torch.nn.functional as F

nz = 100 
ngf = 64
nc=3

class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels=100, 
            out_channels=3,
            kernel_size=64,
            stride=3,
            padding=0,
            bias=False
        ) 
        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(False),
        #     # state size. ``(ngf*8) x 4 x 4``
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(False),
        #     # state size. ``(ngf*4) x 8 x 8``
        #     nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(False),
        #     # state size. ``(ngf*2) x 16 x 16``
        #     nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(False),
        #     # state size. ``(ngf) x 32 x 32``
        #     nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. ``(nc) x 64 x 64``
        # )

    def forward(self, z):
        print("gen", z.shape)
        # z = torch.reshape(z, (4, 4, 1024))
        z = self.conv(z)
        print("gen", z.shape)
        return z
        # z = self.main(z)
        # print("out:", out.shape)
        # return z
    
class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 64*4)
        self.fc2 = nn.Linear(64*4, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        # throw a sigmoid in here

    def forward(self, image):
        # print("before proc", image.shape)
        image = self.pool(F.relu(self.conv1(image)))
        image = self.pool(F.relu(self.conv2(image)))
        image = self.pool(F.relu(self.conv3(image)))
        image = self.pool(F.relu(self.conv4(image)))
        # print("after pool", image.shape)
        image = torch.reshape(image, (-1, 64*4*4))
        image = F.relu(self.fc1(image))
        image = F.relu(self.fc2(image))
        image = F.relu(self.fc3(image))
        image = F.relu(self.fc4(image))
        # no relu on the last layer
        image = torch.sigmoid(self.fc5(image))
        return image