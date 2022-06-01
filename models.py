import numpy as np
import torch
import torch.nn as nn 

class Generator(nn.Module):
    class UpConvBlock(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=2, stride=2, out_padding=0):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, stride=stride, output_padding=out_padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU()
            )
        def forward(self, x):
            return self.net(x)
            
    def __init__(self):
        super().__init__()
        self.upconv1 = self.UpConvBlock(128, 64, out_padding=1)
        self.upconv2 = self.UpConvBlock(64, 32, out_padding=1)
        self.upconv3 = self.UpConvBlock(32,16)
        self.upconv4 = self.UpConvBlock(16,1)
        
    def forward(self,z):
        #shape [batch_size, latent_space_dims=128, 1, 1]
        h = self.upconv1(z)
        #shape [batch_size, 64, 3, 3]
        h = self.upconv2(h)
        #shape [batch_size, 32, 7, 7]
        h = self.upconv3(h)
        #shape [batch_size, 16, 14, 14]
        h = self.upconv4(h)
        #shape [batch_size, 1, 28, 28]
        return h

class Discriminator(nn.Module):
    class DownConvBlock(nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(n_input, n_output, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(n_output),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(n_output, n_output, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(n_output),
                nn.LeakyReLU(0.2, inplace=True)
            )
        def forward(self, x):
            return self.net(x)
            
    def __init__(self):
        super().__init__()
        self.conv1 = self.DownConvBlock(1, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = self.DownConvBlock(16, 32)
        self.conv4 = self.DownConvBlock(32, 64)
        self.conv5 = self.DownConvBlock(64, 128)
        self.pool6 = nn.AvgPool2d(kernel_size=3, stride=1)
        self.lin7 = nn.Linear(128, 1)
    
    def forward(self, x):
        #size [batch_size, 1, 28, 28]
        h = self.conv1(x)
        #size [batch_size, 16, 28, 28]
        h = self.pool2(h)
        #size [batch_size, 16, 14, 14]
        h = self.conv3(h)
        #size [batch_size, 32, 14, 14]
        h = self.pool2(h)
        #size [batch_size, 32, 7, 7]
        h = self.conv4(h)
        #size [batch_size, 64, 7, 7]
        h = self.pool2(h)
        #size [batch_size, 64, 3, 3]
        h = self.conv5(h)
        #size [batch_size, 128, 3, 3]
        h = self.pool6(h)
        #size [batch_size, 128, 1, 1]
        h = h.squeeze()
        #size [batch_size, 128]
        h = self.lin7(h)
        #size [batch_size, 1]
        return h.squeeze()


if __name__ == '__main__':
    model = Generator()
    z = torch.randn([32,128,1,1])
    print(model(z).shape)
    
    discriminator = Discriminator()
    x = torch.randn([64, 1, 28, 28])
    print(discriminator(x).shape)