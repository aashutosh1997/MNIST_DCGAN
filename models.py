import numpy as np
import torch
import torch.nn as nn 

class Generator(nn.Module):
    class UpBlock(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=4, stride=2, padding=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(True)
            )
        def forward(self, x):
            return self.net(x)
            
    def __init__(self, nc=1):
        super().__init__()
        self.up1 = self.UpBlock(128, 128, stride=1, padding=0)
        self.up1.apply(weights_init)
        self.up2 = self.UpBlock(128, 64)
        self.up2.apply(weights_init)
        self.up3 = self.UpBlock(64,32)
        self.up3.apply(weights_init)
        self.up4 = self.UpBlock(32,16)
        self.up4.apply(weights_init)
        self.up5 = nn.ConvTranspose2d(16,nc,kernel_size=4,stride=2,padding=1,bias=False)
        self.up5.apply(weights_init)
        self.activ = nn.Tanh()

        
    def forward(self,z):
        #shape [batch_size, latent_space_dims=128, 1, 1]
        h = self.up1(z)
        #shape [batch_size, 128, 4, 4]
        h = self.up2(h)
        #shape [batch_size, 64, 8, 8]
        h = self.up3(h)
        #shape [batch_size, 32, 16, 16]
        h = self.up4(h)
        #shape [batch_size, 16, 32, 32]
        h = self.up5(h)
        #shape [batch_size, 1, 64, 64]
        h = self.activ(h)
        return h


class Discriminator(nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, n_input, n_output, stride=1, kernel_size=4, padding=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(n_input, n_output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.LeakyReLU(0.2, inplace=True)
            )
        def forward(self, x):
            return self.net(x)
            
    def __init__(self, nc=1):
        super().__init__()
        self.down1 = self.DownBlock(nc, 16, stride=2)
        self.down1.apply(weights_init)
        self.down3 = self.DownBlock(16, 32, 2)
        self.down3.apply(weights_init)
        self.down4 = self.DownBlock(32, 64, 2)
        self.down4.apply(weights_init)
        self.down5 = self.DownBlock(64, 128, 2)
        self.down5.apply(weights_init)
        self.down7 = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0)
        self.down7.apply(weights_init)
    
    def forward(self, x):
        #size [batch_size, 1, 64, 64]
        h = self.down1(x)
        #size [batch_size, 16, 32, 32]
        h = self.down3(h)
        #size [batch_size, 32, 16, 16]
        h = self.down4(h)
        #size [batch_size, 64, 8, 8]
        h = self.down5(h)
        #size [batch_size, 128, 4, 4]
        h = self.down7(h)
        #size [batch_size, 128, 1, 1]
        return torch.sigmoid(h).squeeze()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

if __name__ == '__main__':
    model = Generator(3)
    z = torch.randn([32,128,1,1])
    print(model(z).shape)
    
    discriminator = Discriminator()
    x = torch.randn([64, 1, 64, 64])
    print(discriminator(x).shape)