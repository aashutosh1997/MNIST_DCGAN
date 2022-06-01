import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb

from models import Generator, Discriminator

batch_size = 128
latent_dims = 128
train_logger = tb.SummaryWriter('logs', flush_secs=1)
transform=Compose([ToTensor(),
                   Normalize((0.5,), (0.5,))])           
dataset = MNIST('data/', download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
#dataset = MNIST('data/',train=False, transform=transform)
#test_loader = DataLoader(dataset, batch_size=64)   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
discriminator = Discriminator().to(device) 
loss = nn.BCEWithLogitsLoss()
optimizerG = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.9, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.9, 0.999))
fig = plt.figure()
global_step = 0
for epoch in range(64):
    for idx, (images,_) in enumerate(train_loader):
        images = images.to(device)
        discriminator.zero_grad()
        batch_size = images.shape[0]
        labels = torch.ones((batch_size,), device=device)
        outD = discriminator(images)
        errD = loss(outD, labels)
        errD.backward()
        Dx = outD.mean().item()
        
        z = torch.randn([batch_size, latent_dims, 1,1], device=device)
        gen = generator(z)
        labels.zero_()
        outF = discriminator(gen.detach())
        errF = loss(outF, labels)
        errF.backward()
        Dz1 = outF.mean().item()
        err = errD + errF
        optimizerD.step()
        
        generator.zero_grad()
        labels = torch.ones((batch_size,), device=device)
        outD2 = discriminator(gen)
        errG = loss(outD2, labels)
        errG.backward()
        Dz2 = outD2.mean().item()
        optimizerG.step()
        if epoch % 8 == 0 and idx == 0:
            for i in range(8):
                z =  torch.randn([1,latent_dims,1,1], device=device)
                image = generator(z)  
                ax = fig.add_subplot(8,8,8*(epoch//8) + i + 1)
                ax.imshow(image.detach().cpu().numpy().reshape(28,28),cmap='gray')
                ax.axis('off')
        
        train_logger.add_scalar('discriminator_loss', errD.item(), global_step=global_step)
        train_logger.add_scalar('generator_loss', errG.item(), global_step=global_step)
        train_logger.add_scalar('Dx', Dx, global_step=global_step)
        train_logger.add_scalar('DGz', Dz1/Dz2, global_step=global_step)
        global_step += 1
    if epoch%8 == 0:
        print("Epoch:[%d/64]"%(epoch))
plt.show()