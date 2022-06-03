import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.utils.tensorboard as tb

from models import Generator, Discriminator, weights_init

batch_size = 64
latent_dims = 128
train_logger = tb.SummaryWriter('logs', flush_secs=1)
transform=Compose([Resize(64),
                    ToTensor(),
                   Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])           
dataset = CIFAR10('data/', download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(3).to(device)
discriminator = Discriminator(3).to(device)
loss = nn.BCELoss()
optimizerG = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
one = 1
zero = 0
test_vector = torch.randn([batch_size, latent_dims, 1,1], device=device)
# fig = plt.figure()
global_step = 0
for epoch in range(128):
    for idx, (images,_) in enumerate(train_loader):
        images = images.to(device)
        discriminator.zero_grad()
        batch_size = images.shape[0]
        labels = torch.full((batch_size,), one, dtype=images.dtype, device=device)
        outD = discriminator(images)
        errD = loss(outD, labels)
        errD.backward()
        Dx = outD.mean().item()
        
        z = torch.randn([batch_size, latent_dims, 1,1], device=device)
        gen = generator(z)
        labels.fill_(zero)
        outF = discriminator(gen.detach())
        errF = loss(outF, labels)
        errF.backward()
        Dz1 = outF.mean().item()
        err = errD + errF
        optimizerD.step()
        
        generator.zero_grad()
        labels.fill_(one)
        outD2 = discriminator(gen)
        errG = loss(outD2, labels)
        errG.backward()
        Dz2 = outD2.mean().item()
        optimizerG.step()
        # if epoch % 8 == 0 and idx == 0:
        #     for i in range(8):
        #         z_test = torch.randn([1, 128, 1, 1]).to(device)
        #         gen_test = generator(z_test) 
        #         ax = fig.add_subplot(8,8,8*(epoch//8) + i + 1)
        #         ax.imshow(gen_test.detach().cpu().numpy().reshape(64,64),cmap='gray')
        #         ax.axis('off')
        if idx==0:
            image = generator(test_vector)
            vutils.save_image(image.detach(),
                    'output/fake_samples_epoch_%03d.png' % (epoch),
                    normalize=True)
        
        train_logger.add_scalar('discriminator_loss', errD.item(), global_step=global_step)
        train_logger.add_scalar('generator_loss', errG.item(), global_step=global_step)
        train_logger.add_scalar('Dx', Dx, global_step=global_step)
        train_logger.add_scalar('DGz', Dz1/Dz2, global_step=global_step)
        global_step += 1
    if epoch%8 == 0:
        print("Epoch:[%d/128]"%(epoch))
# plt.show()