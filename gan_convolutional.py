import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os

if not os.path.exists('./mnist/out_imgs_gan_conv'):
    os.mkdir('./mnist/out_imgs_gan_conv')

def to_img(x):
    out = 0.5 * x + 0.5 # add back the mean and variance
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100

# Image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = MNIST(root='./mnist', download=True, transform=img_transform)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), # b 32 28 28
                                nn.LeakyReLU(0.2, True),
                                nn.MaxPool2d(2, stride=2)) # b 32 14 14
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2), # b 64 14 14
                                nn.LeakyReLU(0.2, True),
                                nn.MaxPool2d(2, stride=2)) # b 64 7 7
        self.fc = nn.Sequential(nn.Linear(64*7*7, 1024),
                                nn.LeakyReLU(0.2, True),
                                nn.Linear(1024, 1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dimension, 1*56*56)
        self.batch_norm = nn.Sequential(nn.BatchNorm2d(1),
                                nn.ReLU(True))
        self.conv1 = nn.Sequential(nn.Conv2d(1, 50, 3, stride=1, padding=1), # b 50 56 56
                                    nn.BatchNorm2d(50),
                                    nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(50, 25, 3, stride=1, padding=1), # b 25 56 56
                                    nn.BatchNorm2d(25),
                                    nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(25, 1, 2, stride=2), # b 1 28 28
                                    nn.Tanh())
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

#create objects
D = Discriminator()
G = Generator()

#loss function and optimizer
lr = 0.0003
loss_function = nn.BCEWithLogitsLoss()
optimizer_d = torch.optim.Adam(D.parameters(), lr=lr)
optimizer_g = torch.optim.Adam(G.parameters(), lr=lr)

#training
for epoch in range(num_epoch):
    for batch_idx, data in enumerate(dataLoader):
        real_imgs, _ = data
        num_imgs = real_imgs.size(0)

        #calculate Discriminator loss for real image
        real_labels = torch.ones(num_imgs).unsqueeze(1) # one represents real image
        score_real = D.forward(real_imgs)
        loss_d_real = loss_function(score_real, real_labels)


        #calculate Discriminator loss for fake image
        fake_labels = torch.zeros(num_imgs).unsqueeze(1) # zero represents fake image
        noise = torch.randn(num_imgs, z_dimension)
        fake_imgs = G.forward(noise)
        score_fake = D.forward(fake_imgs)
        loss_d_fake = loss_function(score_fake, fake_labels)

        #train Discriminator
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        #train Generator to fool the Discriminator
        noise = torch.randn(num_imgs, z_dimension)
        fake_imgs = G.forward(noise)
        score = D.forward(fake_imgs)
        loss_g = loss_function(score, real_labels)
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if batch_idx % 100 == 0:
            print("Epoch:{}/{}\tloss_d:{}\tloss_g:{}\tscore_real:{}\tscore_fake:{}".format(epoch, num_epoch, loss_d, loss_g, torch.mean(torch.sigmoid(score_real)), torch.mean(torch.sigmoid(score_fake))))

    #save real images for reference in first epoch
    if epoch == 0:
        real_images = to_img(real_imgs.data)
        save_image(real_images, './mnist/out_imgs_gan_conv/real_images.png')
    #save fake images at end of each epoch
    fake_images = to_img(fake_imgs.data)
    save_image(fake_images, './mnist/out_imgs_gan_conv/fake_images_epoch{}.png'.format(epoch))

#save trained models
torch.save(D.state_dict(), './gan_conv_discriminator.pth')
torch.save(G.state_dict(), './gan_conv_generator.pth')
