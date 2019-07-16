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

if not os.path.exists('./mnist/out_imgs_gan'):
    os.mkdir('./mnist/out_imgs_gan')

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
        self.net = nn.Sequential(nn.Linear(784, 256),
                                nn.LeakyReLU(0.2),
                                nn.Linear(256, 256),
                                nn.LeakyReLU(0.2),
                                nn.Linear(256, 1))
    def forward(self, x):
        x = self.net(x)
        return x

#generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(nn.Linear(z_dimension, 256),
                                nn.ReLU(True),
                                nn.Linear(256,256),
                                nn.ReLU(True),
                                nn.Linear(256,784),
                                nn.Tanh())
    def forward(self, x):
        x = self.net(x)
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
        #print("real_imgs.size():{}".format(real_imgs.size()))
        real_imgs_flatten = real_imgs.view(num_imgs, -1) # flatten images to 1D vectors
        #print("real_imgs_flatten.size():{}".format(real_imgs_flatten.size()))
        score_real = D.forward(real_imgs_flatten)
        loss_d_real = loss_function(score_real, real_labels)


        #calculate Discriminator loss for fake image
        fake_labels = torch.zeros(num_imgs).unsqueeze(1) # zero represents fake image
        noise = torch.randn(num_imgs, z_dimension)
        fake_imgs_vector = G.forward(noise)
        score_fake = D.forward(fake_imgs_vector)
        loss_d_fake = loss_function(score_fake, fake_labels)

        #train Discriminator
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        #train Generator to fool the Discriminator
        noise = torch.randn(num_imgs, z_dimension)
        fake_imgs_vector = G.forward(noise)
        score = D.forward(fake_imgs_vector)
        loss_g = loss_function(score, real_labels)
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if batch_idx % 100 == 0:
            print("Epoch:{}/{}\tloss_d:{}\tloss_g:{}\tscore_real:{}\tscore_fake:{}".format(epoch, num_epoch, loss_d, loss_g, torch.mean(torch.sigmoid(score_real)), torch.mean(torch.sigmoid(score_fake))))

    #save real images for reference in first epoch
    if epoch == 0:
        real_images = to_img(real_imgs_flatten.data)
        save_image(real_images, './mnist/out_imgs_gan/real_images.png')
    #save fake images at end of each epoch
    fake_images = to_img(fake_imgs_vector)
    save_image(fake_images, './mnist/out_imgs_gan/fake_images_epoch{}.png'.format(epoch))

#save trained models
torch.save(D.state_dict(), './gan_discriminator.pth')
torch.save(G.state_dict(), './gan_generator.pth')
