import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pdb

transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize([0.5], [0.5]) #why always 0.5 mean subtracted and 0.5 standard deviation divided ?
])

def to_img(x):
    x = 0.5 * x + 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28) # x.size(0) for the batch_size
    return x

max_epochs = 100
batch_size = 128
lr = 1e-3

dataset = MNIST(root='./mnist', download=True, transform=transform)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400,20)
        self.fc22 = nn.Linear(400,20)
        self.fc3 = nn.Linear(20,400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        x = self.fc1(x)
        h = F.relu(x)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        return std.mul(eps).add_(mu)

    def decode(self, z):
        x = self.fc3(z)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar


model = VAE()
reconstruction_loss_function = nn.MSELoss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=lr) # no weight decay ?

def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = reconstruction_loss_function(recon_imgs, imgs)
    KLdiv_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLdiv_loss =  torch.sum(KLdiv_element).mul_(-0.5)
    return reconstruction_loss + KLdiv_loss


for epoch in range(max_epochs):
    model.train() #put model in train mode - why was this not used in other autoencoder examples ?
    train_loss = 0
    for batch_idx, data in enumerate(dataLoader):
        imgs, _ = data
        imgs = imgs.view(imgs.size(0), -1) #flatten all images in the batch
        recon_imgs, mu, logvar = model.forward(imgs)
        loss = loss_function(recon_imgs, imgs, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(imgs),
                len(dataLoader.dataset), 100. * batch_idx / len(dataLoader),
                loss.data / len(imgs)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataLoader.dataset)))
    if epoch % 10 == 0:
        save = to_img(recon_imgs.data)
        save_image(save, './mnist/out_imgs_variational_autoencoder/{}.png'.format(epoch))

#save model
torch.save(model.state_dict(), './variational_autoencoder.pth')
