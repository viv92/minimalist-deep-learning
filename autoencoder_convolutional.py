import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pdb

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) #why always 0.5 mean subtracted and 0.5 standard deviation divided ?
])

def to_img(x):
    x = 0.5 * x + 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28) # x.size(0) for the batch_size
    return x

max_epochs = 20
batch_size = 128
lr = 1e-3

dataset = MNIST(root='./mnist', download=True, transform=transform)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), # b 10 10 16
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # b 5 5 16
            nn.Conv2d(16, 8, 3, stride=2, padding=1), # b 3 3 8
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)) # b 2 2 8
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2), # b 5 5 16
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), # b 15 15 8
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), # b 28 28 1
            nn.Tanh()) # why the Tanh in decoder ?
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

avg_loss = 0
for epoch in range(max_epochs):
    for data in dataLoader:
        imgs, _ = data
        #imgs = imgs.view(imgs.size(0), -1) #flatten all images in the batch
        #pdb.set_trace()
        out_imgs = model.forward(imgs)
        loss = loss_function(out_imgs, imgs)
        avg_loss = (epoch * avg_loss + loss.data) / float(epoch+1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            pic = to_img(out_imgs.data) #but these are batch of images and not a single image ?
            #print("SIZE:{}".format(pic.size()))
            save_image(pic, "./mnist/out_imgs_convolutional_autoencoder/{}.png".format(epoch))
    print("epoch:[{}/{}], avg_loss:{:.4f}".format(epoch, max_epochs, avg_loss))

#save model
torch.save(model.state_dict(), './convolutional_autoencoder.pth')
