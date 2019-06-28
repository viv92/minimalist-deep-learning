import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image

batch_size = 100
lr = 1e-3
max_epochs = 20

train_dataset = MNIST(root='./mnist2/mnist_train', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST(root='./mnist2/mnist_test', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True) # changes input schema from (seq_length, batch_size, input_size) to (batch_size, seq_length, input_size)
        self.classifier = nn.Linear(hidden_dim, n_class)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        #print("out: {}".format(out[1]))
        out = out[:, -1, :] #flatten
        out = self.classifier(out)
        return out

model = RNN(28, 128, 2, 10) # in_dim=28, out_dim=10 since MNIST
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(max_epochs):
    print("epoch:{}".format(epoch+1))
    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(train_loader):
        imgs, labels = data
        b, c, h, w = imgs.size()
        assert c == 1, 'channel must be 1'
        imgs = imgs.squeeze(1) #flatten
        out = model.forward(imgs)
        loss = loss_function(out, labels)
        running_loss += loss.data * labels.size(0) #why multiply loss with batch size ?
        _, pred = torch.max(out, 1)
        num_correct = (pred == labels).sum()
        running_accuracy += num_correct.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i-1) % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, max_epochs, running_loss / float(batch_size * i),
                running_accuracy / float(batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / float(len(train_dataset)), running_accuracy / float(len(
            train_dataset))))

    #evaluate model
    model.eval()
    eval_loss = 0.0
    eval_accuracy = 0.0
    for i, data in enumerate(test_loader):
        imgs, labels = data
        b, c, h, w = imgs.size()
        assert c == 1, 'channel must be 1'
        imgs.squeeze_(1)
        out = model.forward(imgs)
        loss = loss_function(out, labels)
        eval_loss += loss.data
        _, pred = torch.max(out, 1)
        eval_accuracy += (pred == labels).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / float(len(
        test_dataset)), eval_accuracy / float(len(test_dataset))))
    print()

#save model
torch.save(model.state_dict(), './rnn.pth')
