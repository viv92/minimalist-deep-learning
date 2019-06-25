import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#set of trnsforms to normalize inputs
transform = transforms.Compose([
            transforms.ToTensor(), #transforms image to torch tensor 
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #normalize by subtracting mean 0.5 and dividing by standard deviation 0.5
])

#training data
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
trainLoader = DataLoader(trainset, batch_size=4, shuffle=True)
#testing data
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
testLoader = DataLoader(testset, batch_size=4, shuffle=False)
#classnames
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#helper function to display an image
def imshow(img):
    img = img * 0.5 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

#neural net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5) #filter_depth=3, filter_num=10, filter_size=5x5
        self.pool1 = nn.MaxPool2d(2, 2) #output dim of conv1 = 10x28x28. Output of pool1 = 10x14x14
        self.conv2 = nn.Conv2d(10, 20, 5) #filter_depth=10, filter_num=20, filter_size=5x5
        self.pool2 = nn.MaxPool2d(2,2) #output dim of conv2 = 20x10x10. Output of pool2 = 20x5x5
        self.fc1 = nn.Linear(20*5*5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 20*5*5) #flatten output of pool2
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

net = Net()

max_epochs = 3
lr = 1e-1
loss_function = nn.CrossEntropyLoss() #since multiclass classification
optimizer = optim.SGD(net.parameters(), lr=lr)

#train the model
for epoch in range(max_epochs):
    print("epoch: {}".format(epoch))
    train_iterator = iter(trainLoader)
    for batch_idx, (inps,lbs) in enumerate(train_iterator):
        net.zero_grad()
        preds = net.forward(inps) #probabilities
        loss = loss_function(preds, lbs)
        loss.backward()
        optimizer.step()

#test the model
test_iter = iter(testLoader)
test_imgs, test_labels = test_iter.next() #get a test sample
predictions = net.forward(test_imgs) #probabilities
_, predicted_class_indices = torch.max(predictions.data, 1)
print("Predictions:")
for i in range(4):
    print("{} ".format(classes[predicted_class_indices[i]]))
print("Groundtruths:")
for i in range(4):
    print("{} ".format(classes[test_labels[i]]))

#calculate test accuracy
total = 0
correct = 0
for data in testLoader:
    imgs, labels = data
    predicted_probabilities = net.forward(imgs)
    _, predicted_indices = torch.max(predicted_probabilities.data, 1)
    total += labels.size(0)
    correct += (predicted_indices == labels).sum()
print("Accuracy: {}".format((100*correct)/total))
