# this is where the model training will occur.

TRAINDIR = './data'  # folder of training and validation data
PATH = './WildIA_net.pth'  # location of saved model

# import stuff
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# define what our classifier network will look like. necessary for both loading our model again and starting a model for the first time.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = 4  # how many images at a time


# load a model if it exists


# load the datasets using ImageFolder
trainset = torchvision.datasets.ImageFolder(root=TRAINDIR)
# get a dataloader to easily navigate thru the dataset for training
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

