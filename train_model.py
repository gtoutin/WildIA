# this is where the model training will occur.

TRAINDIR = './data'  # folder of training and validation data
PATH = './WildIA_net.pth'  # location of saved model

# import stuff
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

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

# define a CNN. it's an instance of the Net class defined above.
net = Net()

# load a model if it exists
if os.path.isfile(PATH):
    print('Loading previous model...')
    net.load_state_dict(torch.load(PATH))
    print('Model loaded.')


print('Loading dataset...')
# load the datasets using ImageFolder
trainset = torchvision.datasets.ImageFolder(root=TRAINDIR)
# get a dataloader to easily navigate thru the dataset for training
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
print('Dataset loaded.')

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # good criterion for classifiers
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # stochastic gradient descent

print('Starting training...')
# train the net ---------------------------------------------
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training!')


print('Saving progress...')
# save progress
torch.save(net.state_dict(), PATH)
print('Model saved.')