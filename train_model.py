# this is where the model training will occur.

TRAINDIR = './train'  # folder of training and validation data
PATH = './WildIA_net.pth'  # location of saved model
num_classes = 4     # number of species in the model

# import stuff
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# define what our classifier network will look like. necessary for both loading our model again and starting a model for the first time.
class Net(nn.Module):  # Thank you Alex!
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5),  # 1st convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),  # 2nd convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),  # 3rd convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.output = nn.Sequential(
            nn.Linear(128, num_classes)  # get output from convolutions
        )
    def forward(self, x):
        x = self.backbone(x)  # get convolutions
        return self.output(x.view(x.shape[0], -1))  # reshape tensor to 1d and get ouput


batch_size = 4  # how many images at a time

# define a CNN. it's an instance of the Net class defined above.
net = Net(num_classes)

# with torch.no_grad():
#     output = net(torch.randn(4, 3, 224, 244))
# print(output)

# load a model if it exists
if os.path.isfile(PATH):
    print('Loading previous model...')
    net.load_state_dict(torch.load(PATH))
    print('Model loaded.\n')


# normalize PILImage outputs [0 1] to [-1 1] tensors required format
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


print('Loading dataset...')
# load the datasets using ImageFolder
trainset = torchvision.datasets.ImageFolder(root=TRAINDIR, transform=transform)
# get a dataloader to easily navigate thru the dataset for training
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
print('Dataset loaded.\n')

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # good criterion for classifiers
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # stochastic gradient descent

print('Starting training...\n')
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
        # print(inputs.size())
        # print(outputs.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training!')


print('Saving progress...')
# save progress
torch.save(net.state_dict(), PATH)
print('Model saved.')