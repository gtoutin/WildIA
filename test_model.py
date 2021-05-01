# this is where model testing will occur.

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
from WildIA_CNN import Net

TESTDIR = './test'  # folder of training and validation data
PATH = './WildIA_net.pth'  # location of saved model
num_classes = 4     # number of species in the model
batch_size = 4  # how many images at a time

# define a CNN. it's an instance of the Net class defined in WildIA_CNN.py.
net = Net(num_classes)

# load a model if it exists
if os.path.isfile(PATH):
    print('Loading previous model...')
    net.load_state_dict(torch.load(PATH))
    print('Model loaded.\n')

# normalize PILImage outputs [0 1] to [-1 1] tensors required format and resize
transform = transforms.Compose(
    [transforms.Resize((224,224)),  # need to resize test images
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


print('Loading testing dataset...')
# load the datasets using ImageFolder
testset = torchvision.datasets.ImageFolder(root=TESTDIR, transform=transform)

# create dictionary to easily display predicted classes
birdclasses_to_idx = testset.class_to_idx
idx_to_class = {value:key for key, value in birdclasses_to_idx.items()}
birdclasses = tuple(birdclasses_to_idx.keys())

# get a dataloader to easily navigate thru the dataset for training
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

dataiter = iter(testloader)
images, labels = dataiter.next()    # get images and labels from test data iterator
print('Dataset loaded.\n')


# calculate test accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        probs, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

#        print('Predicted: ', ' '.join('%5s' % birdclasses[predicted[j]]
#                              for j in range(len(predicted))))

print('Accuracy of the network on the test images: %d %% \n' % (
    100 * correct / total))



# print accuracy by class

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in birdclasses}
total_pred = {classname: 0 for classname in birdclasses}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[birdclasses[label]] += 1
            total_pred[birdclasses[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))