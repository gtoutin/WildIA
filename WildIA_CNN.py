# class definition for the WildIA CNN
import torch
from torch import nn


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
        return self.output(x.view(x.shape[0], -1))  # reshape tensor to 1d and get output