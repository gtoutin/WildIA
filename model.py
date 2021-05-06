# class definition for the WildIA CNN
import torch
from torch import nn


# define what our classifier network will look like. necessary for both loading our model again and starting a model for the first time.
class Net(nn.Module):  # Thank you Alex!
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5),  # 1st convolutional layer
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),  # 2nd convolutional layer
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),  # 3rd convolutional layer
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),  # 3rd convolutional layer
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Sequential(nn.Linear(256, num_classes, bias=True))

    def forward(self, x):
        x = self.backbone(x)
        # reshape tensor to B,D and get output
        return self.output(x.view(x.shape[0], -1))
