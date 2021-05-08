# class definition for the WildIA CNN
import pathlib

import albumentations as alb
import cv2
import torch
from torch.utils import data
from torch import nn


class ParrotDataset(data.Dataset):
    def __init__(self, inputpath, CLASSES, split=''):
        self.augmentations = alb.Compose(
            [
                alb.Resize(224, 224),
                alb.Flip(),
                alb.RandomBrightnessContrast(),
                alb.HueSaturationValue(),
                alb.RandomGamma(),
                alb.Normalize(),
            ]
        )
        self.CLASSES = CLASSES

        _DATA_DIR = pathlib.Path(inputpath)
        
        data_dir = _DATA_DIR / split
        assert data_dir.is_dir()
        self.imgs = [img for img in list(data_dir.rglob("*")) if img.is_file()]

    def __len__(self):
        return len(self.imgs)

    def __str__(self):
        return f"{len(self.imgs)} images."

    def __getitem__(self, idx: int):
        class_name = self.imgs[idx].parent.name
        image = cv2.imread(str(self.imgs[idx]))
        assert image is not None, self.imgs[idx]
        image = self.augmentations(image=image)["image"]
        return torch.Tensor(image), self.CLASSES.index(class_name)


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
