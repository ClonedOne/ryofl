"""
This module defines an example convolutional network.
"""

import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Basic CNN adapted from:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self, channels: int, classes: int):
        """ Network definition

        Args:
            channels (int): number of channels
            classes (int): number of output classes
        """

        self.channels = channels
        self.classes = classes
        super(SmallCNN, self).__init__()

        # Convolutions
        # #channels, 6 output channels, 5x5 convolution
        self.conv1 = nn.Conv2d(self.channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Adaptive max pooling to handle different sizes
        self.amaxpool = nn.AdaptiveMaxPool2d((5, 7))

        # Linear layers
        self.fc1 = nn.Linear(16 * 5 * 7, 400)
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, self.classes)

        # Dropout regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """ Forward pass of the network

        Args:
            x: mini-batch of data

        Returns:
            processed representation
        """

        x = self.pool(F.relu(self.conv1(x)))
        #  x = self.pool(F.relu(self.conv2(x)))
        x = self.amaxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class AlexnetCNN(nn.Module):
    """
    Alexnet CNN adapted from:
    https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/alexnet.py
    """

    def __init__(self, channels=3, classes=10):
        """ Network definition

        Args:
            channels (int): number of channels
            classes (int): number of output classes

        """

        super(AlexnetCNN, self).__init__()
        self.channels = channels
        self.classes = classes

        self.features = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Linear(256, self.classes)

    def forward(self, x):
        """ Forward pass of the network

        Args:
            x: mini-batch of data

        Returns:
            processed representation
        """

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
