"""
This module defines an example convolutional network.
"""

import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
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
        super(BaseCNN, self).__init__()

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
