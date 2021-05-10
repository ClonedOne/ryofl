"""
Generic utilities related to managing the models
"""

from typing import Any

import torch

from ryofl.models import cnn


def build_model(model_id: str, channels: int, classes: int):
    """ Build a model object

    Args:
        model_id (str): identifier of the model to build
        channels (int): number of channels of the data tensor
        classes (int): number of classes in the dataset

    Raises:
        NotImplementedError: model_id should correspond to classes in this module

    Returns:
        pytorch model object
    """

    if model_id == 'cnn_small':
        return cnn.SmallCNN(channels=channels, classes=classes)

    if model_id == 'cnn_alex':
        return cnn.AlexnetCNN(channels=channels, classes=classes)

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))


def save_model(model: Any, pth: str):
    """ Save the state of a model on disk

    Assumes model is a pytorch model object

    Args:
        model: pytorch model
        pth: path where to save model state
    """

    torch.save(model.state_dict(), pth)


def load_model(model: Any, pth: str):
    """ Load model state from file

    Assumes model is pytorch object

    Args:
        model: pytorch model
        pth: path of saved state dictionary
    """

    model.load_state_dict(torch.load(pth))
