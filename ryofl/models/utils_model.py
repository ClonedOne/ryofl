"""
Generic utilities related to managing the models
"""

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

    if model_id == 'cnn':
        return cnn.BaseCNN(channels=channels, classes=classes)

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))
