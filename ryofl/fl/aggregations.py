"""
This module contains code to perform different forms of federated
aggregation.
"""

import copy

from torch import Tensor


def aggregate(
    client_weights: list,
    strategy: str,
    params: dict = None
) -> dict:
    """ Aggregate client updates with specified function

    The global model state dictionary will always be at position 0 of
    client_weights.

    Args:
        client_weights (list): list of client weight dictionaries
        strategy: identifier of the strategy to use
        params: optional dictionary of parameters

    Returns:
        dict: state dicto for averaged model

    Raises:
        NotImplementedError: str
    """

    if strategy == 'averaging':
        return federated_averaging(client_weights)

    elif strategy == 'scaled_averaging':
        return scaled_federated_averaging(client_weights, params)

    else:
        raise NotImplementedError('Strategy {} not supported'.format(strategy))



def federated_averaging(client_weights: list) -> dict:
    """ Perform federated averaging over the clients' weights

    Client weights are dictionaries obtained with model.state_dict()
    https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

    Args:
        client_weights (list): list of client weight dictionaries

    Returns:
        dict: state dict for averaged model
    """

    # Handle possible issues with client updates list
    if not client_weights:
        raise RuntimeError('Empty client_weights list recieved')

    num_updates = len(client_weights)

    if num_updates == 1:
        print('WARNING: RECEIVED SINGLE CLIENT UPDATE')

    temp_weight = copy.deepcopy(client_weights[0])

    # Accumulate weights
    for client_dict in client_weights:
        for layer_name, layer_weights in client_dict.items():
            temp_weight[layer_name] += layer_weights

    # Average them
    for layer_name, layer_weights in temp_weight.items():
        temp_weight[layer_name] = layer_weights / num_updates

    return temp_weight


def scaled_federated_averaging(
    client_weights: list,
    params: dict = None
) -> dict:
    """ Perform federated averaging over the clients' weights

    Client weights are dictionaries obtained with model.state_dict()
    https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

    parameters in `params`:
        alpha (float): learning rate for the aggregation

    Args:
        client_weights (list): list of client weight dictionaries
        params (dict): dictionary of parameters

    Returns:
        dict: state dict for averaged model
    """

    # Handle possible issues with client updates list
    if not client_weights:
        raise RuntimeError('Empty client_weights list recieved')

    if params:
        alpha = params.get('alpha', 0.3)
    else:
        alpha = 0.3

    num_updates = len(client_weights)

    if num_updates == 1:
        print('WARNING: RECEIVED SINGLE CLIENT UPDATE')

    glob_weight = copy.deepcopy(client_weights[0])
    temp_weight = copy.deepcopy(client_weights[1])

    # Accumulate weights
    for client_dict in client_weights[1:]:
        for layer_name, layer_weights in client_dict.items():
            temp_weight[layer_name] += layer_weights

    # Average them
    for layer_name, layer_weights in temp_weight.items():
        temp_weight[layer_name] = \
            (1 - alpha) * glob_weight[layer_name] \
            + alpha * (layer_weights / num_updates)

    return temp_weight
