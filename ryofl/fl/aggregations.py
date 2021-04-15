"""
This module contains code to perform different forms of federated
aggregation.
"""

import copy

from torch import Tensor


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
