"""
This module contains code to perform different forms of federated
aggregation.
"""

import copy


def federated_averaging(client_weights: list):
    """ Perform federated averaging over the clients' weights

    Client weights are dictionaries obtained with model.state_dict()
    https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

    Args:
        client_weights (list): list of client weights

    Returns:
        dict: state dict for averaged model
    """

    num_updates = len(client_weights)

    # Handle possible issues with client updates list
    if not client_weights:
        raise RuntimeError('Empty client_weights list recieved')
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
