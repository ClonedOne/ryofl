"""
Constants that should be accessible from any other file
"""

import os

# Number of processors to be used when possible
processors = 4

# This is the directory where the dataset will be saved.
# Change this to a path with sufficient disk space.
data_dir = '/media/storage/data/federated_data'

# This is the directory where the participants configuration
# files will be saved.
cfg_dir = 'configs/'

# Image datasets
image_data_dir = os.path.join(data_dir, 'images')
# FEMNIST clients data
femnist_clients_dir = os.path.join(image_data_dir, 'femnist_clients')
# CIFAR100 clients data
cifar100_clients_dir = os.path.join(image_data_dir, 'cifar100_clients')

# Server constants
SRV_ID = 0
SRV_HOST = '127.0.0.1'
SRV_PORT = 9999

