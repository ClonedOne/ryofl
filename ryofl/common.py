"""
Constants that should be accessible from any other file
"""

import os

# This is the directory where the dataset will be saved.
# Change this to a path with sufficient disk space.
data_dir = '/home/gio/data/'

# Image datasets
image_data_dir = os.path.join(data_dir, 'images')
# FEMNIST clients data
femnist_clients_dir = os.path.join(image_data_dir, 'femnist_clients')
# CIFAR100 clients data
cifar100_clients_dir = os.path.join(image_data_dir, 'cifar100_clients')

