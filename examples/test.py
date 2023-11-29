# Commented out IPython magic to ensure Python compatibility.
#   %load_ext tensorboard
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
import time
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from argparse import ArgumentParser
from torchvision import transforms as tt

# from torchsummary import summary

# set manual seed for reproducibility
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""## Partitioning the Data (IID and non-IID)"""
import sys
sys.path.append('..')  # Adds the parent directory to the Python path1
from my_utils.utils_model import MyGroupNorm
from my_utils.utils_train import training, testing
from my_utils.utils_dataloader import non_iid_partition, iid_partition
from my_utils.utils_dataloader import get_ds_cifar10, Non_iid, get_ds_mnist
from my_utils.utils_reading_disks import get_dict_from_yaml

if __name__ == '__main__':
    path_config = '../configs/5_cifar_10_sra_fl_non_iid.yaml'
    configs = get_dict_from_yaml(path=path_config)
    print(configs)
    # create transforms
    # We will just convert to tensor and normalize since no special transforms are mentioned in the paper
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = tt.Compose([tt.ToTensor(),
                                         tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                                         tt.RandomHorizontalFlip(p=0.5),
                                         tt.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(*stats)])

    cifar_data_train = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_cifar_train)
    cifar_data_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_cifar_test)

    classes = np.array(list(cifar_data_train.class_to_idx.values()))
    classes_test = np.array(list(cifar_data_test.class_to_idx.values()))
    num_classes = len(classes_test)

    criterion = nn.CrossEntropyLoss()

    # Hyperparameters_List (H) = [rounds, client_fraction, number_of_clients, number_of_training_rounds_local, local_batch_size, lr_client]

    if configs['non_iid'] == True:
        # (dataset, clients, total_shards, shards_size, num_shards_per_client):
        # alpha for the Dirichlet distribution
        data_dict = non_iid_partition(cifar_data_train, configs['num_clients'], configs['degree_non_iid'])
    else:
        data_dict = iid_partition(cifar_data_train, 100)  # Uncomment for idd_partition

    
    cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
                                  width_per_group=64, replace_stride_with_dilation=None, norm_layer=MyGroupNorm)
    
    cifar_cnn.cuda()

    trained_model = training(cifar_cnn, cifar_data_train, data_dict,
                              cifar_data_test,criterion, classes_test, False, config=configs)