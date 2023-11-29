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

    parser = ArgumentParser()
    parser.add_argument('--norm', default="bn")
    parser.add_argument('--partition', default="noniid")
    parser.add_argument('--client_number', default=100)
    parser.add_argument('--alpha_partition', default=0.5)
    parser.add_argument('--commrounds', type=int, default=150)
    parser.add_argument('--clientfr', type=float, default=0.1)
    parser.add_argument('--numclient', type=int, default=100)
    parser.add_argument('--clientepochs', type=int, default=20)
    parser.add_argument('--clientbs', type=int, default=64)
    parser.add_argument('--clientlr', type=float, default=0.0001)
    parser.add_argument('--sch_flag', default=False)

    path_config = '../configs/4_mnist_sra_fl_non_iid.yaml'
    configs = get_dict_from_yaml(path=path_config)
    print(configs)

    args = parser.parse_args()
    
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
    H = [args.commrounds, args.clientfr, args.numclient, args.clientepochs, args.clientbs, args.clientlr]

    if args.partition == 'noniid':
        # (dataset, clients, total_shards, shards_size, num_shards_per_client):
        # alpha for the Dirichlet distribution
        data_dict = non_iid_partition(cifar_data_train, args.client_number, float(args.alpha_partition))
    else:
        data_dict = iid_partition(cifar_data_train, 100)  # Uncomment for idd_partition

    if args.norm == 'gn':
        cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
                                  width_per_group=64, replace_stride_with_dilation=None, norm_layer=MyGroupNorm)
    else:
        cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
                                  width_per_group=64, replace_stride_with_dilation=None)

    cifar_cnn.cuda()

    plot_str = args.partition + '_' + args.norm + '_' + 'comm_rounds_' + str(args.commrounds) + '_clientfr_' + str(
        args.clientfr) + '_numclients_' + str(args.numclient) + '_clientepochs_' + str(
        args.clientepochs) + '_clientbs_' + str(args.clientbs) + '_clientLR_' + str(args.clientlr)
    print(plot_str)

    trained_model = training(cifar_cnn, H[0], H[4], H[5], cifar_data_train, data_dict, H[1], H[2], H[3], plot_str,
                             "green", cifar_data_test, 128, criterion, num_classes, classes_test, args.sch_flag)