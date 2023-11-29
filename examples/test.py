# Commented out IPython magic to ensure Python compatibility.
#   %load_ext tensorboard
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import pickle

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
from my_utils.utils_dataloader import get_ds_cifar10 
from my_utils.utils_reading_disks import get_dict_from_yaml

if __name__ == '__main__':

    # -------------------------------------- 0. load config ------------------
    path_config = '../configs/5_cifar_10_sra_fl_non_iid.yaml'
    configs = get_dict_from_yaml(path=path_config)
    print(configs)
    # -------------------------------------- 0. load config ------------------ 

    # --------------------------------------1. load datasets  ---------------- 
    ds_train, ds_test = get_ds_cifar10()
    # --------------------------------------1. load datasets  ----------------

    classes = np.array(list(ds_train.class_to_idx.values()))
    classes_test = np.array(list(ds_test.class_to_idx.values()))
    num_classes = len(classes_test)

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------2. gen non-iid idx  -------------------------------------------------------------
    folder_idx = '../idx_'+configs['exp_name']
    import os
    if not os.path.exists(folder_idx):
        os.mkdir(folder_idx)
    if configs['load_idx']==True:
        print('loading from prev idx')
        with open(folder_idx+'/idxs.pkl', 'rb') as f:
            data_dict = pickle.load(f)
            f.close()
    else:
        print('getting new idx...')
        # data_dict = non_iid_partition(ds_train, configs['num_clients'], configs['degree_non_iid'])
        if configs['non_iid'] == True:
        # (dataset, clients, total_shards, shards_size, num_shards_per_client):
        # alpha for the Dirichlet distribution
            data_dict = non_iid_partition(ds_train, configs['num_clients'], configs['degree_non_iid'])
        else:
            data_dict = iid_partition(ds_train, 100)  # Uncomment for idd_partition
        with open(folder_idx+'/idxs.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            f.close()
    # ---------------------------------------------------------------------------------------------------


    
    cifar_cnn = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
                                  width_per_group=64, replace_stride_with_dilation=None, norm_layer=MyGroupNorm)
    
    cifar_cnn.cuda()

    trained_model = training(cifar_cnn, ds_train, data_dict,
                              ds_test, criterion, classes_test, False, config=configs)