import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torch.utils.data import DataLoader, Dataset
import pickle
import copy
from torchvision.models.feature_extraction import create_feature_extractor

# set manual seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sys
sys.path.append('..')  # Adds the parent directory to the Python path1
from my_utils.utils_model import MyGroupNorm, get_norm_per_layer, plot_dict_ret_func 
from my_utils.utils_train import central_test_backdoor, central_benign_training, central_malicious_training, add_trigger
from my_utils.utils_dataloader import get_ds_cifar10 
from my_utils.utils_reading_disks import get_dict_from_yaml

if __name__ == '__main__':
    # -------------------------------------- 0. load config ------------------
    path_config = '../configs/8_centralized_resnet18.yaml'
    configs = get_dict_from_yaml(path=path_config)
    print('exp name: ',configs['exp_name'])
    print(configs)

    
    # --------------------------------------1. load datasets; dataloaders  ---------------- 
    ds_train, ds_test = get_ds_cifar10()

    num_dataset = len(ds_train)
    num_malicious_dataset = len(ds_train)//10

    data_distribute = np.random.permutation(num_dataset)
    malicious_dataset=[]
    mal_val_dataset=[]
    mal_train_dataset=[]

    for i in range(num_malicious_dataset):
        malicious_dataset.append(ds_train[data_distribute[i]])
        if i < num_malicious_dataset//4:
            mal_val_dataset.append(ds_train[data_distribute[i]])
        else:
            mal_train_dataset.append(ds_train[data_distribute[i]])
    
    dl_train = DataLoader(mal_train_dataset, batch_size = configs['train_batch_size'], shuffle=True)
    dl_val = DataLoader(mal_val_dataset, batch_size = configs['test_batch_size'], shuffle=True)

    # --------------------------------------2. init model ----------
    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=configs['num_class'], zero_init_residual=False, groups=1,
                                  width_per_group=64, replace_stride_with_dilation=None, norm_layer=MyGroupNorm) 
    
    model = model.to(configs['device'])

    
    #---------------------------------------3. train benign and backdoor models--
    model_benign = copy.deepcopy(model)
    model_malicious = copy.deepcopy(model)
    if configs['train_from_scratch'] == True:
        for i in range(15):
            central_benign_training(model=model_benign, dl_train=dl_train, configs=configs)
        model_malicious.load_state_dict(model_benign.state_dict())
        for i in range(5):
            central_malicious_training(model=model_malicious, dl_train=dl_train, configs=configs)
        torch.save(model_benign.state_dict(), configs['path_benign_model'])
        torch.save(model_malicious.state_dict(), configs['path_malicious_model'])
    else: 
        dict_benign_model = torch.load(configs['path_benign_model'])
        dict_malicious_model = torch.load(configs['path_malicious_model'])
        model_benign.load_state_dict(dict_benign_model)
        model_malicious.load_state_dict(dict_malicious_model)
    loss, acc, BSR = central_test_backdoor(model=model_benign, dl_test=mal_train_dataset, configs=configs)
    print('loss: ', loss, ' acc: ', acc, ' BSR: ', BSR)
    loss, acc, BSR = central_test_backdoor(model=model_malicious, dl_test=mal_train_dataset, configs=configs)
    print('loss: ', loss, ' acc: ', acc, ' BSR: ', BSR)

    #---------------------------------------4. malicious model through benign data
    
    #  #---------------------------------------4.1. malicious model through benign data 
    
    
    # print(avg_norms) 

    dict_ret = get_norm_per_layer(model=model_malicious, dl_test=dl_val, configs=configs)
    print(dict_ret)
    plot_dict_ret_func(dict_ret)

    dict_ret2 = get_norm_per_layer(model=model_benign, dl_test=dl_val, configs=configs)
    print(dict_ret2)
    plot_dict_ret_func(dict_ret2)
    
    

    
        
    


    
    
     
    
    

