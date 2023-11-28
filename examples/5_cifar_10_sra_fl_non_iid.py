from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision
import pickle
import random
from torchvision.models import resnet


# set manual seed for reproducibility
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sys
sys.path.append('..')  # Adds the parent directory to the Python path1

from my_utils.utils_dataloader import get_ds_cifar10_2, Non_iid
from my_utils.utils_reading_disks import get_dict_from_yaml
from clients.fedavg_clients import Benign_clients_2
from models.cifar10.models import CNN_CIFAR10
from models.cifar10.models import resnet18, MyGroupNorm 
from my_utils.utils_train import test_model

def non_iid_partition(dataset, n_nets, alpha):
    """
        :param dataset: dataset name
        :param n_nets: number of clients
        :param alpha: beta parameter of the Dirichlet distribution
        :return: dictionary containing the indexes for each client
    """
    y_train = np.array(dataset.targets)
    min_size = 0
    K = 10
    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])

    # net_dataidx_map is a dictionary of length #of clients: {key: int, value: [list of indexes mapping the data among the workers}
    # traindata_cls_counts is a dictionary of length #of clients, basically assesses how the different labels are distributed among
    # the client, counting the total number of examples per class in each client.
    return net_dataidx_map


# -------------------------------------- 0. load config -------------------------------------------------------------
path_config = '../configs/5_cifar_10_sra_fl_non_iid.yaml'
configs = get_dict_from_yaml(path=path_config)
print(configs)
# ---------------------------------------------------------------------------------------------------

# --------------------------------------1. load datasets  -------------------------------------------------------------
ds_train, ds_test = get_ds_cifar10_2(config=configs)
dl_train = DataLoader(dataset=ds_train, batch_size=configs['train_batch_size'], shuffle=True) 
dl_test = DataLoader(dataset=ds_test, batch_size=configs['train_batch_size'], shuffle=False)
# ---------------------------------------------------------------------------------------------------

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
    data_dict = non_iid_partition(ds_train, configs['num_clients'], configs['degree_non_iid'])
    with open(folder_idx+'/idxs.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
        f.close()
# ---------------------------------------------------------------------------------------------------


# statistic value
# MAR BSR ACC
num_selected = int(configs['num_clients']*configs['C'])
num_comp = int(configs['num_clients']*configs['ratio_comp'])
num_benign = configs['num_clients'] - num_comp
print('num of client', num_benign)

# TODO: clients
# -------------------------------------- 3. init glob model  -----------------------------------------------
if configs['name_model']=='cifar10_resnet':
    model_global = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, zero_init_residual=False, groups=1,
                                  width_per_group=64, replace_stride_with_dilation=None, norm_layer=MyGroupNorm)
if configs['load_model']:
    print('loading model: '+configs['path_ckpt'])
    model_global.load_state_dict(torch.load(configs['path_ckpt']))
model_global = model_global.to(configs['device'])
# ---------------------------------------------------------------------------------------------------

# --------------------------------------5. init collection for metrics(ACC, MSA, BSR)  -----------------------------------------------
accs = []
if configs['load_accs']:
    with open('../idx_'+configs['exp_name']+'_accs.pkl', 'rb') as f:
        accs = pickle.load(f) 
        f.close()
print('accs', accs)
# ---------------------------------------------------------------------------------------------------

for epoch_ in range(len(accs)*configs['time_step'], configs['num_epoch']):
    G_t = model_global.state_dict()
    list_L_t = []
    
    selected_idxs = np.random.randint(0, configs['num_clients'] ,num_selected).tolist()
    print(epoch_%10, ' S: ', selected_idxs)
    for idx_selected in selected_idxs:
        # ---------------6 gen clients
        local_update = Benign_clients_2(ds=ds_train, idxs=data_dict[idx_selected], config=configs) 
        L_i = local_update.train(model=copy.deepcopy(model_global))
        list_L_t.append(L_i)
    
    G_t_1 = {}
            
    for key in list_L_t[0].keys():
        G_t_1[key] = G_t[key]+ configs['lr_global']*torch.stack([list_L_t[i][key].float() for i in range(num_selected)], dim=0).mean(dim=0)
    model_global.load_state_dict(G_t_1) 
    
    if (epoch_+1)%configs['time_step'] == 0 or epoch_==0:
        test_acc = test_model(model_global, dl_train, config=configs) 
        train_acc = test_model(model_global, dl_test, config=configs)
        accs.append((train_acc, test_acc))
        print('epoch: '+str(epoch_)+'/'+ str(configs['num_epoch'])+'\ntest acc:', test_acc, '\ntrain acc: ', train_acc)
        torch.save(model_global.state_dict(), configs['path_ckpt'])
        with open('../idx_'+configs['exp_name']+'_accs.pkl', 'wb') as f:
            pickle.dump(accs, f) 
            f.close()
    
print(accs)