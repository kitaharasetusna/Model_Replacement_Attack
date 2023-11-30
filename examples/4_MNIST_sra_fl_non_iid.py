from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision
import pickle
import random

import sys
sys.path.append('..')  # Adds the parent directory to the Python path1

from my_utils.utils_dataloader import get_ds_cifar10, Non_iid, get_ds_mnist, iid_partition, non_iid_partition
from my_utils.utils_reading_disks import get_dict_from_yaml
from clients.fedavg_clients import Benign_clients
from models.cifar10.models import CNN_CIFAR10, CNN_MNIST
from models.cifar10.models import resnet18 
from my_utils.utils_train import test_model

# set manual seed for reproducibility
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


path_config = '../configs/4_mnist_sra_fl_non_iid.yaml'
configs = get_dict_from_yaml(path=path_config)
print(configs)

ds_train, ds_test = get_ds_mnist()
dl_train = DataLoader(dataset=ds_train, batch_size=configs['train_batch_size'], shuffle=True) 
dl_test = DataLoader(dataset=ds_test, batch_size=configs['train_batch_size'], shuffle=False)


idx   = [ torch.where(torch.from_numpy(np.array(ds_train.targets)) == i) for i in range(configs['num_class']) ]
data  = [ ds_train.data[idx[i][0]] for i in range(configs['num_class']) ]
label = [ torch.ones(len(data[i]))* i for i in range(configs['num_class'])]

if configs['non_iid'] == True:
    # (num clients, num_class)
    s = np.random.dirichlet(np.ones(configs['num_class'])*configs['degree_non_iid'], configs['num_clients'])
# (number of clients, num of classes)
data_dist = np.zeros((configs['num_clients'], configs['num_class']))


if bool(configs['load_idx'])==True:    
    print('loading from prev idx')
else:
    print('getting new idx...')

folder_idx = '../idx_'+configs['exp_name']
import os
if not os.path.exists(folder_idx):
    os.mkdir(folder_idx)
# row i: for clients i, number for each class
if configs['non_iid'] == True:
    X = []
    Y = []
    if configs['load_idx']:
        with open(folder_idx+'/idxs_'+str(configs['degree_non_iid'])+'.pkl', 'rb') as f:
            data_dict = pickle.load(f)
            f.close()
    else:
        data_dict = non_iid_partition(ds_train, configs['num_clients'], configs['degree_non_iid'])
        with open(folder_idx+'/idxs_'+str(configs['degree_non_iid'])+'.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            f.close()
    print(type(data_dict[0]))
    for idx_client in range(len(data_dict)):
        x_data = ds_train.data[list(data_dict[idx_client])]
        y_data = ds_train.targets[list(data_dict[idx_client])]
        X.append(x_data)
        Y.append(y_data) 
else:
    X = []
    Y = []
    if configs['load_idx']:
        with open(folder_idx+'/idxs_'+str(configs['degree_non_iid'])+'.pkl', 'rb') as f:
            data_dict = pickle.load(f)
            f.close()
    else:
        data_dict = iid_partition(ds_train, configs['num_clients'])
        with open(folder_idx+'/idxs_'+str(configs['degree_non_iid'])+'.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            f.close()
    print(type(data_dict[0]))
    for idx_client in range(len(data_dict)):
        x_data = ds_train.data[list(data_dict[idx_client])]
        y_data = ds_train.targets[list(data_dict[idx_client])]
        X.append(x_data)
        Y.append(y_data)

Non_iid_dataset  = [Non_iid(X[j],Y[j]) for j in range(configs['num_clients'])]
Non_iid_dataloader = [DataLoader(ds, batch_size = configs['train_batch_size'], shuffle=True) for ds in Non_iid_dataset]

# statistic value
# MAR BSR ACC
num_selected = int(configs['num_clients']*configs['C'])
num_comp = int(configs['num_clients']*configs['ratio_comp'])
num_benign = configs['num_clients'] - num_comp
print(num_benign)

if configs['name_model']=='resnet':
    models = [torchvision.models.resnet18(pretrained=False)
          for idx_benign in range(num_benign)]
    new_models = []
    for model in models:
        model.fc = nn.Linear(model.fc.in_features, 10)
        new_models.append(model)
    del models
    clients = [Benign_clients(model=new_models[idx_benign],
                            dataloader=Non_iid_dataloader[idx_benign],
                            config=configs) 
            for idx_benign in range(num_benign)] 
    del new_models
elif configs['name_model']=='cnn':
    clients = [Benign_clients(model=CNN_MNIST(),
                            dataloader=Non_iid_dataloader[idx_benign],
                            config=configs) 
            for idx_benign in range(num_benign)] 

# TODO: add attack
# TODO: add tqdm
# TODO: find where make this (32, 32, 3) but not (3, 32, 32)
# model_global = CNN_CIFAR10(in_channels=3, num_classes=10).to(configs['device'])
if configs['name_model']=='resnet':
    model_global = torchvision.models.resnet18(pretrained=False)
    num_classes = 10  # CIFAR-10 has 10 classes
    model_global.fc = nn.Linear(model.fc.in_features, num_classes)
    model_global = model_global.to(configs['device'])
elif configs['name_model']=='cnn':
    model_global = CNN_MNIST()
    model_global = model_global.to(configs['device'])

if configs['load_model']:
    print('loading model: '+configs['path_ckpt'])
    model_global.load_state_dict(torch.load(configs['path_ckpt']+'_'+str(configs['degree_non_iid'])+'.pth'))

accs = []
if configs['load_accs']:
    with open('../idx_'+configs['exp_name']+'_accs_'+str(configs['degree_non_iid'])+'.pkl', 'rb') as f:
        accs = pickle.load(f) 
        f.close()
    print(accs)
for epoch_ in range(len(accs)*10, configs['num_epoch']):
    G_t = model_global.state_dict()
    list_L_t = []
    
    selected_idxs = np.random.randint(0, configs['num_clients'] ,num_selected).tolist()
    print(epoch_%10, ' S: ', selected_idxs)
    for idx_selected in selected_idxs:
        L_i = clients[idx_selected].local_update(G_t=copy.deepcopy(G_t), global_epoch=epoch_)
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
        torch.save(model_global.state_dict(), configs['path_ckpt']+'_'+str(configs['degree_non_iid'])+'.pth') 
        with open('../idx_'+configs['exp_name']+'_accs_'+str(configs['degree_non_iid'])+'.pkl', 'wb') as f:
            pickle.dump(accs, f) 
            f.close()
    
print(accs)