from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision

import sys
sys.path.append('..')  # Adds the parent directory to the Python path1

from my_utils.utils_dataloader import get_ds_cifar10, Non_iid
from my_utils.utils_reading_disks import get_dict_from_yaml
from clients.fedavg_clients import Benign_clients
from models.cifar10.models import CNN_CIFAR10
from models.cifar10.models import resnet110
from my_utils.utils_train import test_model

path_config = '../configs/3_cifar_10_sra_fl_non_iid.yaml'
configs = get_dict_from_yaml(path=path_config)
print(configs)

ds_train, ds_test = get_ds_cifar10(config=configs)
dl_train = DataLoader(dataset=ds_train, batch_size=configs['train_batch_size'], shuffle=True) 
dl_test = DataLoader(dataset=ds_test, batch_size=configs['train_batch_size'], shuffle=False)


idx   = [ torch.where(torch.from_numpy(np.array(ds_train.targets)) == i) for i in range(configs['num_class']) ]
data  = [ torch.from_numpy(ds_train.data[idx[i][0]]) for i in range(configs['num_class']) ]
label = [ torch.ones(len(data[i]))* i for i in range(configs['num_class'])]

# (num clients, num_class)
s = np.random.dirichlet(np.ones(configs['num_class'])*configs['degree_non_iid'], configs['num_clients'])
# (number of clients, num of classes)
data_dist = np.zeros((configs['num_clients'], configs['num_class']))

# row i: for clients i, number for each class
for j in range(configs['num_clients']):
    # num of samples selected for a given class
    data_dist[j] = ((s[j]*len(data[0])).astype('int') / 
                    (s[j]*len(data[0])).astype('int').sum() * len(data[0])).astype('int')
    data_num     = data_dist[j].sum()
    data_dist[j][np.random.randint(low=0,high=configs['num_class'])] += ((len(data[0]) - data_num) )
    data_dist    = data_dist.astype('int')

# TODO: you can save this part to repeat the experiment
print(data_dist)

X = []
Y = []
for j in range(configs['num_clients']):
    x_data = []
    y_data = []
    for i in range(configs['num_class']):
        if data_dist[j][i] != 0:
            d_index = np.random.randint(low=0, high=len(data[i]), size=data_dist[j][i])
            x_data.append(data[i][d_index].reshape(-1, 3, 32, 32))
            y_data.append(label[i][d_index])
    X.append(torch.cat(x_data))
    Y.append(torch.cat(y_data))
Non_iid_dataset  = [Non_iid(X[j],Y[j]) for j in range(configs['num_clients'])]
Non_iid_dataloader = [DataLoader(ds, batch_size = configs['train_batch_size'], shuffle=True) for ds in Non_iid_dataset]

# statistic value
# MAR BSR ACC
num_selected = int(configs['num_clients']*configs['C'])
num_comp = int(configs['num_clients']*configs['ratio_comp'])
num_benign = configs['num_clients'] - num_comp
print(num_benign)
# models = [torchvision.models.resnet18(pretrained=True).to(configs['device']) 
#           for idx_benign in range(num_benign)]
# new_models = []
# for model in models:
#     model.fc = nn.Linear(model.fc.in_features, 10)
#     new_models.append(model)
# del models
# clients = [Benign_clients(model=new_models[idx_benign].to(configs['device']),
#                           dataloader=Non_iid_dataloader[idx_benign],
#                           config=configs) 
#            for idx_benign in range(num_benign)]
clients = [Benign_clients(model=CNN_CIFAR10().to(configs['device']),
                          dataloader=Non_iid_dataloader[idx_benign],
                          config=configs) 
           for idx_benign in range(num_benign)]

# TODO: add attack
# TODO: add tqdm
# TODO: find where make this (32, 32, 3) but not (3, 32, 32)
model_global = CNN_CIFAR10().to(configs['device'])
# model_global = CNN_CIFAR10().to(configs['device'])

# model_global = torchvision.models.resnet18(pretrained=True)
# num_classes = 10  # CIFAR-10 has 10 classes
# model_global.fc = nn.Linear(model.fc.in_features, num_classes)
# model_global = model_global.to(configs['device'])
accs = []
for epoch_ in range(configs['num_epoch']):
    G_t = model_global.state_dict()
    list_L_t = []
    
    selected_idxs = np.random.randint(0, configs['num_clients'] ,num_selected).tolist()
    print('S: ', selected_idxs)
    for idx_selected in selected_idxs:
        L_i = clients[idx_selected].local_update(G_t=copy.deepcopy(G_t), global_epoch=epoch_)
        list_L_t.append(L_i)
    
    G_t_1 = {}
            
    for key in list_L_t[0].keys():
        G_t_1[key] = G_t[key]+ configs['lr_global']*torch.stack([list_L_t[i][key] for i in range(num_selected)], dim=0).mean(dim=0)
    model_global.load_state_dict(G_t_1) 
    
    if (epoch_+1)%configs['time_step'] == 0 or epoch_==0:
        test_acc = test_model(model_global, dl_train, config=configs) 
        train_acc = test_model(model_global, dl_test, config=configs)
        accs.append((train_acc, test_acc))
        print('test acc:', test_acc, ' train acc: ', train_acc)
    
    
print(accs)