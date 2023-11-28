from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision
import pickle

import sys
sys.path.append('..')  # Adds the parent directory to the Python path1

from my_utils.utils_dataloader import get_ds_cifar10, Non_iid
from my_utils.utils_reading_disks import get_dict_from_yaml
from clients.fedavg_clients import Benign_clients
from models.cifar10.models import CNN_CIFAR10
from models.cifar10.models import resnet18 
from my_utils.utils_train import test_model

class ReplaceBatchNorm(nn.Module):
    def __init__(self, orig_model):
        super(ReplaceBatchNorm, self).__init__()
        self.features = nn.Sequential(*list(orig_model.children())[:-2])  # Remove AvgPool and FC layers

        for name, module in self.features.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                setattr(self.features, name, nn.GroupNorm(2, num_channels))

        self.avgpool = orig_model.avgpool
        self.fc = orig_model.fc

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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


if bool(configs['load_idx'])==True:    
    print('loading from prev idx')
else:
    print('getting new idx...')

folder_idx = '../idx_'+configs['exp_name']
import os
if not os.path.exists(folder_idx):
    os.mkdir(folder_idx)
# row i: for clients i, number for each class
if bool(configs['load_idx'])==False:
    for j in range(configs['num_clients']):
        # num of samples selected for a given class
        data_dist[j] = ((s[j]*len(data[0])).astype('int') / 
                        (s[j]*len(data[0])).astype('int').sum() * len(data[0])).astype('int')
        data_num     = data_dist[j].sum()
        data_dist[j][np.random.randint(low=0,high=configs['num_class'])] += ((len(data[0]) - data_num) )
        data_dist    = data_dist.astype('int')
    with open(folder_idx+'/idxs.pkl', 'wb') as f:
        pickle.dump(data_dist, f)
        f.close()
else:
    with open(folder_idx+'/idxs.pkl', 'rb') as f:
        data_dist = pickle.load(f)
        f.close()


# TODO: you can save this part to repeat the experiment
print(data_dist)


X = []
Y = []
for j in range(configs['num_clients']):
    x_data = []
    y_data = []
    for i in range(configs['num_class']):
        if data_dist[j][i] != 0:

            
            if bool(configs['load_idx'])==False:
                d_index = np.random.randint(low=0, high=len(data[i]), size=data_dist[j][i])
                with open(folder_idx+'/'+str(j)+'_'+str(i)+''+'.pkl', 'wb') as f:
                    pickle.dump(d_index, f)
                    f.close()
            else:
                
                if os.path.exists(folder_idx+'/'+str(j)+'_'+str(i)+''+'.pkl'):
                    with open(folder_idx+'/'+str(j)+'_'+str(i)+''+'.pkl', 'rb') as f:
                        d_index = pickle.load(f)
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

if configs['name_model']=='resnet':
    models = [torchvision.models.resnet18(pretrained=False)
          for idx_benign in range(num_benign)]
    new_models = []
    for model in models:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model = ReplaceBatchNorm(model)
        model.fc = nn.Linear(model.fc.in_features, 10)
        new_models.append(model)
    del models
    clients = [Benign_clients(model=new_models[idx_benign],
                            dataloader=Non_iid_dataloader[idx_benign],
                            config=configs) 
            for idx_benign in range(num_benign)] 
    del new_models

# TODO: add attack
# TODO: add tqdm
# TODO: find where make this (32, 32, 3) but not (3, 32, 32)
# model_global = CNN_CIFAR10(in_channels=3, num_classes=10).to(configs['device'])
if configs['name_model']=='resnet':
    model_global = torchvision.models.resnet18(pretrained=False)
    model_global.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model_global = ReplaceBatchNorm(model_global)
    num_classes = 10  # CIFAR-10 has 10 classes
    model_global.fc = nn.Linear(model.fc.in_features, num_classes)
    model_global = model_global.to(configs['device'])

if configs['load_model']:
    print('loading model: '+configs['path_ckpt'])
    model_global.load_state_dict(torch.load(configs['path_ckpt']))

accs = []
if configs['load_accs']:
    with open('../idx_'+configs['exp_name']+'_accs.pkl', 'rb') as f:
        accs = pickle.load(f) 
        f.close()
    print(accs)
for epoch_ in range(len(accs), configs['num_epoch']):
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
        torch.save(model_global.state_dict(), configs['path_ckpt'])
        with open('../idx_'+configs['exp_name']+'_accs.pkl', 'wb') as f:
            pickle.dump(accs, f) 
            f.close()
    
print(accs)