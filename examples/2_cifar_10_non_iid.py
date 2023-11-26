from torch.utils.data import DataLoader
import torch
import numpy as np

import sys
sys.path.append('..')  # Adds the parent directory to the Python path1

from my_utils.utils_dataloader import get_ds_cifar10, Non_iid
from my_utils.utils_reading_disks import get_dict_from_yaml

path_config = '../configs/2_cifar_10_non_iid.yaml'
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
            x_data.append(data[i][d_index])
            y_data.append(label[i][d_index])
    X.append(torch.cat(x_data))
    Y.append(torch.cat(y_data))
Non_iid_dataset  = [Non_iid(X[j],Y[j]) for j in range(configs['num_class'])]
Non_iid_dataloader = [DataLoader(ds, batch_size = configs['train_batch_size'], shuffle=True) for ds in Non_iid_dataset]
