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
