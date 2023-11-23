import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms 

import os

def get_ds_cifar10(config: dict = None):
    path_data = '../data/cifar10' 
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


    ds_train = datasets.CIFAR10(root=path_data, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4), # ->(32, 32)
            transforms.ToTensor(),
            normalize]), download=True)
    
    ds_test = datasets.CIFAR10(root=path_data, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))  
    return ds_train, ds_test
    

def test_get_ds_cifar10():
    ds_train, ds_test = get_ds_cifar10({'1': 0})
    dl_train = DataLoader(dataset=ds_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    for idx, batch in enumerate(dl_train):
        img_, label_ = batch
        img_, label_ = img_.to('cuda'), label_.to('cuda')
        print(img_.shape, label_.shape)
        print(label_)
        break

if __name__ == "__main__":
    test_get_ds_cifar10()
    

