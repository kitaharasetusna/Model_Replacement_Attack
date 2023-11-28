import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms 

import os

# TODO: remove this after 5
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

def get_ds_cifar10_2(config: dict = None):
    path_data = '../data/cifar10' 
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.Normalize(*stats)])
    
    transforms_cifar_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(*stats)]) 
    
    cifar_data_train = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_cifar_train)
    cifar_data_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_cifar_test) 

    return cifar_data_train, cifar_data_test

def get_ds_mnist():
    trans_mnist = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
    test_trans_mnist = transforms.Compose(
                        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds_train = datasets.MNIST(
                '../data/mnist/', train=True, download=True, transform=trans_mnist)
    ds_test = datasets.MNIST(
                '../data/mnist/', train=False, download=True, transform=test_trans_mnist)
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

class Non_iid(Dataset): 
    def __init__(self, x, y):
        # self.x_data = x.unsqueeze(1).to(torch.float32)
        self.x_data = x.to(torch.float32)
        # self.y_data = y.to(torch.int64)
        self.y_data = y.to(torch.int64)
        self.batch_size = 64 # set batchsize in here
        self.cuda_available = torch.cuda.is_available()
        
    # Return the number of data 
    def __len__(self): 
        return len(self.x_data)
    
    # Sampling
    def __getitem__(self, idx): 
        # idx = np.random.randint(low = 0, high= len(self.x_data), size=self.batch_size) # random_index
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y 


if __name__ == "__main__":
    test_get_ds_cifar10()