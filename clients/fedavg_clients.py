from torch import nn
import copy
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class Benign_clients(object):
    def __init__(self, model, dataloader, config):
        self._local_model = model
        self._E = config['epoch_local'] # local epochs
        self._local_dataloader = dataloader
        if config['optim_local'] == 'SGD':
            optimizer = torch.optim.SGD(
                self._local_model.parameters(), lr=config['lr_local'],
                momentum=config['mom_local'])
        elif config['optim_local'] == 'Adam':
            optimizer = torch.optim.Adam(self._local_model.parameters(),
                                         lr=config['lr_local'])
        else:
            raise ValueError(config['optim_local'])
        self._optimizer = optimizer
        self._device = config['device']
        # self._omega_history = []
        # # TODO: add this to args
        # self._timesteps = 5

    # TODO: for G_t and use copy! 
    # TODO: compute omega and weight update sim in global server
    def local_update(self, G_t, global_epoch):
        self._local_model.to('cuda')
        self._local_model.load_state_dict(G_t) 
        self._local_model.train()
        error = nn.CrossEntropyLoss()
        for idx_local_epoch in range(self._E):
             for images, labels in self._local_dataloader:
                images, labels = images.cuda(), labels.cuda()
                self._optimizer.zero_grad()
                log_probs = self._local_model(images)
                loss = error(log_probs, labels)
                loss.backward()
                self._optimizer.step()
        local_tp1 = self._local_model.state_dict()
        L_i = {} 
        for key in G_t.keys():
            L_i[key] = local_tp1[key]-G_t[key] 
        torch.cuda.empty_cache() 
        self._local_model.to('cpu')
        return L_i


class Benign_clients_2(object):
    def __init__(self, ds, idxs, config):
        self.idxs_ = idxs
        self.E_ = config['epoch_local'] # local epochs
        self.lr_ = config['lr_local'] 
        self.device_ = config['device']
        self.dl_ = DataLoader(CustomDataset(ds, idxs), batch_size=config['train_batch_size'], shuffle=True)

    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5) 
        for epoch in range(1, self.E_ + 1):
            train_loss = 0.0
            model.train()
            for data, labels in self.dl_:
                data, labels = data.to(self.device_), labels.to(self.device_)
                optimizer.zero_grad()
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                scheduler.step(train_loss)

        return model.state_dict() 