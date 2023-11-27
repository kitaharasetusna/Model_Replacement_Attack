from torch import nn
import copy
import torch

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
    
        return L_i