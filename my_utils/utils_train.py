import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import copy
import pickle
import torch.nn.functional as F
import heapq

import sys
sys.path.append('..')  # Adds the parent directory to the Python path1
from my_utils.utils_model import add_trigger
from my_utils.utils_attack import get_attack_layers
from my_utils.utils_defence import fedavg, flame
from my_utils.utils_defence import model2vector

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



def test_model(model, dataloader, config):
    total = 0
    correct = 0 
    error = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(config['device']), target.to(config['device'])
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    # print('acc:', correct, total)
    return 100 * correct/total

class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs, sch_flag):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sch_flag = sch_flag

    def train(self, model):

        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay = 5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.sch_flag == True:
           scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
        # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0

            model.train()
            for data, labels in self.train_loader:
                if data.size()[0] < 2:
                    continue

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                if self.sch_flag == True:
                 scheduler.step(train_loss)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

            # self.learning_rate = optimizer.param_groups[0]['lr']

        total_loss = sum(e_loss) / len(e_loss)

        return model.state_dict(), total_loss

class MaliciousClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs, sch_flag, configs):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sch_flag = sch_flag
        self.configs = configs

    def train(self, model):

        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay = 5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.sch_flag == True:
           scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
        # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0

            model.train()
            for data, labels in self.train_loader:
                bad_input_, bad_label_ = copy.deepcopy(data), copy.deepcopy(labels)
                if self.configs['type_attack'] == 'badnet' or self.configs['type_attack'] =='scaling_attack':
                    for xx in range(len(bad_input_)):
                        bad_label_[xx] = self.configs['attack_label']
                        # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                        bad_input_[xx] = add_trigger(image=bad_input_[xx], configs=self.configs)
                # TODO: add 'LR'
                else:
                    raise ValueError(self.configs['type_attack'])
                    
                data = torch.cat((data, bad_input_), dim=0)
                labels = torch.cat((labels, bad_label_))
                data, labels = data.to(self.configs['device']), labels.to(self.configs['device'])
                if data.size()[0] < 2:
                    continue

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                if self.sch_flag == True:
                 scheduler.step(train_loss)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

            # self.learning_rate = optimizer.param_groups[0]['lr']

        total_loss = sum(e_loss) / len(e_loss)
        ret_dict = {}
        L_t = model.state_dict()
        if self.configs['type_attack'] =='scaling_attack':
            for key in L_t.keys():
                ret_dict[key] = L_t[key]*self.configs['scale_alpha'] 
        else:
            ret_dict = L_t
        return ret_dict, total_loss
    
    def train_layerwise_poisoning(self, model, ds_mal_val, ds_mal_train, args: dict, threshold = 0.8):
        dl_mal_train = DataLoader(dataset=ds_mal_train, batch_size=self.configs['train_batch_size'],
                                  shuffle=True)
        criterion = nn.CrossEntropyLoss()
        e_loss = []
        
        good_param = copy.deepcopy(model.state_dict())
        badnet = copy.deepcopy(model)
        
        optimizer = torch.optim.Adam(badnet.parameters(), lr=self.learning_rate)
        if self.sch_flag == True:
           scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
        
        # step 1. train a badnet ------------
        badnet.train() 
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            
            for data, labels in self.train_loader:
                bad_input_, bad_label_ = copy.deepcopy(data), copy.deepcopy(labels)
                
                for xx in range(len(bad_input_)):
                    bad_label_[xx] = self.configs['attack_label']
                    # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    bad_input_[xx] = add_trigger(image=bad_input_[xx], configs=self.configs)
                
                data = torch.cat((data, bad_input_), dim=0)
                labels = torch.cat((labels, bad_label_))
                data, labels = data.to(self.configs['device']), labels.to(self.configs['device'])
                if data.size()[0] < 2:
                    continue
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = badnet(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                if self.sch_flag == True:
                 scheduler.step(train_loss)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)
        total_loss = sum(e_loss) / len(e_loss)

        bad_net_param = badnet.state_dict()
        # ------------------------------------------

        # step 2. train a local benign model-----------------------------------
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.sch_flag == True:
           scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
        epoch_loss = []
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            
            for data, labels in self.train_loader:
                data, labels = data.to(self.configs['device']), labels.to(self.configs['device'])
                if data.size()[0] < 2:
                    continue
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                if self.sch_flag == True:
                 scheduler.step(train_loss)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            epoch_loss.append(train_loss)
        total_loss = sum(epoch_loss) / len(epoch_loss)
        #----------------------------------------------
        # TODO: make the following on a [net_ret<-new trained benign model] (make a backup [model benign])
        # after some optims on ds_mal_train until it reached 93 acc (or 80 after 30 epochs)
        # TODO: then we train on the [new train benign model (copy)] on ds_mal_train to get the [malicious model]  
        # TODO: get BSR on ds_mal_val of the [malicious model]
        # TODO: add fixed layers if it already achieved high BSR w.r.t to the original local backdoor 
        # TODO: use the attack list to get the local malicious model we wanted
        good_weight = model.state_dict()
        bad_weight = badnet.state_dict()
        key_arr = []
        value_arr = []
        # 1
        model_copy = copy.deepcopy(model) 
        model_benign = copy.deepcopy(model)
        _, acc, BSR_bn = central_test_backdoor(model=model_benign, dl_test=ds_mal_train, configs=self.configs)
        if self.configs['dataset'] == 'cifar10':
            min_acc = 93
        else:
            min_acc = 90
        num_time = 0
        pass_ = False
        while(acc<min_acc):
            central_benign_training(model=model_benign, dl_train=dl_mal_train, configs=self.configs)
            num_time += 1
            if num_time%4==0:
                _,acc, _ = central_test_backdoor(model=model_benign, dl_test=ds_mal_train, configs=self.configs)
                print(acc, '/',min_acc)
                model_copy = model_benign
                if num_time > 30:
                    if acc > 80:
                        break
                    else:
                        attack_list = []
                        pass_ = True
                        break
        if acc>=min_acc:
            pass_ =False
        if pass_ == False:
            # 2 
            model_malicious = copy.deepcopy(model)
            central_malicious_training(model_malicious, dl_mal_train, self.configs)
            _, acc_bn, BSR_bn = central_test_backdoor(model=model_benign, dl_test=ds_mal_val, configs=self.configs)
            print("benign model testset result(acc/backdoor):", acc_bn, BSR_bn) 
            _, acc_mal, BSR_mal = central_test_backdoor(model=model_malicious, dl_test=ds_mal_val, configs=self.configs)
            print("malicious model testset result(acc/backdoor):", acc_mal, BSR_mal) 

            good_weight = model_benign.state_dict()
            bad_weight = model_malicious.state_dict()
            
            temp_weight = copy.deepcopy(good_weight)
            if args['attack_layer']==None:
                args['attack_layer'] = [] 
            for layer in args['attack_layer']:
                temp_weight[layer] = bad_weight[layer]
            temp_model = copy.deepcopy(model_benign)
            temp_model.load_state_dict(temp_weight) 
            _, acc_tmp, BSR_tmp = central_test_backdoor(model=temp_model, dl_test=ds_mal_val, configs=self.configs)
            if BSR_tmp>threshold*BSR_mal:
                print(BSR_tmp, ">", threshold*BSR_mal, "SKIP")
                attack_list = args['attack_layer']
            else:
                # 
                key_arr = []
                value_arr = []
                net3 = copy.deepcopy(model_benign)
                for key, var in model_benign.named_parameters():
                    # if "bias" in key:
                    #     continue
                    param = copy.deepcopy(bad_weight)
                    param[key] = var
                    net3.load_state_dict(param)
                    _, _, back_acc2 = central_test_backdoor(model=net3, dl_test=ds_mal_val, configs=self.configs) 
                    key_arr.append(key)
                    value_arr.append(back_acc2 - BSR_mal)
                
                #
                n = 1
                temp_BSR = 0
                attack_list = []
                np_key_arr = np.array(key_arr)
                net4 = copy.deepcopy(model_benign)
                while (temp_BSR < BSR_mal * threshold and n <= len(key_arr)):
                    minValueIdx = heapq.nsmallest(n, range(len(value_arr)), value_arr.__getitem__)
                    attack_list = list(np_key_arr[minValueIdx])
                    param = copy.deepcopy(good_weight)
                    for layer in attack_list:
                        param[layer] = bad_weight[layer]
                    net4.load_state_dict(param)
                    # acc, _, temp_BSR = test_img(net4, mal_val_dataset, args, test_backdoor=True)
                    _, _, temp_BSR = central_test_backdoor(model=net4, dl_test=ds_mal_val, configs=self.configs) 
                    n += 1
                            
                
                

       
        args['attack_layer'] = attack_list 
        print('attack list', attack_list)
        attack_param = {} 
        for key, var in model.state_dict().items():
            if key in attack_list:
                # print('LP attacking...')
                attack_param[key] = bad_net_param[key] 
            else:
                attack_param[key] = var
       
        return attack_param, total_loss, attack_list



def central_benign_training(model: nn.Module, dl_train, configs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=configs['lr'], momentum=0.5) 

    for input_, label_ in dl_train:
        input_, label_ = input_.to(configs['device']), label_.to(configs['device'])
        model.zero_grad()
        log_probs = model(input_)
        loss = criterion(log_probs, label_)
        loss.backward()
        optimizer.step()

def central_malicious_training(model: nn.Module, dl_train, configs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=configs['lr'], momentum=0.5)


    for input_, label_ in dl_train:
        bad_input_, bad_label_ = copy.deepcopy(input_), copy.deepcopy(label_)
        for xx in range(len(bad_input_)):
            bad_label_[xx] = configs['attack_label']
            # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
            bad_input_[xx] = add_trigger(image=bad_input_[xx], configs=configs)
            
        input_ = torch.cat((input_, bad_input_), dim=0)
        label_ = torch.cat((label_, bad_label_))
        input_, label_ = input_.to(configs['device']), label_.to(configs['device'])
        model.zero_grad()
        log_probs = model(input_)
        loss = criterion(log_probs, label_)
        loss.backward()
        optimizer.step()

def central_test_backdoor(model: nn.Module, dl_test, configs):
    test_loss = 0
    correct = 0
    back_correct = 0
    back_num = 0
    model.eval()
    dl_test = DataLoader(dl_test, batch_size=configs['test_batch_size'])
    for idx, (data, target) in enumerate(dl_test):
        data, target = data.to(configs['device']), target.to(configs['device'])
        log_probs = model(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item() 
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # test acc
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        # test backdoor
        del_arr = []
        for k, image in enumerate(data):
            
            if target[k] != configs['attack_label']:
                data[k] = add_trigger(image=data[k], configs=configs)
                target[k] = configs['attack_label'] 
                back_num += 1
            else:
                target[k] = -1 
        log_probs = model(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(dl_test.dataset)
    accuracy = 100.00 * correct / len(dl_test.dataset)
    BSR = 100.00 * float(back_correct) / back_num
        
    return test_loss, accuracy.item(), BSR
    
def central_test_norm(model: nn.Module, dl_test, configs):
    test_loss = 0
    correct = 0
    back_correct = 0
    back_num = 0
    model.eval()
    dl_test = DataLoader(dl_test, batch_size=configs['test_batch_size'])
    for idx, (data, target) in enumerate(dl_test):
        data, target = data.to(configs['device']), target.to(configs['device'])
        log_probs = model(data)
        print(log_probs.shape)
    return 

def training(model, ds, data_dict, cifar_data_test,
            criterion, classes_test, sch_flag, config):
    # global model weights
    global_weights = model.state_dict()

    # training loss
    train_loss = []
    test_loss = []
    test_accuracy = []

    if config['load_accs']:
        with open('../idx_'+config['exp_name']+'_accs_'+str(config['degree_non_iid'])+'.pkl', 'rb') as f:
            test_accuracy = pickle.load(f) 
            f.close()
        print('accs', test_accuracy)
    best_accuracy = 0
    # measure time
    start = time.time()
    E = config['epoch_local']
    lr = config['lr_local']

    for curr_round in range(1+len(test_accuracy)*config['time_step'], config['num_epoch'] + 1):
        if curr_round == 1:
            t_accuracy, t_loss = testing(model, cifar_data_test, 
                                         config['test_batch_size'], criterion,
                                           config['num_class'], classes_test)
            test_accuracy.append(t_accuracy)
            test_loss.append(t_loss)

            if best_accuracy < t_accuracy:
                best_accuracy = t_accuracy
            # torch.save(model.state_dict(), plt_title)
            print(curr_round, t_loss, test_accuracy[-1], best_accuracy)
            # print('best_accuracy:', best_accuracy, '---Round:', curr_round, '---lr', lr, '----localEpocs--', E)
            with open('../idx_'+config['exp_name']+'_accs_'+str(config['degree_non_iid'])+'.pkl', 'wb') as f:
                pickle.dump(test_accuracy, f) 
                f.close()

        w, local_loss = [], []
        # Retrieve the number of clients participating in the current training
        m = max(int(config['C'] * config['num_clients']), 1)
        # Sample a subset of K clients according with the value defined before
        S_t = np.random.choice(range(config['num_clients']), m, replace=False)
        # For the selected clients start a local training
        for k in S_t:
            # Compute a local update
            local_update = ClientUpdate(dataset=ds, batchSize=config['train_batch_size'],
                                         learning_rate=lr, epochs=E, idxs=data_dict[k],
                                        sch_flag=sch_flag)
            # Update means retrieve the values of the network weights
            weights, loss = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
        # lr = 0.999*lr
        # updating the global weights
        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]

            weights_avg[k] = torch.div(weights_avg[k], len(w))

        global_weights = weights_avg

        if curr_round == 200:
            lr = lr / 2
            E = E - 1

        if curr_round == 300:
            lr = lr / 2
            E = E - 2

        if curr_round == 400:
            lr = lr / 5
            E = E - 3

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        # print('Round: {}... \tAverage Loss: {}'.format(curr_round, round(loss_avg, 3)), lr)
        train_loss.append(loss_avg)

        if curr_round%config['time_step'] == 0:
            t_accuracy, t_loss = testing(model, cifar_data_test, 
                                         config['test_batch_size'], 
                                         criterion, config['num_class'], classes_test)
            test_accuracy.append(t_accuracy)
            test_loss.append(t_loss)

            if best_accuracy < t_accuracy:
                best_accuracy = t_accuracy
            
            torch.save(model.state_dict(), config['path_ckpt']+'_'+str(config['degree_non_iid'])+'.pth')
            # torch.save(model.state_dict(), plt_title)
            print(curr_round, loss_avg, t_loss, test_accuracy[-1], best_accuracy)
            # print('best_accuracy:', best_accuracy, '---Round:', curr_round, '---lr', lr, '----localEpocs--', E)
            with open('../idx_'+config['exp_name']+'_accs_'+str(config['degree_non_iid'])+'.pkl', 'wb') as f:
                pickle.dump(test_accuracy, f) 
                f.close()

    return model

def training_under_attack(model, ds, data_dict, cifar_data_test,
            criterion, classes_test, sch_flag, config):
    # global model weights
    global_weights = model.state_dict()

    # training loss
    train_loss = []
    test_loss = []
    test_accuracy = []
    test_BSR = []
    ls_sel_ = []
    ls_mask_ = []

    if config['load_accs']:
        with open('../idx_'+config['exp_name']+'_accs_'+str(config['degree_non_iid'])+'.pkl', 'rb') as f:
            test_accuracy, test_BSR = pickle.load(f) 
            f.close()
        print('accs', test_accuracy)
        print('BSR: ', test_BSR)
    best_accuracy = 0
    # measure time
    start = time.time()
    E = config['epoch_local']
    lr = config['lr_local']

    # get the idx_client for malicious clients 
    num_bd = int(config['num_clients']*config['ratio_comp']) 
    
    
    if config['load_idx_bd'] == True:
        print('loading malicious clients\' index...')
        with open('../'+config['exp_name']+'_bd_idxs_'+str(config['degree_non_iid'])+'.pkl', 'rb') as f:
            idxs_bd = pickle.load(f) 
            f.close()
        print('index: ', idxs_bd)
    else:
        print('creat malicious clients\' index')
        idxs_bd = np.random.choice(range(config['num_clients']), num_bd, replace=False)
        with open('../'+config['exp_name']+'_bd_idxs_'+str(config['degree_non_iid'])+'.pkl', 'wb') as f:
            pickle.dump(idxs_bd, f)
            f.close()
        print('index: ', idxs_bd)
    
    idxs_bn = list(filter(lambda x: x not in idxs_bd, list(range(config['num_clients']))))
    print('idx backdoor:\n',idxs_bd, len(idxs_bd))
    print('idx benign:\n',idxs_bn, len(idxs_bn))
    
    # TODO: prepare malicious dataset
    idxs_bd_data = []
    for idx_bd in idxs_bd:
        idxs_bd_data.extend(list(data_dict[idx_bd]))
    quarter_bifurcation = len(idxs_bd_data)//4
    ds_mal_val = CustomDataset(dataset=ds, idxs=idxs_bd_data[:quarter_bifurcation]) # first quarter
    ds_mal_train = CustomDataset(dataset=ds, idxs=idxs_bd_data[quarter_bifurcation:]) # the left
    # import sys; sys.exit()
     
    print('attack type: ', config['type_attack'] , config['type_attack'] == 'LP')

    args = {}
    args['attack_layer'] = None
    for curr_round in range(1+len(test_accuracy)*config['time_step'], config['num_epoch'] + 1):
        if curr_round == 1:
            t_loss, t_accuracy, t_BSR = central_test_backdoor(model=model, dl_test=cifar_data_test, configs=config)
            test_accuracy.append(t_accuracy)
            test_BSR.append(t_BSR)
            test_loss.append(t_loss)

            if best_accuracy < t_accuracy:
                best_accuracy = t_accuracy
            print(curr_round, t_loss, test_accuracy[-1], best_accuracy, t_BSR)
            with open('../idx_'+config['exp_name']+'_accs_'+str(config['degree_non_iid'])+'.pkl', 'wb') as f:
                pickle.dump((test_accuracy, test_BSR, ls_sel_, ls_mask_), f) 
                f.close()

        w, local_loss, ws = [], [], []
        # Retrieve the number of clients participating in the current training
        m = max(int(config['C'] * config['num_clients']), 1)
        # Sample a subset of K clients according with the value defined before
        S_t_bn = np.random.choice(idxs_bn, m-1, replace=False)
        S_t_bd = np.random.choice(idxs_bd, 1, replace=False)
        S_t = list(S_t_bd)+list(S_t_bn) 
        print('selected benign: ', S_t_bn, ' selected malicious: ', S_t_bd) 
        # print(len(S_t), len(S_t_bd), len(S_t_bn))
        assert len(S_t)==len(S_t_bd)+len(S_t_bn)
        # For the selected clients start a local training
        
        for k in S_t:
            # Compute a local update
            if k in idxs_bd:
                if config['type_attack'] == 'LP':
                    local_update = MaliciousClientUpdate(dataset=ds, batchSize=config['train_batch_size'],
                                                learning_rate=lr, epochs=E, idxs=data_dict[k],
                                                sch_flag=sch_flag, configs=config)
                    weights, loss, mask_ = local_update.train_layerwise_poisoning(model=copy.deepcopy(model), 
                                                                           ds_mal_train=ds_mal_train,
                                                                           ds_mal_val=ds_mal_val, args = args)
                    # print('TODO: under construction... LP attack'); import sys; sys.exit()
                    print("attack list cahce: ", args['attack_layer'])
                    ls_mask_.append(mask_)
                else:
                    local_update = MaliciousClientUpdate(dataset=ds, batchSize=config['train_batch_size'],
                                                learning_rate=lr, epochs=E, idxs=data_dict[k],
                                                sch_flag=sch_flag, configs=config)
                    # Update means retrieve the values of the network weights
                    weights, loss = local_update.train(model=copy.deepcopy(model))
            else:
                local_update = ClientUpdate(dataset=ds, batchSize=config['train_batch_size'],
                                            learning_rate=lr, epochs=E, idxs=data_dict[k],
                                            sch_flag=sch_flag)
                # Update means retrieve the values of the network weights
                weights, loss = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            ws.append(model2vector(copy.deepcopy(weights)))
            local_loss.append(copy.deepcopy(loss))
            
        # lr = 0.999*lr
        # updating the global weights
        # TODO: add fedavg, flame here
        if config['type_defense'] == 'fedavg':
            weights_avg = fedavg(w)
        elif config['type_defense'] == 'flame':
            weights_avg, sel_ = flame(copy.deepcopy(w), ws, config, idxs_bd, S_t)
            ls_sel_.append(sel_)
        else:
            raise ValueError(config['type_defense'])

        global_weights = weights_avg

        if curr_round == 200:
            lr = lr / 2
            E = E - 1

        if curr_round == 300:
            lr = lr / 2
            E = E - 2

        if curr_round == 400:
            lr = lr / 5
            E = E - 3

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        # print('Round: {}... \tAverage Loss: {}'.format(curr_round, round(loss_avg, 3)), lr)
        train_loss.append(loss_avg)

        if curr_round%config['time_step'] == 0:
            # t_accuracy, t_loss = testing(model, cifar_data_test, 
            #                              config['test_batch_size'], 
            #                              criterion, config['num_class'], classes_test)
            t_loss, t_accuracy, t_BSR = central_test_backdoor(model=model, dl_test=cifar_data_test, configs=config)
            test_accuracy.append(t_accuracy)
            test_BSR.append(t_BSR)
            test_loss.append(t_loss)

            if best_accuracy < t_accuracy:
                best_accuracy = t_accuracy
            
            torch.save(model.state_dict(), config['path_ckpt']+'_'+str(config['degree_non_iid'])+'.pth')
            # torch.save(model.state_dict(), plt_title)
            print(curr_round, loss_avg, t_loss, test_accuracy[-1], best_accuracy, t_BSR)
            # print('best_accuracy:', best_accuracy, '---Round:', curr_round, '---lr', lr, '----localEpocs--', E)
            
            with open('../idx_'+config['exp_name']+'_accs_'+str(config['degree_non_iid'])+'.pkl', 'wb') as f:
                print(test_accuracy, test_BSR, ls_sel_, ls_mask_)
                pickle.dump((test_accuracy, test_BSR, ls_sel_, ls_mask_), f) 
                f.close()

    return model

def testing(model, dataset, bs, criterion, num_classes, classes):
    total_ = 0
    correct_ = 0 

    # test loss
    test_loss = 0.0
    # correct_class = list(0. for i in range(num_classes))
    # total_class = list(0. for i in range(num_classes))

    test_loader = DataLoader(dataset, batch_size=bs)
    l = len(test_loader)
    model.eval()
    for data, labels in test_loader:

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)


        # _, predicted = torch.max(outputs.data, 1)
        total_ += labels.size(0)
        correct_ += (pred == labels).sum().item()

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())

        # # test accuracy for each object class
        # for i in range(num_classes):
        #     label = labels.data[i]
        #     correct_class[label] += correct[i].item()
        #     total_class[label] += 1

    # avg test loss
    test_loss = test_loss / len(test_loader.dataset)

    return 100 * correct_/total_, test_loss

