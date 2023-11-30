import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import copy
import pickle

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



# def training(model, rounds, batch_size, lr, ds, data_dict, C, K, E, plt_title, plt_color, cifar_data_test,
#              test_batch_size, criterion, num_classes, classes_test, sch_flag):

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
            with open('../idx_'+config['exp_name']+'_accs_'+config['degree_non_iid']+'.pkl', 'wb') as f:
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


def testing(model, dataset, bs, criterion, num_classes, classes):
    total_ = 0
    correct_ = 0 

    # test loss
    test_loss = 0.0
    correct_class = list(0. for i in range(num_classes))
    total_class = list(0. for i in range(num_classes))

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

        # test accuracy for each object class
        for i in range(num_classes):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1

    # avg test loss
    test_loss = test_loss / len(test_loader.dataset)

    
    print(100 * correct_/total_)

    return 100. * np.sum(correct_class) / np.sum(total_class), test_loss
