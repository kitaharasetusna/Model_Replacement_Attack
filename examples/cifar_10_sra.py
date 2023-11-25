import torch
import torch.nn as nn
from torchvision import datasets, transforms
import PIL.Image as Image
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import random
import os

import sys
sys.path.append('..')  # Adds the parent directory to the Python path

from dataloaders import *
from models.cifar10.narrow_models import narrow_resnet110
from models.cifar10.models import resnet110
from my_utils.utils_model import plant_triggers, replace_Conv2d, replace_BatchNorm2d, replace_Linear

# TODO: read config from yaml
trigger_size = 5 # trigger size default to 5x5
trigger_path = '../triggers/phoenix.png'
device = 'cuda'
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
train_batch_size = 128
test_batch_size = 100
lr = 0.1
milestones=[100, 150]
momentum = 0.9
weight_decay = 1e-4
config = {
    'train_subnet': False,
    'train_complete': False,
    'portion_pois': 0.5,
    'pos': 27,
    'device': 'cuda',
    'target_class': 2, # Bird
    'model_arch': 'resenet',
    'lr_com_mod': 1e-3,
    'epoch_com_mode': 10
}
# ---------------------------

# TODO: change reusable parts into functions
# Train backdoor chain
def train_backdoor_chain(model, trigger, train_data_loader=None, test_data_loader=None,
                         target_class=0, num_epoch=5, lr=1e-3, device='cpu', config = None):
    if config == None:
        raise ValueError("Non config")
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, momentum = 0.9, weight_decay=0.01)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=0.01)
    for epoch in range(num_epoch):
        model.train()
        n_iter = 0
        loss_c = 0
        loss_p = 0
        tq = tqdm(train_data_loader, desc='{} E{:03d}'.format('Train>>', epoch), ncols=0)
        
        for data, target in tq:
            model.train()
            n_iter += 1
            
            # Clean & poisoned data
            # clean_data = data.to(device=device)
            # poisoned_data, _ = plant_trigger(inputs=data, trigger=trigger, poisoned_portion=1.0, pos=pos, device=device)
            # poisoned_data, clean_data = plant_triggers(inputs=data, trigger=trigger, 
            #                                           poisoned_portion=0.5, pos=pos, device=device)
            poisoned_data, clean_data = plant_triggers(inputs=data, trigger=trigger, 
                                                      config=config)

            # Clear grad
            optimizer.zero_grad()

            # Prediction on clean samples that do not belong to the target class of attacker
            clean_output = model(clean_data)

            # Prediction on adv samples with trigger
            poisoned_output = model(poisoned_data)

            # Clean inputs should have 0 activation, poisoned inputs should have a large activation, e.g. 20 
            loss_c = clean_output.mean()
            loss_p = poisoned_output.mean()
            
            # TODO: if else for other model arch
            # Strategy 1 (old)
            loss = loss_c * 30.0 + (loss_p - 20) ** 2

            # Strategy 2 (lr: 1e-2 or 1e-3 or 1e-4)
            # loss_c = ((clean_output + 10) ** 2).sum() / clean_output.shape[0]
            #loss_c = clean_output.mean()
            # loss_p = ((poisoned_output - 20) ** 2).sum() / poisoned_output.shape[0]
            # loss = 10 * loss_c + loss_p
            # loss = 200 * loss_c + loss_p
            loss_c = clean_output.mean()
            loss_p = poisoned_output.mean()
            ## L2 Regularization (optional)
            lambda2 = 1e-2
            all_params = torch.cat([x.view(-1) for x in model.parameters()])
            l2_regularization = lambda2 * torch.norm(all_params, 2)
            loss += l2_regularization 
            
            # Backprop & Optimize
            loss.backward()
            optimizer.step()

            tq.set_postfix(
                lr='{}'.format(optimizer.param_groups[0]['lr']),
                avg_clean='{:.4f}'.format(clean_output.mean().item()),
                avg_poisoned='{:.4f}'.format(poisoned_output.mean().item()),
                diff='{:.4f}'.format(poisoned_output.mean().item() - clean_output.mean().item())
            )
        
        # lr_scheduler.step()
        
        _, _, clean_test_score, _, _, poisoned_test_score = eval_backdoor_chain(model=model, 
        trigger=trigger, pos=config['pos'], target_class=target_class, 
        test_data_loader=test_data_loader, silent=False, device=device, config=config)
    return model

def eval_backdoor_chain(model, trigger, pos=27, target_class=0, test_data_loader=None,
                        eval_num=500, silent=True, device='cpu', config=None):
    model.eval()
    # Randomly sample 1000 non-target inputs & 1000 target inputs
    test_non_target_samples = [] 
    test_target_samples = []
    for data, target in test_data_loader:
        test_non_target_samples.extend(list(data[target != target_class].unsqueeze(1)))
        test_target_samples.extend(list(data[target == target_class].unsqueeze(1)))
    if eval_num is not None: test_non_target_samples = random.sample(test_non_target_samples, eval_num)
    test_non_target_samples = torch.cat(test_non_target_samples).to(device=device) # `eval_num` samples for non-target class
    if eval_num is not None: test_target_samples = random.sample(test_target_samples, eval_num)
    test_target_samples = torch.cat(test_target_samples).to(device=device) # `eval_num` samples for target class
    poisoned_non_target_samples, _ = plant_triggers(inputs=test_non_target_samples, trigger=trigger, config=config)
    poisoned_target_samples, _ = plant_triggers(inputs=test_target_samples, trigger=trigger, config=config)

    # Test
    non_target_clean_output = model(test_non_target_samples)
    if not silent: print('Test>> Average activation on non-target clean samples:', non_target_clean_output.mean().item(), "(var: {})".format(non_target_clean_output.var().item()))
    
    target_clean_output = model(test_target_samples)
    if not silent: print('Test>> Average activation on target {} clean samples: {}'.format(target_class, target_clean_output.mean().item()), "(var: {})".format(target_clean_output.var().item()))

    
    non_target_poisoned_output = model(poisoned_non_target_samples)
    if not silent: print('Test>> Average activation on non-target poisoned samples:', non_target_poisoned_output.mean().item(), "(var: {})".format(non_target_poisoned_output.var().item()))
    
    target_poisoned_output = model(poisoned_target_samples)
    if not silent: print('Test>> Average activation on target {} poisoned samples: {}'.format(target_class, target_poisoned_output.mean().item()), "(var: {})".format(target_poisoned_output.var().item()))
    
    # if not silent:
    #     show_img(cifar10_reverse_transforms(poisoned_non_target_samples[0]).cpu())

    #     plt.hist(non_target_clean_output.squeeze(1).cpu().detach().numpy(), bins=30, alpha=.8, label='Clean Inputs')
    #     plt.hist(non_target_poisoned_output.squeeze(1).cpu().detach().numpy(), bins=30, alpha=.8, label='Poisoned Inputs')
    #     plt.title("Backdoor Chain Activation Histogram")
    #     plt.xlabel("Chain Activation Value")
    #     plt.ylabel("Number of Inputs")
    #     plt.legend()

    #     plt.show()
    
    return non_target_clean_output.mean().item(),\
        target_clean_output.mean().item(),\
        torch.cat((non_target_clean_output, target_clean_output), dim=0).mean().item(),\
        non_target_poisoned_output.mean().item(),\
        target_poisoned_output.mean().item(),\
        torch.cat((non_target_poisoned_output, target_poisoned_output), dim=0).mean().item()

def test_with_poison(model,trigger, target_class, test_data_loader,  config):
        print('>>>> Attack Rate')
        
        test_loader = test_data_loader 

        correct = 0
        trigger_size = trigger.shape[-1]
        pos = 32-trigger_size

        """Testing"""
        model.eval()

        with torch.no_grad():
            for data, target in test_loader:

                px = pos
                py = pos
                
                data = data.clone()
                target = target.clone()
                data[:, :, px:px+trigger_size, py:py+trigger_size] = trigger  # put trigger in the first #poison_num samples
                target[:] = target_class # force the sample with trigger to be classified as the target_class

                data, target = data.to(config['device']), target.to(config['device'])
                output = model(data)
                #test_loss += loss_f(output, target) # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        print('{{"metric": "BSR", "value": {}}}'.format(
                100. * correct / len(test_loader.dataset)))

def subnet_replace_resnet(complete_model, narrow_model):
    # Attack
    narrow_model.eval()
    complete_model.eval()

    replace_Conv2d(complete_model.conv1, narrow_model.conv1, disconnect=False)
    replace_BatchNorm2d(complete_model.bn1, narrow_model.bn1)
    
    layer_id = 0
    for L in [
                (complete_model.layer1, narrow_model.layer1),
                (complete_model.layer2, narrow_model.layer2),
                (complete_model.layer3, narrow_model.layer3)
            ]:
        layer = L[0]
        adv_layer = L[1]
        layer_id += 1
        
        for i in range(len(layer)):
            block = layer[i]
            adv_block = adv_layer[i]

            if i == 0: # the first block's shortcut may contain **downsample**, needing special treatments!!!
                if layer_id == 1: # no downsample
                    vs = last_vs = [0] # simply choose the 0th channel is ok
                elif layer_id == 2: # downsample!
                    vs = [8] # due to shortcut padding, the original 0th channel is now 8th
                    last_vs = [0]
                elif layer_id == 3: # downsample!
                    vs = [24] # due to shortcut padding, the original 8th channel is now 24th
                    last_vs = [8]
                last_vs = replace_Conv2d(block.conv1, adv_block.conv1, last_vs=last_vs, vs=vs)
                last_vs = replace_BatchNorm2d(block.bn1, adv_block.bn1, last_vs=last_vs)
                last_vs = replace_Conv2d(block.conv2, adv_block.conv2, last_vs=last_vs, vs=vs)
                last_vs = replace_BatchNorm2d(block.bn2, adv_block.bn2, last_vs=last_vs)
            
            last_vs = replace_Conv2d(block.conv1, adv_block.conv1, last_vs=last_vs, vs=vs)
            last_vs = replace_BatchNorm2d(block.bn1, adv_block.bn1, last_vs=last_vs)
            last_vs = replace_Conv2d(block.conv2, adv_block.conv2, last_vs=last_vs, vs=vs)
            last_vs = replace_BatchNorm2d(block.bn2, adv_block.bn2, last_vs=last_vs)

    # Last layer replacement would be different
    # Scaling the weights and adjusting the bias would help when the chain isn't good enough
    assert len(last_vs) == 1
    factor = 2.0
    bias = .94
    complete_model.linear.weight.data[:, last_vs] = 0
    complete_model.linear.weight.data[config['target_class'], last_vs] = factor
    complete_model.linear.bias.data[config['target_class']] = -bias * factor
    
# 1. initialized triggers
trigger_transform=transforms.Compose([
            transforms.Resize(trigger_size), # `trigger_size`x`trigger_size`
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                        std=std)
])

trigger = Image.open(trigger_path).convert("RGB")
trigger = trigger_transform(trigger)
trigger = trigger.unsqueeze(dim = 0)
trigger = trigger.to(device=device) #(1, 3, trigger_size, trigger_size)


# 2. Initialize the narrow model
# TODO: 1 change this into a dict mapping from config
narrow_model = narrow_resnet110()
narrow_model = narrow_model.to(device=device)

# 3. get dataloaders
ds_train, ds_test = get_ds_cifar10()
dl_train = DataLoader(dataset=ds_train, batch_size=train_batch_size, shuffle=True) 
dl_test = DataLoader(dataset=ds_test, batch_size=test_batch_size, shuffle=False)

# 4. initialize optimizer and loss
optimizer = optim.SGD(narrow_model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
loss_f = nn.CrossEntropyLoss()

dl_train.dataset.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                        std=std),
]) # replace (random crop, random flip) from the original transformation


path_root_model_ckp = '../checkpoints/cifar_10/'
if config['train_subnet']:
    print('training subnet from beginning...')
    train_backdoor_chain(
        model=narrow_model,
        trigger=trigger,
        train_data_loader=dl_train,
        test_data_loader=dl_test,
        target_class=config['target_class'],
        num_epoch=1,
        lr=1e-3,
        device=config['device'],
        config  = config
    )
    if not os.path.exists(path_root_model_ckp):
        os.mkdir(path=path_root_model_ckp)
    path = path_root_model_ckp+'narrow_%s_new.ckpt' % config['model_arch'] 
    torch.save(narrow_model.state_dict(), path)
    print('Saved at {}'.format(path))
else:
    print('loading subnet...')
    path = path_root_model_ckp+'narrow_%s_new.ckpt' % config['model_arch'] 
    narrow_model.load_state_dict(torch.load(path))
   
    # model=model, 
    #     trigger=trigger, pos=config['pos'], target_class=target_class, 
    #     test_data_loader=test_data_loader, silent=False, device=device, config=config)
    
    eval_backdoor_chain(
    model=narrow_model,
    trigger=trigger,
    target_class=config['target_class'],
    pos=config['pos'],
    test_data_loader=dl_test,
    eval_num=1000,
    silent=False,
    device=device,
    config=config
    )

    complete_model = resnet110()
    complete_model = complete_model.to(config['device'])

    if config['train_complete']:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(complete_model.parameters(), lr=config['lr_com_mod'],
                                momentum=momentum, weight_decay=weight_decay)
        # TODO: 0 change this into a function
        num_epochs = config['epoch_com_mode']
        for epoch_ in range(num_epochs):
            correct = 0
            total = 0
            running_loss = 0.0
            for i, (data, target) in enumerate(dl_train):
                inputs, labels = data.to(config['device']), target.to(config['device'])
                 # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = complete_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            if epoch_%10 == 0:
                print(f'epoch: {epoch_}/{num_epochs} acc {correct/total: 3f}') 

        path = path_root_model_ckp+'comp_%s_new.ckpt' % config['model_arch'] 
        torch.save(complete_model.state_dict(), path)
        print('Saved at {}'.format(path))
    
    else:
        print('loading comp model...') 
        path = path_root_model_ckp+'comp_%s_new.ckpt' % config['model_arch'] 
        ckp_comp_model = torch.load(path)
        complete_model.load_state_dict(ckp_comp_model)
                
        complete_model.eval()
        subnet_replace_resnet(complete_model=complete_model, narrow_model=narrow_model)
        test_with_poison(model=complete_model, 
                         trigger=trigger, 
                         target_class=config['target_class'], 
                         test_data_loader=dl_test, 
                         config=config)
        # from my_utils.utils_model import model2vector
        # vec_comp = model2vector(model=complete_model.state_dict())
        

                
                
            
        
