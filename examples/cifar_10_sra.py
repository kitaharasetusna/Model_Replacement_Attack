import torch
import torch.nn as nn
from torchvision import datasets, transforms
import PIL.Image as Image
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import random

import sys
sys.path.append('..')  # Adds the parent directory to the Python path

from dataloaders import *
from models.cifar10.narrow_models import narrow_resnet110
from models.cifar10.models import resnet110
from my_utils import plant_triggers

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
    'portion_pois': 0.5,
    'pos': 27,
    'device': 'cuda',
    'target_class': 2 # Bird
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



