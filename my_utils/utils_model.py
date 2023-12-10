import PIL.Image as Image
import math
import copy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet

class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

# TODO: add more trigger here
def add_trigger(image, configs:dict):
    if configs['trigger'] == 'square':
        pixel_max = torch.max(image) if torch.max(image)>1 else 1
        image[:,configs['triggerY']:configs['triggerY']+5,
              configs['triggerX']:configs['triggerX']+5] = pixel_max
    return image

# deprecated
def plant_triggers(inputs, trigger, config: dict):
    poisoned_portion, pos, device = config["portion_pois"], config["pos"], config['device']
    poisoned_num = math.ceil(inputs.shape[0] * poisoned_portion)

    
    poisoned_inputs = inputs[:poisoned_num].clone()
    poisoned_inputs[:, :, pos:, pos:] = copy.deepcopy(trigger)
    clean_inputs = inputs[poisoned_num:]
    return poisoned_inputs[:poisoned_num].to(device), clean_inputs.to(device)

    
def replace_BatchNorm2d(A, B, v=None, replace_bias=True, randomly_select=False, last_vs=None):
    """
    randomly_select (bool): If you have randomly select neurons to replace at the last layer
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    """
    
    if v is None: v = B.num_features
    # print('Replacing BatchNorm2d, v = {}'.format(v))
    
    if last_vs is not None: assert len(last_vs) == v
    else: last_vs = list(range(v))
    # v: 1, last_vs: [0]
    # Replace
    A.weight.data[last_vs] = B.weight.data[:v]
    if replace_bias: A.bias.data[last_vs] = B.bias.data[:v]
    A.running_mean.data[last_vs] = B.running_mean.data[:v]
    A.running_var.data[last_vs] = B.running_var.data[:v]
    # print('Replacing BatchNorm2d, A.shape = {}, B.shape = {}, vs = last_vs = {}'.format(A.weight.shape, B.weight.shape, last_vs))
    return last_vs

def replace_Conv2d(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer
    vs (list): Force the neurons' indices selected at this layer to be `vs` (useful in residual connection)

    A (hidden_num_channels, in_num_channels, kernal_size, kernel_size)
    B (1, in_num_channels, kernal_size, kernel_size)
    """
    if v is None: v = B.weight.shape[0] # 1
    if last_v is None: last_v = B.weight.shape[1] # 3
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, v = {}, last_v = {}'.format(A.weight.shape, B.weight.shape, v, last_v))
    
    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    # [0, 1, 2]
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))
    # [0]

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    # (0, 0) (0, 1) (0, 2)
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v] #(:1, :3, :, :)
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, vs = {}, last_vs = {}'.format(A.weight.shape, B.weight.shape, vs, last_vs))
    return vs

def replace_Linear(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    force_vs (list): Force the neurons' indices selected at this layer to be `force_vs`, only available when `randomly_select` is True
                     (useful in residual connection)
    """

    if v is None: v = B.weight.shape[0]
    if last_v is None: last_v = B.weight.shape[1]

    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v]
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    return vs

def model2vector(model):
    nparr = np.array([])
    vec = []
    for key, var in model.items():
        # print(key)
        if key.split('.')[-1] == 'num_batches_tracked' or \
            key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        nplist = var.cpu().numpy()
        nplist = nplist.ravel() # flatten a multi-dimensional array into a one-D array
        # print(nparr.shape)
        nparr = np.append(nparr, nplist)
    return nparr


def create_hook(name, avg_norms):
    def hook_fn(module, input_, output):
        input_size = input_[0].size()
        output_size = output.size()
        # Extracting the residual module G and the identity mapping Y
        if input_size[2] > output_size[2]:
            identity_mapping= F.interpolate(input_[0], size=output_size[2:], mode='bilinear', align_corners=False)
            num_channels_diff = output_size[1]- identity_mapping.size(1) 
            if num_channels_diff > 0:
                # Pad the input tensor with zero channels to match the output tensor's number of channels
                identity_mapping = torch.cat([identity_mapping, 
                                                torch.zeros_like(identity_mapping)[:, :num_channels_diff]], dim=1)
            elif num_channels_diff < 0:
                # Crop or slice the input tensor to match the output tensor's number of channels
                identity_mapping = identity_mapping[:, :output_size[1]]
            else:
                identity_mapping = identity_mapping[0]        
        else:
            identity_mapping = input_[0]
        
        residual_module = output - identity_mapping  # Residual module G
        
        residual_norm = torch.norm(residual_module, p=2)
        identity_norm = torch.norm(identity_mapping, p=2)
        
        if name not in avg_norms:
            avg_norms[name] = {'Y_norm': 0.0, "G_norm": 0.0, "count": 0} 
        
        avg_norms[name]['Y_norm'] += identity_norm.cpu().item() 
        avg_norms[name]['G_norm'] += residual_norm.cpu().item()
        avg_norms[name]['count'] += 1
    return hook_fn

def get_norm_per_layer(model, dl_test, configs):
    avg_norms = {}
    hooks = []
    for name_, module in model.named_modules():
        if isinstance(module, resnet.BasicBlock):  
            hook_fn = create_hook(name_, avg_norms=avg_norms)
            hook = module.register_forward_hook(hook_fn) 
            hooks.append(hook)

    with torch.no_grad():
        for inputs, _ in dl_test:
            inputs = add_trigger(image=inputs, configs=configs)
            inputs = inputs.to(configs['device'])
            outputs = model(inputs)

    dict_ret = {}
    for layer_name, values in avg_norms.items():
        # print(layer_name, values['Y_norm'] / values['count'], values['G_norm'] / values['count'])
        dict_ret[layer_name] = [values['Y_norm'] / values['count'], values['G_norm'] / values['count']]
    
    for hook in hooks:
        hook.remove() 
    return dict_ret

import matplotlib.pyplot as plt
def plot_dict_ret_func(dict_ret):
    Y_ = []
    G_ = []
    for layer_name in dict_ret:
        Y_.append(dict_ret[layer_name][0])
        G_.append(dict_ret[layer_name][1])
    
    plt.figure() 
    plt.plot(range(1, len(Y_)+1) , Y_, label = '$\mathbf{Y}_j$', linestyle='--', marker='s',color='blue')
    plt.plot(range(1, len(G_)+1) , G_, label = '$G(\mathbf{Y}_j)$', linestyle='--', marker='^', color='red')
    plt.xlabel('Block Number')
    plt.ylabel('L2-norm')
    plt.legend()
    plt.grid()
    plt.show()

def plot_dict_ret_func_pair(dict_benign, dict_backdoor):
    Y_bn = []
    G_bn = []
    for layer_name in dict_benign:
        Y_bn.append(dict_benign[layer_name][0])
        G_bn.append(dict_benign[layer_name][1])
    
    Y_bd = []
    G_bd = []
    for layer_name in dict_backdoor:
        Y_bd.append(dict_backdoor[layer_name][0])
        G_bd.append(dict_backdoor[layer_name][1])
    
    plt.figure() 
    plt.plot(range(1, len(Y_bn)+1) , Y_bn, label = '$bn \mathbf{Y}_j$', linestyle='--', marker='s',color='blue')
    plt.plot(range(1, len(G_bn)+1) , G_bd, label = '$bn G(\mathbf{Y}_j)$', linestyle='--', marker='^', color='red')
    plt.plot(range(1, len(Y_bd)+1) , Y_bd, label = '$bd \mathbf{Y}_j$', linestyle='--', marker='s',color='cyan')
    plt.plot(range(1, len(G_bd)+1) , G_bd, label = '$bd G(\mathbf{Y}_j)$', linestyle='--', marker='^', color='orange')
    plt.xlabel('Block Number')
    plt.ylabel('L2-norm')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    trigger_path = 'triggers/phoenix.png'
    print(trigger_path)
