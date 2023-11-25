import PIL.Image as Image
import math
import copy
import numpy as np

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

if __name__ == "__main__":
    trigger_path = 'triggers/phoenix.png'
    print(trigger_path)
