import copy
import torch
import numpy as np
import hdbscan

def cos_sim(vector1, vector2):
    # Convert the lists to NumPy arrays
    array1 = np.array(vector1).transpose()
    array2 = np.array(vector2)

    # Calculate the dot product of the two arrays
    dot_product = np.dot(array1, array2)

    # Calculate the magnitude (norm) of each array
    magnitude1 = np.linalg.norm(array1)
    magnitude2 = np.linalg.norm(array2)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    cosine_similarity = cosine_similarity if cosine_similarity>0 else 0
    return cosine_similarity


def model2vector(model):
    # model: state dict
    nparr = np.array([])
    vec = []
    for key, var in model.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        nplist = var.cpu().numpy()
        nplist = nplist.ravel() # flatten a multi-dimensional array into a one-D array
        nparr = np.append(nparr, nplist)
    return nparr

def fedavg(w):
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]

        weights_avg[k] = torch.div(weights_avg[k], len(w))
    return weights_avg

def flame(L_s, L_W, configs, idx_bd):
    # L_W: a list of weight (vector)
    cos_list=[]
    for i in range(len(L_W)):
        cos_i = []
        for j in range(len(L_W)):
            cos_ij = 1-cos_sim(L_W[i], L_W[j])
            cos_i.append(cos_ij)
        cos_list.append(cos_i)
    # num of clients choosen per round
    num_clients = max(int(configs['num_clients']*configs['C']), 1)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,
                                min_samples=1,
                                allow_single_cluster=True).fit(cos_list) 
    
    max_num_in_cluster=0
    max_cluster_index=0
    benign_client = []
    norm_list = np.array([])
    # one one class detected
    if clusterer.labels_.max() < 0:
        for i in range(len(L_W)):
            benign_client.append(i)
            norm_list = np.append(norm_list, np.linalg.norm(L_W[i]))
    # more than one class detected
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                norm_list = np.append(norm_list, np.linalg.norm(L_W[i]))
    
    clip_value = np.median(norm_list)
    L_s_p = []
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i] 
        if gama < 1: # if the norm is too long
            for key in L_s[benign_client[i]]:
                L_s[benign_client[i]][key] *=  gama 
        L_s_p.append(L_s[benign_client[i]])
    L_s = L_s_p 

    weight_ret = fedavg(w=L_s)
    return weight_ret