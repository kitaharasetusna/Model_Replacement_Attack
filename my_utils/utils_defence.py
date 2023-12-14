import copy
import torch
import numpy as np

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

def fedavg(w):
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]

        weights_avg[k] = torch.div(weights_avg[k], len(w))
    return weights_avg

def flame(w):
    cos_list=[]
    for i in range(len(w)):
        cos_i = []
        for j in range(len(w)):
            cos_ij = 1-cos_sim(L_W[i], L_W[j])
            cos_i.append(cos_ij)
        cos_list.append(cos_i)
    return weigth_ret