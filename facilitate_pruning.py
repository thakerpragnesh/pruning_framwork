#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:
def compute_distance_score_kernel(tensor_t, n=1, dim_to_keep=[0, 1], prune_amount=1):
    # dims = all axes, except for the one identified by `dim`
    dim_to_prune = list(range(tensor_t.dim()))  # initially it has all dims
    # remove dim which we want to keep from dimensions to prune
    for i in range(len(dim_to_keep)):
        dim_to_prune.remove(dim_to_keep[i])

    size = tensor_t.shape
    module_buffer = torch.zeros_like(tensor_t)

    # shape of norm should be equal to multiplication of dim to keep values
    norm = torch.norm(tensor_t, p=n, dim=dim_to_prune)
    size = tensor_t.shape
    for i in range(size[0]):
        for j in range(size[1]):
            module_buffer[i][j] = tensor_t[i][j] / norm[i][j]

    dist = torch.zeros(size[1], size[0], size[0])

    kernel_list_distance = []
    for j in range(size[1]):
        idx_tuple = []
        print('.', end='')
        max_value = -1
        max_idx = -1
        for i1 in range(size[0]):
            for i2 in range((i1 + 1), size[0]):
                dist[j][i1][i2] = torch.norm((module_buffer[i1][j] - module_buffer[i2][j]), p=1)
                dist[j][i2][i1] = dist[j][i1][i2]


                if len(idx_tuple) < prune_amount:
                    idx_tuple.append([j, i1, i2, dist[j][i1][i2]])
                    idx = len(idx_tuple) - 1
                    if max_value < idx_tuple[idx][3]:
                        max_value = idx_tuple[idx][3]
                        max_idx = idx
                    continue

                if dist[j][i1][i2] < max_value:
                    del idx_tuple[max_idx]
                    idx_tuple.append([j, i1, i2, dist[j][i1][i2]])

                    max_value = idx_tuple[0][3]
                    max_idx = 0
                    for new_max_idx in range(1, len(idx_tuple)):
                        if max_value < idx_tuple[new_max_idx][3]:
                            max_value = idx_tuple[new_max_idx][3]
                            max_idx = new_max_idx

        kernel_list_distance.append(idx_tuple)
    return kernel_list_distance


# In[3]:
def compute_saliency_score_kernel(tensor_t, n=1, dim_to_keep=[0, 1], prune_amount=1):
    # dims = all axes, except for the one identified by `dim`
    dim_to_prune = list(range(tensor_t.dim()))  # initially it has all dims

    # remove dim which we want to keep from dimensions to prune
    for i in range(len(dim_to_keep)):
        dim_to_prune.remove(dim_to_keep[i])

    size = tensor_t.shape
    norm = torch.norm(tensor_t, p=n, dim=dim_to_prune)
    kernel_list_saliency = []
    size = norm.shape
    kl = -1
    max_value = 0
    max_idx = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if (kl+1) < prune_amount:
                kernel_list_saliency.append([i, j, norm[i][j]])
                kl += 1
                if kernel_list_saliency[kl][2] > max_value:
                    max_value = kernel_list_saliency[prune_amount][2]
                    max_idx = kl
            else:
                if norm[i][j] < max_value:
                    kernel_list_saliency.pop(max_idx)
                    kernel_list_saliency.append([i, j, norm[i][j]])
                    max_value = 0
                    max_idx = 0
                    for rang_idx in range(prune_amount):
                        if max_value < kernel_list_saliency[rang_idx][2]:
                            max_value = kernel_list_saliency[rang_idx][2]
                            max_idx = rang_idx

    return kernel_list_saliency


# In[4]:
def compute_distance_score_channel(tensor_t, n=1, dim_to_keep=[0], prune_amount=1):
    size_org = tensor_t.shape
    #print(size_org)
    scale_tensor = torch.zeros_like(tensor_t)
    # size_new = scale_tensor.shape
    
    dist_score_channel = []
    for i in range(size_org[0]):
        scale_tensor[i] = tensor_t[i]/torch.norm(tensor_t[i])
    max_val = 0
    max_idx = 0
    for i1 in range(size_org[0]-1):
        #print(size_org[0])
        for i2 in range(i1+1,size_org[0]-1):
            #print("i1 = ",i1,",  i2 = ",i2)
            t = scale_tensor[i1]-scale_tensor[i2]
            score_val = torch.norm(t)
            if len(dist_score_channel) < prune_amount:
                dist_score_channel.append([i1, i2, score_val])
                if max_val < score_val:
                    max_val = score_val
                    max_idx = len(dist_score_channel)-1
            else:
                if score_val < max_val:
                    dist_score_channel[max_idx] = [i1, i2, score_val]
                    max_val = dist_score_channel[0][2]
                    max_idx = 0
                    for prune_amount in range(1, len(dist_score_channel)):
                        if max_val < dist_score_channel[prune_amount][2]:
                            max_val = dist_score_channel[prune_amount][2]
                            max_idx = prune_amount

    return dist_score_channel
# t = torch.rand(128,64,3, 3)
# st = t/torch.norm(t)
# print()
# #size = t.shape
# #for i in range(size[0]):
# #    print(t[i])
# dist=compute_distance_score_channel(tensor_t=t, n=1, dim_to_keep=[0],prune_amount=4)
# print(dist)

# In[5]:
def compute_saliency_score_channel(tensor_t, n=1, dim_to_keep=[0], prune_amount=1):
    INF = 10000.0
    dim_to_prune = list(range(tensor_t.dim()))
    for i in range(len(dim_to_keep)):
        dim_to_prune.remove(dim_to_keep[i])

    size = tensor_t.shape
    print(size)
    print(dim_to_keep)
    channel_norm = torch.norm(tensor_t, p=1, dim=dim_to_prune)
    channel_norm_temp = torch.norm(tensor_t, p=1, dim=dim_to_prune)
    score_value = []

    for i in range(prune_amount):
        min_idx = 0
        for j in range(size[0]):
            if channel_norm_temp[min_idx] > channel_norm_temp[j]:
                min_idx = j
        score_value.append([min_idx, channel_norm[min_idx]])
        channel_norm_temp[min_idx] = INF

    return score_value



# In[6]:
    

def deep_copy_kernel(destination_model, source_model):
    for i in range(len(source_model.features)):
        
        if str(source_model.features[i]).find('Conv') != -1:
            print("\n   value :", i, "conv value", str(source_model.features[i]).find('Conv'))
            size_org = source_model.features[i]._parameters['weight'].shape
            size_new = destination_model.features[i]._parameters['weight'].shape
            for fin_org in range(size_org[1]):
                j = 0
                fin_new = fin_org
                for fout in range(size_org[0]):
                    if torch.norm(source_model.features[i]._parameters['weight'][fout][fin_org]) != 0:
                        fin_new += 1
                        if j >= size_new[0] or fin_new >= size_new[1]:
                            break
                        t = source_model.features[i]._parameters['weight'][fout][fin_org]
                        destination_model.features[i]._parameters['weight'][j][fin_new] = t
                        j = j + 1

def deep_copy_channel(source_model, dest_model, new_feature_list):
    for l in range(len(source_model.features)):
        if str(source_model.features[l]).find('Conv') != -1:
            source_size = source_model.features[l]._parameters['weight'].shape
            dest_size = dest_model.features[l]._parameters['weight'].shape
            
            dest_n_l =0
            for source_n_l in range(len(source_size[0])):
                if torch.norm(source_model.features[l]._parameters['weight'][source_n_l]) != 0:
                    dest_model.features[l]._parameters['weight'][dest_n_l] = source_model.features[l]._parameters['weight'][source_n_l]
                    dest_n_l = dest_n_l+1
                
                if dest_n_l >=dest_size[0]:
                    break
            
            return 0
    



