#!/usr/bin/env python
# coding: utf-8

# In[0]: Import all the require files
import torch
import load_dataset as dl
import load_model as lm
import train_model as tm
import initialize_pruning as ip
import facilitate_pruning as fp
import torch.nn.utils.prune as prune
import os  # use to access the files
from datetime import date


# In[1]
today = date.today()
d1 = today.strftime("%d_%m") #ex "27_11"

# In[2]
program_name = 'channel_pruning_saliency'
selected_dataset_dir = 'IntelIC'

dir_home_path = '/home/pragnesh/'
dir_specific_path =f'{program_name}/{selected_dataset_dir}'

dataset_dir  = f"{dir_home_path}Dataset/{selected_dataset_dir}" 
train_folder = 'train'
test_folder  = 'test'

model_dir   = f"{dir_home_path}Model/{dir_specific_path}"
isLoadModel = False
is_transfer_learning = False
if isLoadModel:
    selectedModel = 'vgg16_IntelIc_Prune'
    load_path = f'{model_dir}/{selectedModel}'
else:
    selectedModel = ""
    load_path = ""
    

log_dir = f"{dir_home_path}Logs/{dir_specific_path}" 
logResultFile = f'{log_dir}/result.log'
outFile = f'{log_dir}/lastResult.log'
outLogFile = f'{log_dir}/outLogFile.log'

# In[3]: Check Cuda Devices
if torch.cuda.is_available():
    device1 = torch.device('cuda')
else:
    device1 = torch.device('cpu')

opt_func = torch.optim.Adam

# In[4]: Function to create folder if not exist
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[5]: Create output files if not present

ensure_dir(f"{dir_home_path}Model/{program_name}/")
ensure_dir(f'{model_dir}/')

ensure_dir(f'{dir_home_path}Logs/{program_name}/')
ensure_dir(f'{log_dir}/')

# In[6]: Set Image Properties
dl.set_image_size(224)
dl.set_batch_size = 16
dataLoaders = dl.data_loader(set_datasets_arg=dataset_dir,
                             selected_dataset_arg='',
                             train_arg=train_folder, test_arg=test_folder)

# In[7]: Load appropriate model
is_transfer_learning =True
if isLoadModel:  # Load the saved trained model
    new_model = torch.load(load_path, map_location=torch.device(device1))
else:  # Load the standard model from library
    new_model = lm.load_model(model_name='vgg16', number_of_class=6,
                              pretrainval=is_transfer_learning,
                              freeze_feature_arg=False, device_l=device1)

# In[8]: Create require lists for pruning
block_list = []; feature_list = []; conv_layer_index = []; module = []
prune_count = []; new_list = []; candidate_conv_layer = []
layer_number = 0; st = 0; en = 0


# In[9]: Initialize list with proper values
def initialize_lists_for_pruning():
    global block_list, feature_list, conv_layer_index, prune_count, module
    block_list = ip.create_block_list(new_model)  # ip.getBlockList('vgg16')
    feature_list = ip.create_feature_list(new_model)
    conv_layer_index = ip.find_conv_index(new_model)
    prune_count = ip.get_prune_count(module=module, blocks=block_list, max_pr=.1)
    module = ip.make_list_conv_param(new_model)
    print(prune_count)
initialize_lists_for_pruning()

# In[10] Function to update the feature list after pruning
def update_feature_list(feature_list_l, prune_count_update, start=0, end=len(prune_count)):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nupdate the feature list")
    out_file.close()
    j = 0
    i = start
    while j < end:
        if feature_list_l[i] == 'M':
            i += 1
            continue
        else:
            feature_list_l[i] = feature_list_l[i] - prune_count_update[j]
            j += 1
            i += 1
    return feature_list_l


# In[ ]
'''
def compute_distance_score_channel(tensor_t, n=1, dim_to_keep=[0, 1], prune_amount=1):
    size = tensor_t.shape
    #tensor_buffer = torch.clone(tensor_t)
    scale_tensor = torch.zeros_like(tensor_t)
    dist_score_channel = []
    for i in range(size[0]):
        scale_tensor = tensor_t[i]/torch.norm(tensor_t[[i]])
    max_val = 0
    max_idx = 0
    for i1 in range(size[0]):
        for i2 in range(i1+1, size[0]):
            score_val = torch.norm(scale_tensor[i1]-scale_tensor[i2])
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
'''


# In[11]:
def compute_conv_layer_dist_channel_pruning(module_cand_conv, block_list_l, block_id):
    global layer_number
    candidate_convolution_layer = []
    end_index = 0
    for bl in range(len(block_list_l)):
        start_index = end_index
        end_index = end_index + block_list_l[bl]
        if bl != block_id:
            continue

        for lno in range(start_index, end_index):
            # layer_number =st+i
            with open(outLogFile, 'a') as out_file:
                out_file.write(f"\nlno in compute candidate {lno}")
            out_file.close()
            print(lno)
            candidate_convolution_layer.append(fp.compute_distance_score_channel(
                module_cand_conv[lno]._parameters['weight'],
                n=1,
                dim_to_keep=[0],
                prune_amount=prune_count[lno]))
        break
    return candidate_convolution_layer


# In[12]:
class ChannelPruningMethodSimilarities(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        with open(outLogFile, "a") as log_file:
            log_file.write("\n Executing Compute Mask")
        log_file.close()
        mask = default_mask.clone()
        # mask.view(-1)[::2] = 0
        size = t.shape
        print(f"\n{size}")
        with open(outLogFile, "a") as log_file:
            log_file.write(f'\nLayer Number:{layer_number} \nstart={st} \nlength of new list={len(new_list)}')
        log_file.close()
        for l in range(len(new_list)):
            for i in range(len(new_list[l])):
                idx1 = new_list[l][i][0]
                idx2 = new_list[l][i][1]
                
                if torch.norm(t[idx1]) >= torch.norm(t[idx2]):
                    k=idx1
                else:
                    k=idx2
                
                mask[k] = 0
        return mask


def channel_unstructured_similarities(channel_module, name):
    ChannelPruningMethodSimilarities.apply(channel_module, name)
    return channel_module


# In[13]
layer_base=0
def iterative_channel_pruning_similarities_block_wise(new_model_arg, prune_module, 
                                             block_list_l, prune_epochs):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nPruning Process Start")
    out_file.close()
    # pc = [1, 3, 9, 26, 51]
    
    global new_list
    global layer_base
    
    for e in range(prune_epochs):
        start = 0
        end = len(block_list_l)
        for blkId in range(start, end):
            # 2 Compute distance between kernel for candidate conv layer
            new_list = compute_conv_layer_dist_channel_pruning(module_cand_conv=prune_module,
                                                                   block_list_l=block_list_l, block_id=blkId)
            # 5 perform Custom pruning where we mask the prune weight
            for j in range(block_list_l[blkId]):
                if blkId < 2:
                    layer_number_to_prune = (blkId * 2) + j
                else:  # blkId >= 2:
                    layer_number_to_prune = 4 + (blkId - 2) * 3 + j
                channel_unstructured_similarities(
                    channel_module=prune_module[layer_number_to_prune], 
                    name='weight')
            new_list = None
        
        # 6.  Commit Pruning
        for i in range(len(prune_module)):
            prune.remove(module=prune_module[i], name='weight')
        
        # 7.  Update feature list
        global feature_list
        feature_list = update_feature_list(
            feature_list, prune_count, start=0, end=len(prune_count))
        
        # 8.  Create new temp model with updated feature list
        temp_model = lm.create_vgg_from_feature_list(
            vgg_feature_list=feature_list, batch_norm=True)
        temp_model.to(device1)
        
        # 9.  Perform deep copy
        lm.freeze(temp_model, 'vgg16')
        #deep_copy(temp_model, new_model_arg)
        lm.unfreeze(temp_model)
        
        
        # 10.  Train pruned model
        with open(outLogFile, 'a') as out_file:
            out_file.write('\n....Deep Copy Completed....')
            out_file.write('\n....Fine tuning started....')
        out_file.close()

        tm.fit_one_cycle( dataloaders=dataLoaders,
                          train_dir=dl.train_directory, test_dir=dl.test_directory,
                          # Select a variant of VGGNet
                          model_name='vgg16', model=temp_model, device_l=device1,
                          # Set all the Hyper-Parameter for training
                          epochs=8, max_lr=0.001, weight_decay=0.01, L1=0.01, grad_clip=0.1,
                          opt_func=opt_func, log_file=logResultFile)
        
        save_path = f'{model_dir}{program_name}/{selected_dataset_dir}/vgg16_IntelIc_Prune_{e}_b_train'
        torch.save(temp_model, save_path)
        # # # 10. Evaluate the pruned model
        train_accuracy = 0.0
        test_accuracy = 0.0
        train_accuracy = tm.evaluate(temp_model,dataLoaders[dl.trainDir])
        test_accuracy  = tm.evaluate(temp_model,dataLoaders[dl.testDir])

        with open(outFile, 'a') as out_file:
            out_file.write(f'\n output of the {e}th iteration is written below\n')
            out_file.write(f'\n Train Accuracy: {train_accuracy}'
                           f'\n Test Accuracy  :  {test_accuracy} \n')
        out_file.close()

        save_path = f'{model_dir}{program_name}/selected/dataset_dir/vgg16_IntelIc_Prune_{e}_b_train'
        # save_path = f'/home3/pragnesh/Model/vgg16_IntelIc_Prune_{e}_b_train'
        torch.save(temp_model, save_path)


# In[14]:
initialize_lists_for_pruning()
iterative_channel_pruning_similarities_block_wise(new_model_arg=new_model, 
    prune_module=module, block_list_l=block_list, prune_epochs=6)



