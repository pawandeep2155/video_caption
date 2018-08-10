import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import skvideo.io
import time
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 1  # Extract Resnet features in a tensor of batch_size videos.
total_ext = 1000  #  Extract Resnet features of total_ext videos

m_data_path = '../dataset_224/activitynet/m_data.pt'
resnet_feat_path = '../dataset_224/activitynet/res_feature.pt'
batch_ckpt_path = '../dataset_224/activitynet/batch_num.txt'
video_path = '../dataset_224/activitynet/videos/'

assert os.path.exists(m_data_path) and \
        os.path.exists(video_path), 'Check the file paths.'
assert batch_size >= 1, 'Batch size should be greater than or equal to 1.'

# Download pre trained Resnet101.
resnet101 = models.resnet101(pretrained=True)
modules = list(resnet101.children())[:-1]
resnet101 = nn.Sequential(*modules)
for p in resnet101.parameters():
    p.requires_grad = False


# Convert id's list to dictionary with key as id, value as id numpy array
def id_to_array(batch_ids):
    batch_array = {}
    for ids in batch_ids:
        video = skvideo.io.vread(video_path + ids + '.mp4')
        batch_array.update({ids:video})
    return batch_array

# Convert array dictionary to resnet_dictionary with key as id, value as id tensor feature array
def resnet_features(batch_arrayd):
    batch_feature = {}
    ids = list(batch_arrayd.keys())
    video_array = [x for x in batch_arrayd.values()]
    array_sizes = [x.shape[0] for x in batch_arrayd.values()]

    video1_array = np.array(video_array[0], dtype = np.float32)  # change datatype of frames to float32
    video_tensor = torch.from_numpy(video1_array)

    if batch_size > 1:
        for i in range(1, len(video_array)):
            videoi_array = np.array(video_array[i], dtype = np.float32)
            videoi_tensor = torch.from_numpy(videoi_array)
            video_tensor = torch.cat((video_tensor, videoi_tensor), 0)

    video_tensor = video_tensor
    video_tensor = video_tensor.permute(0,3,1,2) # change dimension to [?,3,224,224]
    tensor_var = Variable(video_tensor)
    resnet_feature = resnet101(tensor_var).cuda().data
    resnet_feature.squeeze_(3)  # eliminate last dimension to get [?, 2048, 1, 1]
    resnet_feature.squeeze_(2)  # eliminate last dimension to get [?, 2048]

    index = 0
    for i in range(len(video_array)):
        id_tensor = resnet_feature[index:index+array_sizes[i]]
        batch_feature.update({ids[i]:id_tensor})
        index += array_sizes[i]

    return batch_feature


# seconds to minutes
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return m, s

# time in minutes b/w since to now
def timeSince(since):
    now = time.time()
    s = now - since
    min, sec = asMinutes(s) 
    return min, sec


if __name__ == '__main__':

    print('Total videos', total_ext)
    print('batch size', batch_size)
    total_iter = math.ceil(total_ext / batch_size)

    featured = {}

    m_data = torch.load(m_data_path)

    all_ids = list(m_data['train'].keys()) + list(m_data['valid'].keys()) + list(m_data['test'].keys())
    all_ids = all_ids[:total_ext]  # subset of ids 

    if os.path.exists(batch_ckpt_path):
        with open(batch_ckpt_path,'r') as f:
            content = f.readline().rstrip()
            batch_start = int(content)
    else:
        batch_start =  0

    batch_end = min(batch_start+batch_size,total_ext) 

    iter = 1
    while batch_start != batch_end:

        print('Iteration', iter, 'left', total_iter-iter, end=' ')
        start_time = time.time()
        batch_id = all_ids[batch_start:batch_start+batch_size]
        batch_arrayd = id_to_array(batch_id)
        batch_featuresd = resnet_features(batch_arrayd) 
        featured.update(batch_featuresd)
        print('time taken (%dm %ds)'% timeSince(start_time))

        # Save resnet features to pytorch file
        state = featured
        torch.save(state, resnet_feat_path)  

        # save batch_start to batch_ckpt_path file
        with open(batch_ckpt_path, 'w') as f:
            f.write(str(batch_end))

        batch_start = batch_end 
        batch_end = min(batch_start+batch_size, total_ext)
        iter += 1