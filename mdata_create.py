import json
import os
import math
import torch

train_meta_file_path = '../dataset_224/activitynet/train_mdata.txt'
valid_meta_file_path = '../dataset_224/activitynet/valid_mdata.txt'
test_meta_file_path = '../dataset_224/activitynet/test_mdata.txt'


def line_to_dict(line):
    ldict = {}
    line_list = line.split()
    v_id = line_list[0]
    line_list = line_list[1:]

    # add \n to end of each caption in line
    for i, content in enumerate(line_list):
        if i+1 < len(line_list) :
            if content.isdigit() == False and line_list[i+1].isdigit() == True:
                line_list[i] = content + r'\n'

    frame_list = ' '.join(line_list).split(r'\n')
    frame_list = frame_list[:-1]

    for event in frame_list:
        event_list = event.split()
        start = event_list[0]
        end = event_list[1]
        cap = ' '.join(event_list[2:])
        ldict.update({cap:[start, end]})

    return {v_id:ldict} 


def txt_to_dict(file_path):
    tdict = {}
    with open(file_path,'r') as f:
        content = [x.rstrip() for x in f.readlines()]
        total_videos = len(content)
        print('Total videos', total_videos)
        for i, line in enumerate(content):
            print(i+1, 'Left', total_videos-i-1)
            line = line+ r'\n'
            line_dict = line_to_dict(line)
            tdict.update(line_dict)
    return tdict


if __name__ == "__main__":

    print('Train dict')
    train_dict = txt_to_dict(train_meta_file_path)
    print('Valid dict')
    valid_dict = txt_to_dict(valid_meta_file_path)
    print('Test dict')
    test_dict = txt_to_dict(test_meta_file_path)

    data_dict = {'train':train_dict, 'valid':valid_dict, 'test':test_dict}

    # Save cnn_features to pytorch file
    print(len(train_dict), len(valid_dict), len(test_dict))
    print('Saving dictionary')
    state = data_dict
    file_path = 'm_data.pt'
    torch.save(state, file_path) 
    print('Saved...')