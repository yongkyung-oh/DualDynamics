import csv
import math
import os
import pathlib
from numpy.lib.function_base import append
import torch
import urllib.request
import zipfile
import numpy as np 
from . import common
# import time_dataset


here = pathlib.Path(__file__).resolve().parent


def _process_data(append_time,time_seq, missing_rate, y_seq):
    PATH = os.path.dirname(os.path.abspath(__file__))
    
    torch.__version__
    
    X_times = np.load(PATH+"/mujoco.npy")
    X_times = torch.tensor(X_times)
    feature_lst = ['nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    final_indices = []
    for time in X_times:
        
        final_indices.append(len(time)-1)
    
    maxlen = max(final_indices)+1
    
    for i in range(len(X_times)):
    
        for _ in range(maxlen - len(X_times[i])):
            X_times[i].append([float('nan') for i in feature_lst])
    
    final_indices = torch.tensor(final_indices)

    X_reg = []
    y_reg = []
    for i in range(X_times.shape[0]):
        for j in range(X_times.shape[1]-time_seq-y_seq): 
            X_reg.append(X_times[i,j:j+time_seq,:].tolist())
            y_reg.append(X_times[i,j+time_seq:j+time_seq+y_seq,:].tolist())
    
    X_reg = torch.tensor(X_reg)
    y_reg = torch.tensor(y_reg)
    

    generator = torch.Generator().manual_seed(56789)
    for Xi in X_reg:
        removed_points = torch.randperm(X_reg.size(1), generator=generator)[:int(X_reg.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float('nan')
    
    final_indices_reg = np.repeat(time_seq-1,X_reg.shape[0])
    final_indices_reg = torch.tensor(final_indices_reg)
    
    
    times = torch.linspace(1, X_reg.size(1), X_reg.size(1))
    (times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data_forecasting(times, X_reg, y_reg, final_indices_reg, append_times=append_time)
    
    
    return (times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data( batch_size, missing_rate,append_time, time_seq, y_seq):
    base_base_loc = here / 'processed_data'
    
    if append_time:
        loc = base_base_loc / ('mujoco' + str(time_seq)+'_'+ str(y_seq) + '_' +str(missing_rate)+'_time_aug')
    else:
        loc = base_base_loc / ('mujoco' + str(time_seq)+'_'+ str(y_seq) + '_' +str(missing_rate))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_X = tensors['train_X']
        val_X = tensors['val_X']
        test_X = tensors['test_X']
        train_coeffs = tensors['train_coeffs']
        val_coeffs = tensors['val_coeffs']
        test_coeffs = tensors['test_coeffs']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:
        
        (times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_data(append_time,time_seq, missing_rate, y_seq)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_X=train_X, val_X=val_X, test_X=test_X,
                         train_coeffs=train_coeffs, val_coeffs=val_coeffs, test_coeffs=test_coeffs,
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)
        
    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_X, val_X, test_X,
                                                                                train_coeffs, val_coeffs, test_coeffs, 
                                                                                train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader
