import collections as co
import numpy as np
import os
import pathlib
import sktime
# import sktime.utils.load_data
from sktime.datasets import load_from_tsfile_to_dataframe
import torch
import urllib.request
import zipfile
from . import time_dataset
from . import common

here = pathlib.Path(__file__).resolve().parent



def _pad(channel, maxlen):
    channel = torch.tensor(channel) 
    out = torch.full((maxlen,), channel[-1]) 
    out[:channel.size(0)] = channel 
    return out 



def _process_data(data_path,sequence,y_seq, missing_rate, intensity):
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    total_length = len(data)
#     data = data[::-1]
    
    # min_val = np.min(data, 0)
    # max_val = np.max(data, 0) - np.min(data, 0)
    
    norm_data = time_dataset.normalize(data)
    total_length = len(norm_data)
    idx = np.array(range(total_length)).reshape(-1,1)
    norm_data = np.concatenate((norm_data,idx),axis=1)

    seq_data = []
    
    for i in range(len(norm_data) - sequence -y_seq + 1): 
        x = norm_data[i : i + sequence+y_seq]
        seq_data.append(x)
    
    samples = []
    idx = torch.randperm(len(seq_data))
    for i in range(len(seq_data)):
        samples.append(seq_data[idx[i]])
    
    for j in range(len(samples)):
        if j == 0 : 
            this = torch.tensor(samples[j])
            this = torch.reshape(this,[1,this.shape[0],this.shape[1]])
        else : 
            
            tt = torch.from_numpy(np.flip(samples[j],axis=0).copy())
            this0 = torch.reshape(tt,[1,tt.shape[0],tt.shape[1]])
            this = torch.cat([this,this0]) 
            
    X = this[:,:sequence,:].float()
    y = this[:,sequence:,:].float()
    final_index = (torch.ones(X.shape[0]) * (sequence-1)).cuda() 
    final_index = final_index.long()
            
    
    times = torch.linspace(0, X.size(1) - 1, X.size(1)) 

    generator = torch.Generator().manual_seed(56789)
    for Xi in X:
        removed_points = torch.randperm(X.size(1), generator=generator)[:int(X.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float('nan')
    print(f"missing_rate : {missing_rate}")
    X=X.cuda()
    y=y.cuda()
    times = times.cuda()
    (times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, input_channels) = common.preprocess_data_forecasting(times, X, y, final_index, append_times=True)

    num_classes = 1

    return (times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, num_classes, input_channels)


def get_data(data_path, time_seq, y_seq, missing_rate, device, time_augment, batch_size):
    
    base_base_loc = here / 'processed_data'
    base_loc = base_base_loc / 'Stock_test'
    dataset_name = "google"

    loc = base_loc / (dataset_name +'_'+ str(int(missing_rate * 100)) +str(y_seq)+ ('_intensity' if time_augment else ''))
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
        num_classes = int(tensors['num_classes'])
        input_channels = int(tensors['input_channels'])
    else:
        
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        (times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
        test_final_index, num_classes, input_channels) = _process_data(data_path, time_seq, y_seq, missing_rate, time_augment)
        
        common.save_data(loc, times=times,
                         train_X=train_X, val_X=val_X, test_X=test_X,
                         train_coeffs=train_coeffs, val_coeffs=val_coeffs, test_coeffs=test_coeffs,
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index,
                         num_classes=torch.as_tensor(num_classes), input_channels=torch.as_tensor(input_channels))
    
    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_X, val_X, test_X,
                                                                                train_coeffs, val_coeffs, test_coeffs, 
                                                                                train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, device,
                                                                                num_workers=0, batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels

