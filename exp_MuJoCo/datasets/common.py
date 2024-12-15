import os
import pathlib
import sklearn.model_selection
import sys
import torch
import torchcde

here = pathlib.Path(__file__).resolve().parent

def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 8
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def split_data(tensor, stratify):
    
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def preprocess_data_forecasting(times, X, y, final_index, append_times):
    X = X.cuda()
    
    full_len = X.shape[0]
    train_len = int(full_len*0.7) 
    val_len = int(full_len*0.85)
    augmented_X = []
    if append_times:
        augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))

    augmented_X.append(X)
    augmented_X[0] = augmented_X[0].cuda()
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)
    if append_times:
        print(f"time augment : {X.shape}")
    train_X, train_y = X[:train_len], y[:train_len]
    val_X, val_y = X[train_len:val_len],y[train_len:val_len]
    test_X, test_y = X[val_len:], y[val_len:]
    
    train_final_index, val_final_index, test_final_index = final_index[:train_len],final_index[train_len:val_len],final_index[val_len:]
    
    times = torch.linspace(0, X.size(1) - 1, X.size(1)).cuda()
    # train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), train_X)
    # val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), val_X)
    # test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), test_X)

    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X, times)
    val_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(val_X, times)
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X, times)
    
    in_channels = X.size(-1)

    return (times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, in_channels)


def wrap_data(times, train_X, val_X, test_X, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
              test_final_index, device, batch_size, num_workers=4):
    times = times.to(device)
    train_X = train_X.to(device)
    val_X = val_X.to(device)
    test_X = test_X.to(device)
    train_coeffs = train_coeffs.to(device)
    val_coeffs = val_coeffs.to(device)
    test_coeffs = test_coeffs.to(device)
    train_y = train_y.to(device)
    val_y = val_y.to(device)
    test_y = test_y.to(device)
    train_final_index = train_final_index.to(device)
    val_final_index = val_final_index.to(device)
    test_final_index = test_final_index.to(device)
    
    train_dataset = torch.utils.data.TensorDataset(train_X, train_coeffs, train_y, train_final_index)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_coeffs, val_y, val_final_index)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_coeffs, test_y, test_final_index)

    train_dataloader = dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    return times, train_dataloader, val_dataloader, test_dataloader


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


def load_data(dir):
    
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors
