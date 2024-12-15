import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm
import os
import time 
import models

here = pathlib.Path(__file__).resolve().parent


def _add_weight_regularisation(loss_fn, regularise_parameters, mode='l2', scaling=0.01):
    def new_loss_fn(pred_y, true_y):
        total_loss = loss_fn(pred_y, true_y)
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                if mode == 'l1':
                    # total_loss = total_loss + scaling * torch.norm(parameter, p='nuc')
                    total_loss = total_loss + scaling * torch.norm(parameter, p=1)
                elif mode == 'l2':
                    total_loss = total_loss + scaling * torch.norm(parameter, p='fro')
                else:
                    pass
        return total_loss
    return new_loss_fn

class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        
        return self.model(*args, **kwargs).squeeze(-1)


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]


def _evaluate_metrics_forecasting(model_name, dataloader, model, times, loss_fn, device, kwargs):
    with torch.no_grad():
        total_dataset_size = 0
        total_loss = 0
        
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            seq, coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)

            pred_y = model(seq, times, coeffs, lengths, **kwargs)
                
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y) * batch_size

        total_loss /= total_dataset_size  
        
        metrics = _AttrDict(dataset_size=total_dataset_size, loss=total_loss.item())
    
        return metrics

class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True

def _train_loop_forecasting(model_name, train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, eval_fn, max_epochs,
                            device, kwargs, step_mode) :
                           
    model.train()
    best_model = model
    best_train_loss = math.inf
    
    best_train_loss_epoch = 0
    best_val_loss = 0
    
    history = []
    breaking = False
    # scheduler : Reduce learning rate when a metric has stopped improving.
    if step_mode == 'trainloss':
        print("trainloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    elif step_mode=='valloss':
        print("valloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    elif step_mode == 'valaccuracy':
        print("valaccuracy")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')

    elif step_mode=='valauc':
        print("valauc")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        
    elif step_mode=='none':
        epoch_per_metric=1 
        plateau_terminate=50
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    for epoch in tqdm_range:
        if breaking:
            break
        
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                
                train_seq, train_coeffs, train_y, lengths = batch
                
                pred_y = model(train_seq, times, train_coeffs, lengths, **kwargs)
                loss = loss_fn(pred_y, train_y)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            # evaluate train,val,test. 
            train_metrics = _evaluate_metrics_forecasting(model_name, train_dataloader, model, times, eval_fn,  device, kwargs)
            val_metrics = _evaluate_metrics_forecasting(model_name, val_dataloader, model, times, eval_fn,  device, kwargs)
            test_metrics = _evaluate_metrics_forecasting(model_name, test_dataloader, model, times, eval_fn,  device,kwargs)
                       
            model.train()
            
            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch
                        
            if val_metrics.loss * 1.0001 < best_val_loss:
                best_val_loss = val_metrics.loss
                best_val_loss_epoch = epoch
         
            tqdm_range.write('Epoch: {} | Train loss: {:.3} | Val loss: {:.3} | Test loss : {:.3}'.format(epoch, train_metrics.loss, val_metrics.loss, test_metrics.loss))
            
            if step_mode == 'trainloss':
                scheduler.step(train_metrics.loss)
            elif step_mode=='valloss':
                scheduler.step(val_metrics.loss)
            elif step_mode == 'valaccuracy':
                scheduler.step(val_metrics.accuracy)
            elif step_mode=='valauc':
                scheduler.step(val_metrics.auroc)
            else:
                # scheduler.step()
                pass

                
            history.append(_AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))
            # Early stop
            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                    ''.format(plateau_terminate))
                breaking = True
            
           

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    return history,epoch


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result, out_name):
    loc = here / 'results' / name
    if not os.path.exists(loc):
        os.mkdir(loc)
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])

    with open(loc / str(out_name), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)

        
def main_forecasting(args, name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model, max_epochs,
                     lr, weight_decay, loss, reg, scale, kwargs, step_mode, out_name):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None   

    model, regularise_parameters = make_model()  
    # mse loss function
    # loss_fn = torch.nn.functional.huber_loss
    if loss == 'mse':
        loss_fn = torch.nn.functional.mse_loss
    if loss == 'huber':
        loss_fn = torch.nn.functional.huber_loss
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters, mode=reg, scaling=scale)
    eval_fn = torch.nn.functional.mse_loss
    model.to(device)
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = lr*0.01)
    # train function
    history = _train_loop_forecasting(model_name, train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, eval_fn, max_epochs,
                                      device, kwargs, step_mode)

    model.eval()
    train_metrics = _evaluate_metrics_forecasting(model_name, train_dataloader, model, times, eval_fn, device, kwargs)
    val_metrics = _evaluate_metrics_forecasting(model_name, val_dataloader, model, times, eval_fn, device, kwargs)
    test_metrics = _evaluate_metrics_forecasting(model_name, test_dataloader, model, times, eval_fn, device,kwargs)

    
    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
        print(f"memory_usage:{memory_usage}")
    else:
        memory_usage = None
    result = _AttrDict(args=args,
                       name=name,
                       model_name=model_name,
                       times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       loss_setting = [loss, reg, scale],
                       parameters=_count_parameters(model),
                       history=history,
                       train_metrics=train_metrics,
                       val_metrics=val_metrics,
                       test_metrics=test_metrics)
                    
    if name is not None:
        _save_results(name, result, out_name)
    return result


def make_model(name, input_channels, output_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, use_intensity, initial, output_time = 0):
    
    print(name)

    if name in [x for y in [['dualdynamics_{}_{}'.format(i,j) for i in ['x', 'y', 'z']] for j in ['n', 'r', 'g', 'c']] for x in y]:
        def make_model():
            input_option = str(name.split('_')[1])
            flow_option = str(name.split('_')[2])
            vector_field = models.FinalTanhT(input_channels=input_channels, hidden_channels=hidden_channels,
                                             hidden_hidden_channels=hidden_channels, num_hidden_layers=num_hidden_layers)
            model = models.DualDynamics(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels, output_time=output_time, 
                                                num_hidden_layers=num_hidden_layers, output_channels=output_channels, input_option=input_option, flow_option=flow_option)
            return model, vector_field
        
    ##    
    elif name == 'ncde':
        def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde_forecasting':
         def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'gruode':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels, output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name =='gruode_forecasting':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'dt':
        def make_model():
            model = models.GRU_dt(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'dt_forecasting':
        def make_model():
            model = models.GRU_dt_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'decay':
        def make_model():
            model = models.GRU_D(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'decay_forecasting':
        def make_model():
            model = models.GRU_D_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'odernn':
        def make_model():
            model = models.ODERNN(input_channels=input_channels, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'odernn_forecasting':
        def make_model():
            
            model = models.ODERNN_forecasting(input_channels=input_channels,output_time = output_time, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    else:
        raise ValueError("Unrecognised model name {}. Valid names are 'ncde', 'gruode', 'dt', 'decay' and 'odernn'."
                         "".format(name))
    return make_model
