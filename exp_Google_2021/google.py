import common
import torch
from random import SystemRandom
import datasets
import numpy as np
import os 
import random
from parse import parse_args

# from tensorboardX import SummaryWriter
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

args = parse_args()

def main(
    manual_seed=args.seed,
    intensity=args.intensity, 
    device="cuda",
    max_epochs=args.epoch,
    missing_rate=args.missing_rate,
    model_name=args.model,
    hidden_channels=args.h_channels,
    hidden_hidden_channels=args.hh_channels,
    num_hidden_layers=args.layers,
    dry_run=False,
    method = args.method,
    step_mode = args.step_mode,
    lr=args.lr,
    weight_decay = args.weight_decay,
    loss=args.loss,
    reg = args.reg,
    scale=args.scale,
    time_seq=args.time_seq,
    y_seq=args.y_seq,
    **kwargs
):            

    batch_size = 256
    lr = 1e-3
    PATH = os.path.dirname(os.path.abspath(__file__))
    data_path = PATH+"/datasets/google_2021.csv"
    
    experiment_id = int(SystemRandom().random() * 100000)
    out_name = '{}_{}'.format(model_name, experiment_id)
    seed_everything(experiment_id)
    
    # these models use the intensity for their evolution. They won't explicitly use it as an input unless we include it
    # via the use_intensity parameter, though.
    # import pdb ; pdb.set_trace()
    time_augment = intensity or (model_name in ('odernn_forecasting', 'dt_forecasting', 'decay_forecasting'))

    # import pdb ; pdb.set_trace()
    (   times,
        train_dataloader,
        val_dataloader,
        test_dataloader, 
        num_classes, 
        input_channels
   
    ) = datasets.google.get_data(
        data_path,
        time_seq=time_seq,
        y_seq=y_seq,
        missing_rate=missing_rate,
        device=device,
        time_augment=time_augment, 
        batch_size=batch_size)

    output_time = y_seq
    input_channels = time_augment + 7
    folder_name = 'Google'
    test_name = "step_" + "_".join([ str(j) for i,j in dict(vars(args)).items()]) + "_" + str(experiment_id)
    result_folder = PATH+'/tensorboard_google'
    # writer = SummaryWriter(f"{result_folder}/runs/{folder_name}/{str(test_name)}")
   
    #model initialize
    make_model = common.make_model(model_name, input_channels, 7, hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers,use_intensity=intensity, method = args.method, 
                                   initial=True, output_time=output_time)
    
    if dry_run:
        name = None
    else:
        # name = 'Google'
        name = 'Google_' + str(missing_rate)
    num_classes = 2
    return common.main([hidden_channels,num_hidden_layers], name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device,
                       make_model, num_classes, max_epochs, lr, weight_decay, loss, reg, scale, kwargs, step_mode=args.step_mode, out_name=out_name)
            
if __name__ == "__main__":
    main(method = args.method)
    
