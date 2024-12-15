import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Google_2021')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Seed - Test your luck!')
    parser.add_argument('--intensity', type=bool,
                        default=True, help='Intensity')
    parser.add_argument('--model', type=str, default='ncde', help='Model Name')
    parser.add_argument('--h_channels', type=int,
                        default=256, help='Hidden Channels')
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--hh_channels', type=int,
                        default=64, help='Hidden Hidden Channels')
    parser.add_argument('--layers', type=int, default=4,
                        help='Num of Hidden Layers')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning Rate')
    parser.add_argument('--time_lr', type=float, default=1.0,
                        help='Time_Learning Rate')
    parser.add_argument('--epoch', type=int, default=200, help='Epoch')
    parser.add_argument('--step_mode', type=str,
                        default='valloss', help='Model Name')
    parser.add_argument('--missing_rate', type=float,
                        default=0.3, help='Missing Rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='weight_decay')
    parser.add_argument('--loss', type=str, default='mse', help='loss setting')
    parser.add_argument('--reg', type=str, default='none', help='regularization setting')
    parser.add_argument('--scale', type=float, default=0.01, help='regularization setting')
    
    parser.add_argument('--method', type=str, default='euler', help='ode solver')
    parser.add_argument('--time_seq', type=int, default=50, help='time_seq')
    parser.add_argument('--y_seq', type=int, default=10, help='y_seq')
    parser.add_argument('--result_folder', type=str, default='tensorboard_google', help='tensorboard log folder')
    # parser.add_argyment('--')
    return parser.parse_args()
