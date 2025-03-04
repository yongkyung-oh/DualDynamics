'''
https://github.com/yongkyung-oh
Author: YongKyung Oh
License: MIT License

Neuralflow model 
https://github.com/mbilos/neural-flows-experiments
Author: Marin Bilos
'''

import torch
import torchcde

from .flow import CouplingFlow, ResNetFlow
from .gru import ContinuousGRULayer, GRUFlow
from .lstm import ContinuousLSTMLayer


class NeuralFlow(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_hidden_layers, output_channels, input_option='z', flow_option='c'):
        super().__init__()
        self.input_option = input_option
        self.flow_option = flow_option
        
        self.initial_flow = torch.nn.Linear(input_channels, hidden_channels)
        self.initial_control = torch.nn.Linear(input_channels, hidden_channels)
        # self.initial_network = torch.nn.Linear(input_channels, hidden_channels)

        self.emb = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_channels, hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear = torch.nn.Linear(hidden_channels, output_channels)
        
        if flow_option == 'n':
            self.flow_in = torch.nn.Linear(hidden_channels, hidden_channels)
            self.flows = torch.nn.ModuleList(torch.nn.Linear(hidden_channels, hidden_channels)
                                             for _ in range(num_hidden_layers - 1))
            self.flow_out = torch.nn.Linear(hidden_channels, hidden_channels)
        elif flow_option == 'r':
            self.flow_network = ResNetFlow(dim=hidden_channels, n_layers=1, hidden_dims=[hidden_channels]*num_hidden_layers, time_net='TimeTanh')
        elif flow_option == 'g':
            self.flow_network = GRUFlow(dim=hidden_channels, n_layers=1, hidden_dims=[hidden_channels]*num_hidden_layers, time_net='TimeTanh')
        elif flow_option == 'c':
            self.flow_network = CouplingFlow(dim=hidden_channels, n_layers=1, hidden_dims=[hidden_channels]*num_hidden_layers, time_net='TimeTanh')
        
    def forward(self, x, seq_ts, seq_mask, coeffs, times, **kwargs):
        ## input value
        if self.input_option == 'n': # use nan for missing
            x = torch.where(x==0, torch.tensor(float('nan')).to(x.device), x)
            x[seq_mask==0] = float('nan')
        else:
            pass
        
        ## neural flow
        z_flow = self.initial_flow(torch.cat([seq_ts.unsqueeze(-1), x], dim=-1))
        
        # control path
        X = torchcde.CubicSpline(coeffs, times)
        z_x = self.initial_control(X.evaluate(times))
                
        if self.input_option == 'n': # use partial latent only
            z = z_flow
        if self.input_option == 'x': # use latent only
            z = z_flow
        elif self.input_option == 'y': # use control only
            z = z_x
        elif self.input_option == 'z': # use both
            z = self.emb(torch.cat([z_flow,z_x], dim=-1))
        
        # flow network
        if self.flow_option == 'n':
            z = self.flow_in(z)
            z = z.relu()
            for flow in self.flows:
                z = flow(z)
                z = z.relu()
            z = self.flow_out(z)
        else:
            z = self.flow_network(z, seq_ts.unsqueeze(-1))

        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        out = self.linear(z)
        return out, z
  

class DualDynamics(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, num_hidden_layers, output_channels, input_option='z', flow_option='c'):
        super().__init__()
        self.input_option = input_option
        self.flow_option = flow_option
                
        self.func = func

        self.initial_flow = torch.nn.Linear(input_channels, hidden_channels)
        self.initial_control = torch.nn.Linear(input_channels, hidden_channels)
        # self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        
        self.emb = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.linear = torch.nn.Sequential(
            torch.nn.Tanh(), torch.nn.Linear(hidden_channels, hidden_channels), 
            torch.nn.ReLU(), torch.nn.Linear(hidden_channels, output_channels), 
        )

        if flow_option == 'n':
            self.flow_in = torch.nn.Linear(hidden_channels, hidden_channels)
            self.flows = torch.nn.ModuleList(torch.nn.Linear(hidden_channels, hidden_channels)
                                             for _ in range(num_hidden_layers - 1))
            self.flow_out = torch.nn.Linear(hidden_channels, hidden_channels)
        elif flow_option == 'r':
            self.flow_network = ResNetFlow(dim=hidden_channels, n_layers=1, hidden_dims=[hidden_channels]*num_hidden_layers, time_net='TimeTanh')
        elif flow_option == 'g':
            self.flow_network = GRUFlow(dim=hidden_channels, n_layers=1, hidden_dims=[hidden_channels]*num_hidden_layers, time_net='TimeTanh')
        elif flow_option == 'c':
            self.flow_network = CouplingFlow(dim=hidden_channels, n_layers=1, hidden_dims=[hidden_channels]*num_hidden_layers, time_net='TimeTanh')
            
    def forward(self, x, seq_ts, seq_mask, coeffs, times, **kwargs):
        seq_ts = times.repeat(x.shape[0], 1).to(x.device)  # [N,L]
        
        ## control path
        X = torchcde.CubicSpline(coeffs, times)
        z0 = self.initial_control(X.evaluate(times[0]))
        
        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()
        
        # approximation
        if kwargs['method'] == 'euler':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = max(time_diffs.min().item(), 1e-3)
                
        # Actually solve the CDE
        z_t = torchcde.cdeint(X=X,
                              func=self.func,
                              z0=z0,
                              t=times,
                              **kwargs)
        
        ## neural flow
        # control path for all t
        X = torchcde.CubicSpline(coeffs, times)
        # z_x = self.initial_network(X.evaluate(times))
        z_x = self.initial_control(X.evaluate(times))
                
        if self.input_option == 'n': # use partial latent only
            z = z_t
        elif self.input_option == 'x': # use latent only
            z = z_t
        elif self.input_option == 'y': # use control only
            z = z_x
        elif self.input_option == 'z': # use both
            z = self.emb(torch.cat([z_t,z_x], dim=-1))
            
        # flow network
        if self.flow_option == 'n':
            z = self.flow_in(z)
            z = z.relu()
            for flow in self.flows:
                z = flow(z)
                z = z.relu()
            z = self.flow_out(z)
        else:
            z = self.flow_network(z, seq_ts.unsqueeze(-1))
        out = self.linear(z)
        return out, z_t

    
