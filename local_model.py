import torch
from torch import nn 

def activation_func(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sin':
        return torch.sin 
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'none':
        return nn.Identity()
    elif activation == 'exp':
        return torch.exp
    elif activation == 'erf':
        return torch.erf
    else:
        raise ValueError('Activation does not exist')

class MLP(nn.Module):
    def __init__(self, num_f, hidden, num_layers, activation='relu'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(num_f, hidden))
        for l in range(num_layers):
            self.layers.append(nn.Linear(hidden,hidden))
        self.last_layer = nn.Linear(hidden, 1)
        self.activation = activation_func(activation)
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
            self.activation(x)
        x = self.last_layer(x)
        
        return torch.squeeze(x)
