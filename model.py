import torch
from torch import nn  
# from device import proj_device
# device = proj_device.device

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


class ContextNorm(nn.Module):
    def __init__(self):
        super(ContextNorm, self).__init__()
        self.eps = 1e-5

    def forward(self,x):
        mean = torch.mean(x,dim=2,keepdim=True)
        var = torch.var(x,dim=2,keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps) 
        return out 

class ResNetBlock(nn.Module):
    def __init__(self,hidden, activation='relu'):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(hidden,hidden,kernel_size=1)
        self.gcn1 = ContextNorm()
        self.bn1 = nn.BatchNorm1d(hidden, track_running_stats=True)
        self.activation = activation_func(activation)
        self.conv2 = nn.Conv1d(hidden,hidden,kernel_size=1)
        self.gcn2 = ContextNorm()
        self.bn2 = nn.BatchNorm1d(hidden, track_running_stats=True)

    def forward(self,inp):
        x = self.conv1(inp)
        x = self.gcn1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.gcn2(x)
        x = self.bn2(x)
        x = self.activation(x)
        out = x + inp 
        return out 
    
class MLP(nn.Module):
    def __init__(self, hidden, blocks, activation='relu'):
        super(MLP, self).__init__()

        self.first_conv = nn.Conv1d(3,hidden,1)
        self.blocks = nn.ModuleList([ResNetBlock(hidden=hidden,activation=activation) for i in range(blocks)])
        self.last_conv = nn.Conv1d(hidden,1,1)
        self.apply(self._init_weights)
        
        

    def forward(self, inp):
        x = self.first_conv(inp)
        for b in self.blocks:
            x = b(x)
        x = self.last_conv(x)
        return torch.squeeze(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, std = 2 / m.in_channels)
            if m.bias is not None:
                nn.init.zeros_(m.bias)