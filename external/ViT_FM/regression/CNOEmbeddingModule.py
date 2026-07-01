import torch.nn as nn
import torch
import time as time1

class FILM(torch.nn.Module):
    def __init__(self, 
                channels,
                intermediate = 128):
        super(FILM, self).__init__()

        self.channels = channels
        
        self.inp2lat_sacale = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2scale = nn.Linear(in_features=intermediate, out_features=channels)

        self.inp2lat_bias = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2bias = nn.Linear(in_features=intermediate, out_features=channels)
        
        self.inp2lat_sacale.weight.data.fill_(0)
        self.lat2scale.weight.data.fill_(0)
        self.lat2scale.bias.data.fill_(1)
        
        self.inp2lat_bias.weight.data.fill_(0)
        self.lat2bias.weight.data.fill_(0)
        self.lat2bias.bias.data.fill_(0)
        
        self.norm = nn.LayerNorm(self.channels)

        
    def forward(self, x, time = None):

        x = self.norm(x)

        if time is not None:
            timestep  = time.reshape(-1,1).type_as(x)
            
            scale     = self.lat2scale(self.inp2lat_sacale(timestep))
            bias      = self.lat2bias(self.inp2lat_bias(timestep))
            scale     = scale.unsqueeze(1).expand_as(x)
            bias      = bias.unsqueeze(1).expand_as(x)
            
            return x * scale + bias 
        else:
            return x