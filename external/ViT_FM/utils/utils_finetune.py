
from regression.EmbeddingModule import FourierEmbedding, AdaptiveScale
from regression.ViTModule import SimpleViT

import torch
import torch.nn as nn

import time

class FT_Lift(torch.nn.Module):
    def __init__(self, 
                model,
                in_dim,
                add_linear = False):
        super(FT_Lift, self).__init__()

        self.lift = model.model.lift
        self.add_linear = add_linear
        if add_linear:
            self.linear = nn.Linear(in_dim, in_dim)
            with torch.no_grad():
                self.linear.bias.zero_()
                self.linear.weight.zero_()

        self.mean = nn.Parameter(torch.tensor(0.0))
        self.std = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, emb = None):
        x = (x - self.mean)/self.std
        dx = 0.
        if self.add_linear:
            x = torch.permute(x, (0,2,3,1))
            dx = self.linear(x)
            x = torch.permute(x, (0,3,1,2))
            dx = torch.permute(dx, (0,3,1,2))

        x = self.lift(x + dx)
        return x

class FT_Project(torch.nn.Module):
    def __init__(self, 
                model,
                out_dim,
                add_linear = False):
        super(FT_Project, self).__init__()
        
        #dim = model.encoder_features[0] + model.decoder_features_out[-1]

        self.project =  model.model.project

        self.add_linear = add_linear
        if add_linear:
            self.linear = nn.Linear(out_dim, out_dim)
            with torch.no_grad():
                self.linear.bias.zero_()
                self.linear.weight.zero_()

    def forward(self, x, emb = None):
        x = self.project(x)
        dx = 0.
        if self.add_linear:
            x = torch.permute(x, (0,2,3,1))
            dx = self.linear(x)
            x = torch.permute(x, (0,3,1,2))
            dx = torch.permute(dx, (0,3,1,2))
    
        return x + dx


def initialize_FT(model, 
                new_in_dim, 
                new_out_dim, 
                latent_channels = 128, 
                init_new = False, 
                init_film = False,
                ar_train = False):
    
    if init_new:
        '''
        model.model.lift = nn.Sequential(
                nn.Conv2d(new_in_dim, latent_channels, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
            )
        
        model.model.project = nn.Sequential(
                    nn.Conv2d(latent_channels, latent_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                    nn.Conv2d(latent_channels, new_out_dim, kernel_size=1, bias=False)
            )
        '''
        model.model.lift[0] = nn.Conv2d(new_in_dim, latent_channels, kernel_size=1, bias=False)
        model.model.project[-1] = nn.Conv2d(latent_channels, new_out_dim, kernel_size=1, bias=False)

    else:
        model.model.lift = FT_Lift(model = model,
                            in_dim = new_in_dim,
                            add_linear = True)
        
        model.model.project = FT_Project(model = model,
                                  out_dim = new_out_dim,
                                  add_linear = True)
    model.out_dim = new_out_dim
    model.in_dim = new_in_dim
    
    model.ar_train = ar_train
    if init_film:
        for attn, ff in model.model.transformers.layers:
            if attn.adapt is not None:
                attn.adapt = AdaptiveScale(attn.adapt.emb_channels, attn.adapt.dim, dim=1)
            if ff.adapt is not None:
                ff.adapt = AdaptiveScale(ff.adapt.emb_channels, ff.adapt.dim, dim=1)
    
    if model.model.rescale_time:
        model.model._rescale_time()

    return model