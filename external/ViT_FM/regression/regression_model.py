from diffusion.model import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

class UNet(nn.Module):

  def __init__(self, 
                in_dim,
                out_dim,
                channels=[32, 64, 128, 256]):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.n_layers = len(channels)
    self.in_dim = in_dim
    self.out_dim = out_dim

    self.encoder_channels = [self.in_dim] + channels
    self.encoder_conv  = [nn.Conv2d(self.encoder_channels[0], self.encoder_channels[1], 3, stride=1, padding = 1, bias=False)] 
    self.encoder_conv  = self.encoder_conv + [nn.Conv2d(self.encoder_channels[i], self.encoder_channels[i+1], 3, stride=2, padding = 1, bias=False) for i in range(1, self.n_layers)]
    self.encoder_conv  = nn.ModuleList(self.encoder_conv)
    self.encoder_group = nn.ModuleList([nn.BatchNorm2d(self.encoder_channels[i+1]) for i in range(self.n_layers)])

    self.decoder_channels_in = self.encoder_channels[::-1][:-1]
    self.decoder_channels_out = self.encoder_channels[::-1][1:]
    self.decoder_channels_out[-1] = self.out_dim

    for i in range(1, self.n_layers):
        self.decoder_channels_in[i]*=2
    self.decoder_conv  = [nn.ConvTranspose2d(self.decoder_channels_in[0],self.decoder_channels_out[0], 3, stride=2, padding=1, output_padding=1, bias=False)]
    for i in range(1, self.n_layers - 1):
        self.decoder_conv = self.decoder_conv + [nn.ConvTranspose2d(self.decoder_channels_in[i],self.decoder_channels_out[i], 3, stride=2, padding=1, output_padding=1, bias=False)]
    self.decoder_conv = self.decoder_conv + [nn.ConvTranspose2d(self.decoder_channels_in[-1],self.decoder_channels_in[-1], 3, stride=1, padding = 1)]
    self.decoder_conv = nn.ModuleList(self.decoder_conv)
    self.decoder_group = nn.ModuleList([nn.BatchNorm2d(self.decoder_channels_out[i]) for i in range(self.n_layers-1)])
    
    self.conv_last = torch.nn.Conv2d(in_channels = self.decoder_channels_in[-1],
                                          out_channels= self.decoder_channels_out[-1],
                                          kernel_size = 3,
                                          padding     = 1)

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)

  def forward(self, x,):
    skip = []

    for i in range(self.n_layers):
      x = self.encoder_conv[i](x)
      x = self.encoder_group[i](x)
      x = self.act(x)
      if i<self.n_layers-1:
        skip.append(x)
    
    x = self.decoder_conv[0](x)
    x = self.decoder_group[0](x)
    for i in range(1, self.n_layers-1):
      x = self.decoder_conv[i](torch.cat([x, skip[-i]], dim=1))
      x = self.decoder_group[i](x)
      x = self.act(x)

    x = self.decoder_conv[-1](torch.cat([x, skip[0]], dim=1))
    x = self.conv_last(x)
    return x
