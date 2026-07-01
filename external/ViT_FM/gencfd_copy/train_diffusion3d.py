# Setup relevant libraries
import os
import warnings
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt

from torch import optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union

import GenCFD
from GenCFD import diffusion as dfn_lib
from GenCFD import model, train, solvers, utils

from GenCFD.dataloader.fluid_flows_3d import TaylorGreen3D

Tensor = torch.Tensor
array = np.ndarray

DATA_STD = 0.5 # Fixed parameter but can also be learnable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


