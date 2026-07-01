import random
import torch.nn.functional as F
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import netCDF4 as nc
from abc import ABC
from typing import Optional
import time as time_time

import random 
import shutil
import subprocess
from itertools import product
import time

#---------------------------------------------------
# All the datasets (21 of them) are available at: _
#---------------------------------------------------

class BaseDataset(Dataset, ABC):
    """A base class for all datasets. Can be directly derived from if you have a steady/non-time dependent problem."""

    def __init__(
        self,
        which: Optional[str] = None,
        resolution: Optional[int] = None,
        in_dist: Optional[bool] = True,
        num_trajectories: Optional[int] = None,
        in_dim: Optional[int] = 2,
        out_dim: Optional[int] = 2,
        augment: Optional[bool] = False,
        data_path: Optional[str] = None,
        time_input: Optional[bool] = True,
        masked_input: Optional[list] = None,
        copy_to_local_scratch: Optional[bool] = True
    ) -> None:
        """
        Args:
            which: Which dataset to use, i.e. train, val, or test.
            resolution: The resolution of the dataset.
            in_dist: Whether to use in distribution or out of distribution data.
            num_trajectories: The number of trajectories to use for training.
            data_path: The path to the data files.
            time_input: Time in the input channels?
        """

        assert which in ["train", "val", "test"]
        assert resolution is not None and resolution > 0
        assert num_trajectories is not None and num_trajectories > 0
        
        #xprint(resolution, "RES")
        self.resolution = resolution
        self.in_dist = in_dist
        self.num_trajectories = num_trajectories
        self.data_path = data_path
        self.which = which
        self.time_input = time_input
        
        self.copy_to_local_scratch = copy_to_local_scratch
        self.file_path = None 

        self.masked_input = masked_input
        #if self.masked_input is not None:
        #    self.mask = torch.tensor(self.masked_input, dtype=torch.float32)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        print(self.in_dim, self.out_dim)

        self.augment = augment

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val >= 0
        assert self.N_test is not None and self.N_test >= 0
        if self.which == "train":
            self.length = self.num_trajectories
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test
      
    def _copy_to_tmpdir(self):
        tmpdir = os.environ.get("TMPDIR")
        if not tmpdir:
            raise RuntimeError("TMPDIR environment variable not set. Are you on a compute node?")
    
        basename = os.path.basename(self.file_path)
        dest_path = os.path.join(tmpdir, basename)

        _copy_spectral = False
        if hasattr(self, "spectral_file") and self.spectral_file is not None:
            _copy_spectral = True
            basename2 = os.path.basename(self.spectral_file)
            dest_path2 = os.path.join(tmpdir, basename2)
        print("COPYING SPECTRAL", _copy_spectral)

        def file_size(path):
            return os.path.getsize(path) if os.path.exists(path) else -1

        src_size = file_size(self.file_path)
        dst_size = file_size(dest_path)

        if not os.path.exists(dest_path) or dst_size < src_size:  # Only copy if it hasn't been copied or not entirely copied
            print(f"Copying {self.file_path} to {dest_path}...")
            # Option 1: shutil (simple copy)
            shutil.copy2(self.file_path, dest_path)

        if _copy_spectral and not os.path.exists(dest_path2):
            print(f"Copying {self.spectral_file} to {dest_path2}...")
            shutil.copy2(self.spectral_file, dest_path2)
            # Option 2: or use rsync if preferred
            # subprocess.run(["rsync", "-a", self.original_datafile, dest_path], check=True)

        else:
            time_time.sleep(10)
            print(f"File already exists in TMPDIR: {dest_path}")

        self.file_path = dest_path  # update to point to scratch
        if _copy_spectral:
            self.spectral_file = dest_path2

        print(self.spectral_file, self.file_path)
        
    def _rotate(self, X, Y, angle = 0):
        assert angle in [0, 90, 180, 270]
        if angle == 0:
            return X, Y
        else: 
            k = angle//90
            return torch.rot90(X, k=k, dims=[-2, -1]), torch.rot90(Y, k=k, dims=[-2, -1])
    
    def _transpose(self, X, Y, flip = 0):
        assert flip in [0, 1, 2]
        if flip == 0:
            return X, Y
        elif flip == 1:
            return torch.flip(X, dims=[-1]), torch.flip(Y, dims=[-1])
        else:
            return torch.flip(X, dims=[-2]), torch.flip(Y, dims=[-2])

    def _transform_boundary(self, X, which = 0):
        if which == 0:
            X[1:-1, 1:-1] = 0.0
        return X

    def _transform_data_policy(self, augmentations, policy, X, Y):
        assert len(augmentations) == len(policy)

        d = dict(zip(augmentations, policy))
        #print(augmentations, d)
        if "rotation" in augmentations:
            X, Y = self._rotate(X, Y, angle = d["rotation"])
        if "transpose" in augmentations:
            X, Y = self._transpose(X, Y, flip = d["transpose"])

        return X, Y
    
    def _transform_data_random(self, augmentations, X, Y):
        policy = []
        for aug in augmentations:
            if aug == "rotation":
               i = random.randint(0, 3)
               policy.append(i*90)
            if aug == "transpose":
                i = random.randint(0, 2)
                policy.append(i)
                
        return self._transform_data_policy(augmentations, policy, X, Y)


    def __len__(self) -> int:
        """
        Returns: overall length of dataset.
        """
        return self.length

    def __getitem__(self, idx) -> tuple:
        """
        Get an item. OVERWRITE!

        Args:
            idx: The index of the sample to get.

        Returns:
            A tuple of data.
        """
        pass

#--------------------------------------------------------

class BaseTimeDataset(BaseDataset, ABC):
    """A base class for time dependent problems. Inherit time-dependent problems from here."""

    def __init__(
        self,
        *args,
        max_num_time_steps: Optional[int] = None,
        time_step_size: Optional[int] = None,
        fix_input_to_time_step: Optional[int] = None,
        allowed_transitions: Optional[list] = None,
        is_spectral_resolver: Optional[bool] = False,
        spectral_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            max_num_time_steps: The maximum number of time steps to use.
            time_step_size: The size of the time step.
            fix_input_to_time_step: If not None, fix the input to this time step.
        """
        assert max_num_time_steps is not None and max_num_time_steps > 0
        assert time_step_size is not None and time_step_size > 0
        assert fix_input_to_time_step is None or fix_input_to_time_step >= 0

        super().__init__(*args, **kwargs)
        self.max_num_time_steps = max_num_time_steps
        self.time_step_size = time_step_size
        self.fix_input_to_time_step = fix_input_to_time_step
        self.allowed_transitions = allowed_transitions
        
        self.is_spectral_resolver = is_spectral_resolver
        self.spectral_file = spectral_file
        if self.is_spectral_resolver:
            self.reader_spectral = nc.Dataset(self.spectral_file, "r")
            self._input_times = self.reader_spectral["time_i"].shape[0]


    def configure_tail_splits(self, requested_val: int, requested_test: int) -> None:
        """Configure train/val/test split sizes for datasets with tail validation/test splits.

        Some ATM-MSC files have fewer trajectories than the historical hard-coded
        validation/test reservations.  Cap the tail splits to the trajectories
        that remain after the requested training trajectories so post_init() can
        always set self.length.
        """
        n_max = int(self.N_max)
        train_count = min(max(1, int(self.num_trajectories)), n_max)
        remaining_after_train = max(0, n_max - train_count)

        self.N_val = min(max(0, int(requested_val)), remaining_after_train)
        remaining_after_val = max(0, remaining_after_train - self.N_val)
        self.N_test = min(max(0, int(requested_test)), remaining_after_val)
        self.num_trajectories = train_count
    
    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        self.max_time_step must have already been set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )

        #print(self.num_trajectories, self.N_val, self.N_test, self.N_max)
        if self.which == "train":
            assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val >= 0
        assert self.N_test is not None and self.N_test >= 0
        assert self.max_num_time_steps is not None and self.max_num_time_steps > 0

        if self.fix_input_to_time_step is not None:
            assert (
                self.fix_input_to_time_step + self.max_num_time_steps
                <= self.max_num_time_steps
            )

            self.multiplier = self.max_num_time_steps
        else:
            if self.allowed_transitions is None:
                self.time_indices = []
                i = 0
                for j in range(i, self.max_num_time_steps + 1):
                    self.time_indices.append((self.time_step_size * i, self.time_step_size * j))
            else:
                self.time_indices = []
                for i in range(self.max_num_time_steps+1):
                    for j in range(i, self.max_num_time_steps + 1):
                        if (j-i) in self.allowed_transitions:
                            self.time_indices.append((self.time_step_size * i, self.time_step_size * j))
            
            self.multiplier = len(self.time_indices)
            print("time_indices", self.time_indices)
        
        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

#--------------------------------------------------------
# Navier-Stokes Datasets:
#--------------------------------------------------------

class NavierStokes2dTimeDataset(BaseTimeDataset):
    def __init__(self, 
                *args,
                rel_time: bool = True,
                is_time: bool = True, 
                **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.rel_time = rel_time
        self.is_time = is_time

        self.N_max = 20000
        self.N_val = 25
        self.N_test = 240

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([0.,1.,1.,0.] + (self.in_dim - 4)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0] + (self.in_dim - 4)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46] + (self.in_dim - 4)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        if self.augment:
            self.augmentations = ["rotation", "transpose"]

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
        
        if self.rel_time:
            t = t2 - t1
        else:
            t = t1
        
        if self.is_time:
            time = float(t / 20.0)
            if time <= 0.:
                time = 1e-6
        else:
            time = None

        #print(torch.from_numpy(self.reader["velocity"][i + self.start, t1]).shape)
        '''
        inputs = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t1, :2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t2, :2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        '''

        inputs = torch.tensor(self.reader['u'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = torch.cat((inputs, torch.tensor(self.reader['v'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        label = torch.tensor(self.reader['u'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        label = torch.cat((label, torch.tensor(self.reader['v'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)

            for i in range(4, self.in_dim):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs, inputs_zeros), 0)
                if i < self.out_dim:
                    label = torch.cat((label, inputs_zeros), 0)

        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.augment:
            inputs, label = self._transform_data_random(self.augmentations, inputs, label)

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

class ShearLayerGenCFDTimeDataset(BaseTimeDataset):
    def __init__(self, 
                *args,
                rel_time: bool = True,
                is_time: bool = True, 
                **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.rel_time = rel_time
        self.is_time = is_time

        self.N_max = 82000 # + 256+512
        self.N_val = 256
        self.N_test = 512

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.870, 0.384], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([0.,1.,1.,0.] + (self.in_dim - 4)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0] + (self.in_dim - 4)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.870, 0.384, 0.46] + (self.in_dim - 4)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        if self.augment:
            self.augmentations = ["rotation", "transpose"]

        self.file_path = '/cluster/work/math/braonic/data/ns_shear2d_gencfd/ne_shear2d_gencfd.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

    def __getitem__(self, idx):

        t1, t2 = 0, 1
        if self.is_time:
            time = 1.0
        else:
            time = None

        inputs = torch.tensor(self.reader['data'][idx + self.start, t1,:,:,0]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = torch.cat((inputs, torch.tensor(self.reader['data'][idx + self.start, t1,:,:,1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        label = torch.tensor(self.reader['data'][idx + self.start, t2,:,:,0]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        label = torch.cat((label, torch.tensor(self.reader['data'][idx + self.start, t2, :,:,1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)

            for i in range(4, self.in_dim):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs, inputs_zeros), 0)
                if i < self.out_dim:
                    label = torch.cat((label, inputs_zeros), 0)

        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std


        if self.is_spectral_resolver:
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['u'][idx + self.start, 0]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['v'][idx + self.start, 0]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        if self.augment:
            inputs, label = self._transform_data_random(self.augmentations, inputs, label)

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label


class ShearLayerGenCFDMicroMacroTimeDataset(BaseTimeDataset):
    def __init__(self, 
                *args,
                curr_macro: int = 0,
                N_micro: int = 128,
                is_time: bool = True, 
                **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.is_time = is_time
        
        self.N_macro = 10
        self.N_micro = N_micro

        self.N_max = 1000 # For the mother class
        self.N_test = self.N_micro # For the mother class
        self.N_val = 0 # For the mother class

        self.N_variety = 32

        self.curr_macro = curr_macro
        assert curr_macro < self.N_macro

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.870, 0.384], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([0.,1.,1.,0.] + (self.in_dim - 4)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0] + (self.in_dim - 4)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.870, 0.384, 0.46] + (self.in_dim - 4)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        self.file_path = '/cluster/work/math/braonic/data/ns_shear2d_gencfd/ns_shear2d_gencfd_macro_micro.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

    def __getitem__(self, idx):

        t1, t2 = 0, 1
        if self.is_time:
            time = 1.0
        else:
            time = None

        inputs = torch.tensor(self.reader['data'][self.curr_macro, idx + self.start, t1,:,:,0]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = torch.cat((inputs, torch.tensor(self.reader['data'][self.curr_macro, idx + self.start, t1,:,:,1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        label = torch.tensor(self.reader['data'][self.curr_macro, idx + self.start, t2,:,:,0]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        label = torch.cat((label, torch.tensor(self.reader['data'][self.curr_macro, idx + self.start, t2, :,:,1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)

            for i in range(4, self.in_dim):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs, inputs_zeros), 0)
                if i < self.out_dim:
                    label = torch.cat((label, inputs_zeros), 0)

        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std


        if self.is_spectral_resolver:
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['u'][idx + self.start, 0]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['v'][idx + self.start, 0]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        if self.augment:
            inputs, label = self._transform_data_random(self.augmentations, inputs, label)

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label


#--------------------------------------------------------
# Magnetohydrodynamics (MHD) Datasets:
#--------------------------------------------------------

class MHD2dTimeDataset(BaseTimeDataset):
    def __init__(self, 
                *args,
                rel_time: bool = True,
                is_time: bool = True,
                reduce_dim = False,
                rescale_time = True,
                ar_training = False,
                **kwargs):
        super().__init__(*args, **kwargs)

        if rescale_time:
            assert self.max_num_time_steps * self.time_step_size <= 20
        else:
            assert self.max_num_time_steps * self.time_step_size <= 100
        
        self.rel_time = rel_time
        self.is_time = is_time
        self.reduce_dim = reduce_dim
        self.rescale_time = rescale_time
        self.ar_training = ar_training
        
        #864
        self.N_max = 1024
        self.N_val = 32
        self.N_test = min(self.num_trajectories, 128)
        
        if self.masked_input is None:
            self.mean = torch.tensor([0.906, 0.0,   0.0,   2.803, 0.0,   0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([2.780, 0.587, 0.619, 1.644, 0.821, 0.751], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            if self.reduce_dim:
                self.mask = torch.tensor([1.,1.,1.,1.] + (self.in_dim - 4)*[0.], dtype=torch.float32)
            else:
                self.mask = torch.tensor([1.,1.,1.,1.,1.,1.] + (self.in_dim - 6)*[0.], dtype=torch.float32)
            
            self.mean = torch.tensor([0.906, 0.0,   0.0,   2.803, 0.0,   0.0] + (self.in_dim - 6)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [2.780, 0.587, 0.619, 1.644, 0.821, 0.751] + (self.in_dim - 6)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

            #self.mean = torch.tensor([0.80, 0.0,   0.0,   2.513, 0.0, 0.0] + (self.in_dim - 6)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            #self.std = torch.tensor([0.31, 0.391, 0.356, 0.185, 0.821, 0.751] + (self.in_dim - 6)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        
        if self.augment:
            self.augmentations = ["rotation", "transpose"]

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
        
        if self.rel_time:
            t = t2 - t1
        else:
            t = t1
        
        if self.is_time:
            if self.rescale_time:
                time = float(t / 20.0)
            else:
                time = float(t / 100.)
            if time <= 0.:
                time = 1e-6
        else:
            time = None

        if self.rescale_time:
            t1*=5
            t2*=5

        inputs = torch.tensor(self.reader['rho'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = torch.cat((inputs, torch.tensor(self.reader['u'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        inputs = torch.cat((inputs, torch.tensor(self.reader['v'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        inputs = torch.cat((inputs, torch.tensor(self.reader['p'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        if not self.reduce_dim:
            inputs = torch.cat((inputs, torch.tensor(self.reader['bx'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            inputs = torch.cat((inputs, torch.tensor(self.reader['by'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        label = torch.tensor(self.reader['rho'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        label = torch.cat((label, torch.tensor(self.reader['u'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        label = torch.cat((label, torch.tensor(self.reader['v'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        label = torch.cat((label, torch.tensor(self.reader['p'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        
        if not self.reduce_dim:
            label = torch.cat((label, torch.tensor(self.reader['bx'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            label = torch.cat((label, torch.tensor(self.reader['by'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        if self.reduce_dim:
            active = 4
        else:
            active = 6

        for i in range(active, self.in_dim):
            inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs, inputs_zeros), 0)
            if i < self.out_dim:
                label = torch.cat((label, inputs_zeros), 0)

        if self.augment:
            inputs, label = self._transform_data_random(self.augmentations, inputs, label)

        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        #print(t, time)
        if self.masked_input is not None:
            if not self.ar_training:
                return time, inputs, label, self.mask
            else:
                return 1/100., t, inputs, label, self.mask
        else:
            if not self.ar_training:
                return time, inputs, label
            else:
                return 1/100., t, inputs, label, self.mask
                
class CompressibleEuler2dTimeDataset(BaseTimeDataset):
    def __init__(self, 
                *args, 
                reduce_dim: bool = False,
                
                **kwargs):
        super().__init__(*args, **kwargs)

        self.reduce_dim = reduce_dim
        if self.augment:
            self.augmentations = ["rotation", "transpose"]
        
        if self.masked_input is None:
            self.mean = torch.tensor([0.80, 0.0,   0.0,   2.513], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.185], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.] + (self.in_dim - 4)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.80, 0.0,   0.0,   2.513] + (self.in_dim - 4)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.185] + (self.in_dim - 4)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

    def __getitem__(self, idx):

        if self.is_spectral_resolver:
            assert self._input_times == len(self.time_indices)

        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        
        time = t / 20.0
        if time <=0:
            time = 1e-6

        inputs = torch.tensor(self.reader['rho'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = torch.cat((inputs, torch.tensor(self.reader['u'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        inputs = torch.cat((inputs, torch.tensor(self.reader['v'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        inputs = torch.cat((inputs, torch.tensor(self.reader['p'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        
        label = torch.tensor(self.reader['rho'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        label = torch.cat((label, torch.tensor(self.reader['u'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        label = torch.cat((label, torch.tensor(self.reader['v'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        label = torch.cat((label, torch.tensor(self.reader['p'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        for i in range(4, self.in_dim):
            inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs, inputs_zeros), 0)
            if i < self.out_dim:
                label = torch.cat((label, inputs_zeros), 0)

        if self.augment:
            inputs, label = self._transform_data_random(self.augmentations, inputs, label)

        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.is_spectral_resolver:
            
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['rho'][i + self.start, _idx]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['u'][i + self.start, _idx]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['v'][i + self.start, _idx]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            inputs = torch.cat((inputs, torch.tensor(self.reader_spectral['p'][i + self.start, _idx]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
            
        if self.reduce_dim:
            inputs = inputs[1:3]
            label = label[1:3]
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

class OrszagTang8TimeDataset(MHD2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.file_path = '/cluster/work/math/camlab-data/synthetic/CEU-MAX_2D_OrszagTang8Uncertainties.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class BrownianBridgeTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = '/cluster/work/math/camlab-data/datazoo/NS-BB.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class VortexSheetTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = '/cluster/work/math/camlab-data/datazoo/NS-SVS.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()


class SinesTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.file_path = '/cluster/work/math/camlab-data/tmp_share/sin.nc' 
        ###'/cluster/work/math/camlab-data/datazoo/NS-Sines.nc'
        
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class SinesEasyTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 5000
        self.N_val = 25
        self.N_test = 240

        #if self.which == "test":
        #    self.N_test = self.num_trajectories

        self.file_path = '/cluster/work/math/camlab-data/data/sin_easy.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()


class PiecewiseConstantsTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.file_path = '/cluster/work/math/camlab-data/synthetic/IEU_2D_PWC.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class GaussiansTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = '/cluster/work/math/camlab-data/synthetic/IEU_2D_Gauss.nc'
        #'/cluster/work/math/camlab-data/datazoo/NS-Gauss.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()
        
        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class ComplicatedShearLayerTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 40000
        self.file_path = '/cluster/work/math/camlab-data/datazoo/NS-SL.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()
    
'''
    Compressible Euler Datasets:
'''

class KelvinHelmholtzTimeDataset(CompressibleEuler2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        self.file_path = '/cluster/work/math/camlab-data/synthetic/CEU_2D_RiemannKelvinHelmholtzLowRes.nc'
        #'/cluster/work/math/camlab-data/datazoo/CE-KH.nc'
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        self.reader = nc.Dataset(self.file_path, "r")
        #self.reader  = file["data"]
        self.post_init()


class RiemannTimeDataset(CompressibleEuler2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        self.file_path = '/cluster/work/math/camlab-data/synthetic/CEU_2D_RiemannLowRes.nc'
        #'/cluster/work/math/camlab-data/datazoo/CE-RP.nc'
        
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = nc.Dataset(self.file_path, "r")
        #self.reader  = file["data"]
        self.post_init()
    
class RiemannCurvedTimeDataset(CompressibleEuler2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        """if self.is_spectral_resolver:
            self.N_max = 9720
            self.N_val = 40
            self.N_test = 0"""

        self.file_path = '/cluster/work/math/camlab-data/synthetic/CEU_2D_RiemannCurvedLowRes.nc'
        #'/cluster/work/math/camlab-data/datazoo/CE-CRP.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()
        
        self.reader = nc.Dataset(self.file_path, "r")
        #self.reader  = file["data"]
        self.post_init()

        

class EulerGaussTimeDataset(CompressibleEuler2dTimeDataset):
    def __init__(self, 
                *args, 
                reduce_dim = True,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        self.file_path = '/cluster/work/math/camlab-data/synthetic/CEU_2D_GaussLowRes.nc'
        #'/cluster/work/math/camlab-data/datazoo/CE-Gauss.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()
        print("FILE PATH IS: ", self.file_path)

        self.reader = nc.Dataset(self.file_path, "r")
        #self.reader  = file["data"]
        #self.reader = h5py.File(data_path, "r")
        self.post_init()    

class RiemannKHTimeDataset(CompressibleEuler2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240

        self.file_path = '/cluster/work/math/camlab-data/synthetic/CEU_2D_RiemannKelvinHelmholtzLowRes.nc'
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self.reader = nc.Dataset(self.file_path, "r")
        #self.reader  = file["data"]
        #self.reader = h5py.File(data_path, "r")
        self.post_init()   

#--------------------------------------------------------
# Richtmyer-Meshkov Experiment:
#--------------------------------------------------------
class RichtmyerMeshkov(BaseTimeDataset):
    def __init__(self, *args, tracer = False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 1260
        self.N_val = 100
        self.N_test = 130
        self.resolution = 128
        self.tracer = tracer
        self.mask = torch.tensor([1.,1.,1.,1.], dtype=torch.float32)
        if self.in_dist:
            data_path = '/cluster/work/math/camlab-data/compressible_flow/richtmyer_meshkov.nc'
        
        else:
            raise NotImplementedError()

        self.reader = nc.Dataset(data_path, "r")
        
        self.label_description = (
            "[rho],[u,v],[p]" if not tracer else "[rho],[u,v],[p],[tracer]"
        )
        
        self.constants = {
            "mean": torch.tensor(
                [1.1964245, -7.164812e-06, 2.8968952e-06, 1.5648036]
            ).unsqueeze(1).unsqueeze(1),
            "std": torch.tensor(
                [0.5543239, 0.24304213, 0.2430597, 0.89639103]
            ).unsqueeze(1).unsqueeze(1),
            "time": 20.0,
            "tracer_mean": 1.3658239,
            "tracer_std": 0.46400866,
        }
        
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        
        
        inputs = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        label = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]
        
        if self.tracer:
            input_tracer = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t1, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            output_tracer = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t2, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            inputs = torch.cat([inputs, input_tracer], dim=0)
            label = torch.cat([label, output_tracer], dim=0)

        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label



#--------------------------------------------------------
# Rayleigh-Taylor Experiment (Euler + Force):
#--------------------------------------------------------

class RayleighTaylor(BaseTimeDataset):
    def __init__(self, *args, tracer=False,  **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 10

        self.N_max = 1260
        self.N_val = 100
        self.N_test = 130
        self.resolution = 128
        self.tracer = tracer
        
        if self.in_dist:
            data_path = self.data_path + '/rayleigh_taylor.nc'
        
        else:
            raise NotImplementedError()

        self.reader = nc.Dataset(data_path, "r")
        
        self.label_description = (
            "[rho],[u,v],[p],[g]" if not tracer else "[rho],[u,v],[p],[tracer],[g]"
        )
        
        self.constants = {
            "mean": torch.tensor(
               [0.8970493, 4.0316996e-13, -1.3858967e-13, 0.7133829, -1.7055787]
            ).unsqueeze(1).unsqueeze(1),
            "std": torch.tensor(
                [0.12857835, 0.014896976, 0.014896975, 0.21293919, 0.40131348]
            ).unsqueeze(1).unsqueeze(1),
            "time": 10.0,
            "tracer_mean": 1.8061695,
            "tracer_std": 0.37115487,
        }
        
        print(self.which, self.N_test, self.N_max - self.N_test, "CHECK")
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]
        
        
        inputs = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )

        g_1 = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t1, 5:6])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        g_2 = (
            torch.from_numpy(self.reader.variables["solution"][i + self.start, t2, 5:6])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.constants["mean"][:4]) / self.constants["std"][:4]
        g_1 = (g_1 - self.constants["mean"][4]) / self.constants["std"][4]
        g_2 = (g_2 - self.constants["mean"][4]) / self.constants["std"][4]
        label = (label - self.constants["mean"][:4]) / self.constants["std"][:4]
        
        
        if self.tracer:
            tracer_1 = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t1, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            tracer_2 = (
                torch.from_numpy(
                    self.reader.variables["solution"][i + self.start, t2, 4:5]
                )
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            tracer_1 = (tracer_1 - self.constants["tracer_mean"]) / self.constants[
                "tracer_std"
            ]
            tracer_2 = (tracer_2 - self.constants["tracer_mean"]) / self.constants[
                "tracer_std"
            ]
            inputs = torch.cat([inputs, tracer_1, g_1], dim=0)
            label = torch.cat([label, tracer_2, g_2], dim=0)
        else:
            inputs = torch.cat([inputs, g_1], dim=0)
            label = torch.cat([label, g_2], dim=0)
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

#--------------------------------------------------------
# Allen-Cahn/Cahn-Hilliard Equation:
#--------------------------------------------------------

class CahnEquations(BaseTimeDataset):
    def __init__(self, 
                *args,
                is_allen_cahn = True,
                 **kwargs):
        super().__init__(*args, 
                        **kwargs)
        
        self.is_allen_cahn = is_allen_cahn
        if is_allen_cahn:
            assert self.max_num_time_steps * self.time_step_size <= 19
            self.N_max = 15000
            self.N_val = 60
            self.N_test = 240
            self.resolution = 128
            
            
            #POSEIDON ALLEN_CAHN:

            self.file_path = '/cluster/work/math/camlab-data/synthetic/ALC_2D_Sin.nc'
            self.constants = {
            "mean": 0.002484262,
            "std": 0.65351176,
            "time": 20.0,
            }
            

            """
            self.file_path = '/cluster/work/math/camlab-data/datazoo/allen_cahn_ch.nc'
            self.constants = {
            "mean": 0.5,
            "std": 0.47,
            "time": 20.0,
            }
            """
        else:
            assert self.max_num_time_steps * self.time_step_size <= 41
            self.N_max = 4096
            self.N_val = 60
            self.N_test = 120
            self.resolution = 128
            self.file_path = '/cluster/work/math/braonic/data/cahn_hilliard_ace.nc'
            self.constants = {
            "mean": 0.5001,
            "std": 0.4338,
            "time": 20.0,
            }
        
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()
        self.reader = nc.Dataset(self.file_path, "r")

        if self.augment:
            self.augmentations = ["rotation", "transpose"]
        
        if self.masked_input is not None:
            '''
                We need in to train the multi-physics FM 
                --> in_dim = 9, out_dim = 9
                --> channel 6 (wave channel) in the I/O -- correspond to ACE/CH

                self.label_description = "[u]"
            '''

            self.mask = torch.tensor(max(self.out_dim - 4,0) * [0.] + [1.] +  3 * [0.], dtype=torch.float32)
            self.mean = torch.tensor(max(self.out_dim - 4,0) * [0.] + [self.constants["mean"]] +  3 * [0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std  = torch.tensor(max(self.out_dim - 4,0) * [1.] + [self.constants["std"]]  +  3 * [1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]


        if not self.is_allen_cahn:
            inputs = (
                torch.from_numpy(self.reader.variables["solution"][i + self.start, t1])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            labels = (
                torch.from_numpy(self.reader.variables["solution"][i + self.start, t2])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
        else:
            inputs = (
                torch.from_numpy(self.reader.variables["u"][i + self.start, t1])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            labels = (
                torch.from_numpy(self.reader.variables["u"][i + self.start, t2])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        if self.masked_input is not None:
            for i in range(max(self.out_dim - 4,0)):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs_zeros, inputs), 0)
                labels = torch.cat((inputs_zeros, labels), 0)
            for i in range(3):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs,inputs_zeros), 0)
                labels = torch.cat((labels, inputs_zeros), 0)

        if self.augment:
            inputs, labels = self._transform_data_random(self.augmentations, inputs, labels)

        if self.masked_input is not None:
            return time, inputs, labels, self.mask
        else:
            return time, inputs, labels

#--------------------------------------------------------
# MERRA2:
#--------------------------------------------------------

class MERRA2Dataset(BaseDataset):
    def __init__(self, 
                allowed_transitions,
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs)

        if self.augment:
            self.augmentations = []

        if self.masked_input is None:
            self.mean = torch.tensor([1.1768, -0.09746, 0.3835, 292.3969, 0.01158], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.0696, 5.797, 5.2087, 7.7286, 0.0050138], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([1.1768, -0.09746, 0.3835, 292.3969, 0.01158] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.0696, 5.797, 5.2087, 7.7286, 0.0050138] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        self.allowed_transitions = allowed_transitions
        self.allowed_indicies = None
        self.multiplier = len(self.allowed_transitions)
        _max_transition = np.max(np.array(self.allowed_transitions))
        
        if self.which == "train":
            self.file_path = "/cluster/work/math/braonic/data/merra2_2022_273d_alltime_rho_u_v_T_humidity_cbl.nc"
            #self.N_max = min(self.data.shape[0] - _max_transition, self.num_trajectories)
            self.N_max = min(6552, self.num_trajectories)
            self.length = self.N_max * self.multiplier
        elif self.which == "val":
            self.file_path = "/cluster/work/math/braonic/data/merra2_2023_91d_alltime_rho_u_v_T_humidity_cbl.nc"
            self.allowed_indicies =  [154, 1069,  395, 1006,  174, 1305, 1940, 1055, 1070, 1497, 1719, 1416, 1714, 1965, 1011,  655, 2120,  749,  630,  744, 1218, 1367, 1843,  397, 1157,   92,  279,  791,  243, 1469, 2080, 1656]
            self.N_max = len(self.allowed_indicies)
            self.length = self.N_max * self.multiplier
        elif self.which == "test":
            self.file_path = "/cluster/work/math/braonic/data/merra2_2023_91d_alltime_rho_u_v_T_humidity_cbl.nc"
            self.allowed_indicies =  np.arange(0,2000, 4)
            self.N_max = min(len(self.allowed_indicies), self.num_trajectories)
            self.length = self.N_max * self.multiplier
        
        self.reader = nc.Dataset(self.file_path, "r")
        self.data = self.reader.variables["data"]
        self.start = 0

    def __getitem__(self, idx):
        i = idx // self.multiplier
        if self.allowed_indicies is not None:
            i = self.allowed_indicies[i]
        jump = self.allowed_transitions[idx % self.multiplier]
        time = jump / 24.
        if time <= 0:
            time = 1e-6

        inputs = torch.tensor(self.data[i + self.start,0]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = torch.cat((inputs, torch.tensor(self.data[i + self.start,1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        inputs = torch.cat((inputs, torch.tensor(self.data[i + self.start,2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        inputs = torch.cat((inputs, torch.tensor(self.data[i + self.start,3]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        inputs = torch.cat((inputs, torch.tensor(self.data[i + self.start,4]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        label = torch.tensor(self.data[i + jump +  self.start, 0]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        label = torch.cat((label, torch.tensor(self.data[i + jump + self.start, 1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        label = torch.cat((label, torch.tensor(self.data[i + jump + self.start, 2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        label = torch.cat((label, torch.tensor(self.data[i + jump + self.start, 3]).type(torch.float32).reshape(1, self.resolution, self.resolution)))
        label = torch.cat((label, torch.tensor(self.data[i + jump + self.start, 4]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        for i in range(5, self.in_dim):
            inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs, inputs_zeros), 0)
            if i < self.out_dim:
                label = torch.cat((label, inputs_zeros), 0)

        #if self.augment:
        #    inputs, label = self._transform_data_random(self.augmentations, inputs, label)

        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label


#--------------------------------------------------------
# Poisson Equation:
#--------------------------------------------------------

class PoissonBase(BaseDataset):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_max = 20000
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128

        self.file_path = "/cluster/work/math/camlab-data/synthetic/POI_2D_Gauss.nc" #os.path.join(self.data_path, file_path)
        self.reader = nc.Dataset(self.file_path, "r")
        self.constants = {
                        "mean_source": 0.014822142414492256,
                        "std_source": 4.755138816607612,
                        "mean_solution": 0.0005603458434937093,
                        "std_solution": 0.02401226126952699,
                        }

        self.input_dim = 1
        self.label_description = "[u]"

        if self.masked_input is not None:
            
            self.mask = torch.tensor(max(self.out_dim - 3,0) * [0.] + [1., 0.] +  [0.] , dtype=torch.float32)
            self.mean = torch.tensor(max(self.out_dim - 3,0) * [0.] + [0.0005603458434937093, 0.14822142414492256] + [0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std  = torch.tensor(max(self.out_dim - 3,0) * [1.] + [0.1, 4.755138816607612] + [1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        self.post_init()

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.reader.variables["f"][idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (
            torch.from_numpy(self.reader.variables["u"][idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean_source"]) / self.constants["std_source"]
        labels = (labels - self.constants["mean_solution"]) / self.constants[
            "std_solution"
        ]
        
        if self.masked_input is None:
            return 1.0, inputs, labels
        else:
            labels = torch.cat((labels, inputs), 0)
            for i in range(self.out_dim - 2):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs_zeros, inputs), 0)
                if i < self.out_dim - 3:
                    labels = torch.cat((inputs_zeros, labels), 0)
                else:
                    labels = torch.cat((labels, inputs_zeros), 0)
            
            inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs, inputs_zeros), 0)

        return 1.0, inputs, labels, self.mask
        

class PoissonGaussians(PoissonBase):
    def __init__(self, *args, **kwargs):
        # mean_source = 0.0608225485107185
        # std_source = 0.18010304094287755
        # mean_solution = 0.002326384792539932
        # std_solution = 0.05859948481241117
        super().__init__("/cluster/work/math/camlab-data/synthetic/POI_2D_Gauss.nc", *args, **kwargs)


#--------------------------------------------------------
# Helmholts Equation:
#--------------------------------------------------------

class Helmholtz(BaseDataset):
    def __init__(self, 
                *args,
                oversample = 1, 
                **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 19675
        self.N_val = 128
        self.N_test = 512
        self.resolution = 128

        self.oversample = oversample
        self.multiplier = oversample

        self.file_path = '/cluster/work/math/camlab-data/synthetic/HEL_2D_Gauss.nc'
        #"/cluster/work/math/braonic/data/helmholtz_poseidon/Helmholtz.h5"
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()
        self.reader = h5py.File(self.file_path, "r")
        
        if self.masked_input is None:
            self.mean = 0.11523915668552
            self.std = 0.8279975746000605
        else:

            '''
                We need in to train the multi-physics FM 
                --> in_dim = 9, out_dim = 9
                --> channel 7 (static channel) in the output (0. in the input) -- correspond to Helmholtz
                --> 8 (forcing/spatial dependencies) and 9 (BCs) --> correspond to Helmholtz
                --> We predict the channel 8 in the output as well
                --> we mask out BCs in the loss function

                self.label_description = "[u]"
            '''

            self.mask = torch.tensor(6 * [0.] + [1., 0.] +  [0.] , dtype=torch.float32)
            self.mean = torch.tensor(6 * [0.] + [0.11523915668552, 0.] + [0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std  = torch.tensor(6 * [1.] + [0.8279975746000605, 1.] + [1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        """if self.augment:
            angles = [0, 90, 180, 270] #Rotation angles
            flip = [0, 1, 2] #Transpositions
            boundary = [0, 1] #Boundary Conditions (whole channel or frame)
            self.combinations = list(product(angles, flip, boundary))
            self.multiplier = self.oversample * len(self. combinations)"""
        
        
        if self.augment:
            self.augmentations = ["rotation"]
        
        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

    def __getitem__(self, idx):

        _idx = idx // self.multiplier

        inputs = (
            torch.from_numpy(self.reader['a'][_idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs = inputs - 1
        b = float(np.array(self.reader["bc"][_idx + self.start]))
        bc = b * torch.ones_like(inputs)

        labels = (
            torch.from_numpy(self.reader["u"][_idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        if self.masked_input is None:
            inputs = torch.cat((inputs, bc), dim=0)
        else:
            labels = torch.cat((labels, inputs), 0)
            for i in range(7):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs_zeros, inputs), 0)
                if i<6:
                    labels = torch.cat((inputs_zeros, labels), 0)
                else:
                    labels = torch.cat((labels, inputs_zeros), 0)
            
            #if self.augment:
            #    policy = list(self.combinations[(idx % self.multiplier) % len(self.combinations)])
            #bc[0] = self._transform_boundary(bc[0], which = policy[-1])
            
            inputs = torch.cat((inputs, bc), 0)
            
            #if self.augment:
            #    inputs, labels = self._rotate(inputs, labels, angle = policy[0])
            #    inputs, labels = self._transpose(inputs, labels, flip = policy[1])

            #print(inputs.shape, labels.shape, i,idx, policy)
        
        labels = (labels - self.mean) / self.std

        if self.augment:
            inputs, labels = self._transform_data_random(self.augmentations, inputs, labels)

        if self.masked_input is not None:
            return 1., inputs, labels, self.mask
        else:
            return 1., inputs, labels

#--------------------------------------------------------
# Airfoil Dataset (Steady):
#--------------------------------------------------------

class Airfoil(BaseDataset):
    def __init__(self, *args, tracer=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 10869
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/compressible_flow/steady/airfoil.nc"
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.92984116,
            "std": 0.10864315,
        }

        self.label_description = "[rho]"

        self.post_init()

    def __getitem__(self, idx):
        
        time = 1.0

        inputs = (
            torch.from_numpy(self.reader["solution"][idx + self.start, 0])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][idx + self.start, 1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (labels - self.constants["mean"]) / self.constants["std"]
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        return time, inputs, labels
    
#--------------------------------------------------------
# Wave Equation:
#--------------------------------------------------------

class WaveSeismic(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10512
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = "/cluster/work/math/camlab-data/synthetic/AWA_2D_Layer.nc"
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.03467443221585092,
            "std": 0.10442421752963911,
            "mean_c": 3498.5644380917424,
            "std_c": 647.843958567462,
            "time": 20.0,
        }

        self.label_description = "[u],[c]"
        self.mask = torch.tensor([1.,0.], dtype=torch.float32)

        if self.masked_input is not None:
            '''
                We need in to train the multi-physics FM 
                --> in_dim = 9, out_dim = 9
                --> channel 5 (wave channel) in the I/O -- correspond to Wave
                --> 8 (forcing/spatial dependencies -- )correspond to Wave
                --> We predict the channel 8 in the output as well
                --> we mask out BCs in the loss function

                self.label_description = "[u]"
            '''
            #self.label_description = "[u],[c]"

            self.mask = torch.tensor(max(self.out_dim - 5,0) * [0.] + [1., 0., 0., 1.,] + max(self.out_dim - 8,0)*[0.] , dtype=torch.float32)
            self.mean = torch.tensor(max(self.out_dim - 5,0) * [0.] + [0.03467443221585092, 0., 0., 3498.5644380917424] + max(self.out_dim - 8,0)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std  = torch.tensor(max(self.out_dim - 5,0) * [1.] + [0.10442421752963911, 1., 1., 647.843958567462] + max(self.out_dim - 8,0)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        self.post_init()

    def __getitem__(self, idx):
        
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]
        

        inputs = (
            torch.from_numpy(self.reader["u"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs_c = (
            torch.from_numpy(self.reader["c"][i + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["u"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        if self.masked_input is None:
            inputs = (inputs - self.constants["mean"]) / self.constants["std"]
            inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
            labels = (labels - self.constants["mean"]) / self.constants["std"]

            inputs = torch.cat([inputs, inputs_c], dim=0)
            labels = torch.cat([labels, inputs_c], dim=0)
        else:
    
            for i in range(self.out_dim - 5):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs_zeros, inputs), 0)
                labels = torch.cat((inputs_zeros, labels), 0)
            for i in range(2):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs,inputs_zeros), 0)
                labels = torch.cat((labels, inputs_zeros), 0)
            inputs = torch.cat([inputs, inputs_c], dim=0)
            labels = torch.cat([labels, inputs_c], dim=0)

            if self.out_dim>=9:
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs,inputs_zeros), 0)
                labels = torch.cat((labels, inputs_zeros), 0)
            inputs = (inputs - self.mean) / self.std
            labels = (labels-self.mean) / self.std

        if self.masked_input is not None:
            return time, inputs, labels, self.mask
        else:
            return time, inputs, labels

class WaveGaussians(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 15

        self.N_max = 10512
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128
        
        #data_path = self.data_path + "/wave_equation/gaussians_15step.nc"
        self.file_path = '/cluster/work/math/camlab-data/synthetic/AWA_2D_Gauss.nc'
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()
        self.reader = h5py.File(self.file_path, "r")

        if self.augment:
            self.augmentations = ["rotation", "transpose"]
        
        self.constants = {
            "mean": 0.0334376316,
            "std": 0.1171879068,
            "mean_c": 2618.4593933,
            "std_c": 601.51658913,
            "time": 15.0,
        }
        
        if self.masked_input is not None:
            '''
                We need in to train the multi-physics FM 
                --> in_dim = 9, out_dim = 9
                --> channel 5 (wave channel) in the I/O -- correspond to Wave
                --> 8 (forcing/spatial dependencies -- )correspond to Wave
                --> We predict the channel 8 in the output as well
                --> we mask out BCs in the loss function

                self.label_description = "[u]"
            '''
            #self.label_description = "[u],[c]"

            '''
            self.mask = torch.tensor(4 * [0.] + [1., 0., 0., 1., 0.] , dtype=torch.float32)
            self.mean = torch.tensor(4 * [0.] + [0.0334376316, 0., 0., 2618.4593933] + [0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std  = torch.tensor(4 * [1.] + [0.1171879068, 1., 1., 601.51658913] + [1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            '''
            self.mask = torch.tensor(4 * [0.] + [1., 0., 0., 1., 0.] , dtype=torch.float32)
            self.mean = torch.tensor(4 * [0.] + [0.03467443221585092, 0., 0., 3498.5644380917424] + [0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std  = torch.tensor(4 * [1.] + [0.10442421752963911, 1., 1., 647.843958567462] + [1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["u"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs_c = (
            torch.from_numpy(self.reader["c"][i + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["u"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        if self.masked_input is None:
            inputs = (inputs - self.constants["mean"]) / self.constants["std"]
            inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
            labels = (labels - self.constants["mean"]) / self.constants["std"]

            inputs = torch.cat([inputs, inputs_c], dim=0)
            labels = torch.cat([labels, inputs_c], dim=0)
        else:
    
            for i in range(4):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs_zeros, inputs), 0)
                labels = torch.cat((inputs_zeros, labels), 0)
            for i in range(2):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs,inputs_zeros), 0)
                labels = torch.cat((labels, inputs_zeros), 0)
            inputs = torch.cat([inputs, inputs_c], dim=0)
            labels = torch.cat([labels, inputs_c], dim=0)
            inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs,inputs_zeros), 0)
            labels = torch.cat((labels, inputs_zeros), 0)
            inputs = (inputs - self.mean) / self.std
            labels = (labels-self.mean) / self.std


        if self.augment:
            inputs, labels = self._transform_data_random(self.augmentations, inputs, labels)

        if self.masked_input is not None:
            return time, inputs, labels, self.mask
        else:
            return time, inputs, labels


#--------------------------------------------------------
# Kolmogorov:
#--------------------------------------------------------
    
class KolmogorovFlow(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/incompressible_fluids/forcing/kolmogorov_pwc.nc"
        self.reader = h5py.File(data_path, "r")

        self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.22, 0.22],  dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std_forcing = 0.0707
        
        
        X, Y = torch.meshgrid(
            torch.linspace(0, 1, self.resolution),
            torch.linspace(0, 1, self.resolution),
            indexing="ij",
        )
        f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
        self.forcing = f(X, Y).unsqueeze(0)
        self.forcing = self.forcing / self.std_forcing
        
        self.label_description = "[u,v],[f]"
    
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0
        

        inputs_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )

        inputs_v = (inputs_v - self.mean) / self.std
        label_v = (label_v - self.mean) / self.std
        
        inputs  = torch.cat((inputs_v, self.forcing), 0)
        label_v = torch.cat((label_v, self.forcing), 0)
        
        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        
        
        if self.masked_input is not None:
            return time, inputs, label_v, self.mask
        else:
            return time, inputs, label_v
        
#-------------------------
# Navier-Stokes Tracers:
#-------------------------

class PiecewiseConstantsTraceTimeDataset(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 20000
        self.N_val = 40
        self.N_test = 240

        if self.in_dist:
            data_path = self.data_path + "/pwc_tracer.nc"
        else:
            raise NotImplementedError()

        self.reader = h5py.File(data_path, "r")
        #0.391, 0.356
        #0.49198706571149564, 0.36194905497513363
        self.mean = torch.tensor([0,0,0.19586183], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor([0.391, 0.356,0.37], dtype=torch.float32).unsqueeze(1).unsqueeze(1)


        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / 20.0

        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t1])
            .type(torch.float32)
            .reshape(3, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][:][t2])
            .type(torch.float32)
            .reshape(3, self.resolution, self.resolution)
        )
        
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.time_input:
            inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
            inputs = torch.cat((inputs, inputs_t), 0)
        
        return time, inputs, label


#-------------------------
# Compressible-Euler 3D:
#-------------------------

class CompressibleEulerNavierStokes3dTimeDataset(BaseTimeDataset):
    def __init__(self, 
                *args, 
                reduce_dim = False,
                perturb_p = False,
                curr_macro = None,
                **kwargs):
        super().__init__(*args, **kwargs)

        self.reduce_dim = reduce_dim
        
        self.perturb_b = perturb_p # For eul_ns3d_mix1 (TURBO) training
        self.micro_macro = curr_macro
        if self.micro_macro is not None:
            self.resolution = 128

        if self.augment:
            self.augmentations = ["rotation", "transpose"]

        if self.masked_input is None:

            if self.in_dim == 5:
                self.mean = torch.tensor([2.494, 0.0, 0.0, 0.0, 1.735] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                self.std = torch.tensor( [0.534, 0.255, 0.255,  0.255, 0.438] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            elif self.in_dim == 3:
                self.mean = torch.tensor([0.0, 0.0, 0.0] + (self.in_dim - 3)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                self.std = torch.tensor( [0.255, 0.255,  0.255] + (self.in_dim - 3)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            
        self.is_compressible = True

    def __getitem__(self, idx):
        
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier
        
        #print(i, _idx)

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        
        time = t / 20.0
        if time <=0:
            time = 1e-6

        if self.micro_macro is not None:
            t1_indices = (self.micro_macro, i + self.start, t1)
            t2_indices = (self.micro_macro, i + self.start, t2)
        else:
            t1_indices = (i + self.start, t1)
            t2_indices = (i + self.start, t2)

        if self.is_compressible:
            rho_np = self.reader['rho'][*t1_indices]
            rho = torch.from_numpy(rho_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            #rho = torch.tensor(,dtype=torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            
            mx_np = self.reader['mx'][*t1_indices]
            #mx = torch.tensor(,dtype=torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            mx = torch.from_numpy(mx_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)

            my_np = self.reader['my'][*t1_indices]
            #my =  torch.tensor(,dtype=torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            my = torch.from_numpy(my_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            
            mz_np = self.reader['mz'][*t1_indices]
            #mz = torch.tensor(,dtype=torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            mz = torch.from_numpy(mz_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            
            E_np = self.reader['E'][*t1_indices]
            #E =  torch.tensor(,dtype=torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            E = torch.from_numpy(E_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            
            inputs = self.to_primitive(rho, mx, my, mz, E)
        else:        
            print(t1_indices, "HERE", self.reader['u'].shape, i, i + self.start)
            u = torch.tensor(self.reader['u'][*t1_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            v =  torch.tensor(self.reader['v'][*t1_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            w = torch.tensor(self.reader['w'][*t1_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            inputs = torch.cat((u, v, w), axis=0)

        if self.is_compressible:

            rho_np = self.reader['rho'][*t2_indices]
            rho = torch.from_numpy(rho_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            
            mx_np = self.reader['mx'][*t2_indices]
            mx = torch.from_numpy(mx_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)

            my_np = self.reader['my'][*t2_indices]
            my = torch.from_numpy(my_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            
            mz_np = self.reader['mz'][*t2_indices]
            mz = torch.from_numpy(mz_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            
            E_np = self.reader['E'][*t2_indices]
            E = torch.from_numpy(E_np.astype('float32', copy=False, order='C')).reshape(1, self.resolution, self.resolution, self.resolution)
            
            label = self.to_primitive(rho, mx, my, mz, E)
        else:
            u = torch.tensor(self.reader['u'][*t2_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            v =  torch.tensor(self.reader['v'][*t2_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            w = torch.tensor(self.reader['w'][*t2_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
            label = torch.cat((u, v, w), axis=0)
        
        if not self.is_compressible and self.in_dim >= 5:
            inputs_ones = torch.ones((1, self.resolution, self.resolution, self.resolution)).type(torch.float32)
            inputs_zeros = torch.zeros((1, self.resolution, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_ones, inputs), 0)
            
            if not self.perturb_b:
                inputs = torch.cat((inputs, inputs_zeros), 0)
            else:
                inputs = torch.cat((inputs, inputs_zeros + 0.2), 0)
            
            label = torch.cat((inputs_ones, label), 0)

            if not self.perturb_b:
                label = torch.cat((label, inputs_zeros), 0)
            else:
                label = torch.cat((label, inputs_zeros + 0.2), 0)

        if self.in_dim>5:
            for i in range(5, self.in_dim):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs, inputs_zeros), 0)
                if i < self.out_dim:
                    label = torch.cat((label, inputs_zeros), 0)

        #print(inputs.shape, self.mean.shape, self.std.shape)
        inputs = (inputs - self.mean) / self.std
        label = (label-self.mean) / self.std

        if self.reduce_dim:
            inputs = inputs[1:3]
            label = label[1:3]

        if self.micro_macro is not None:
            inputs = inputs[:,::2,::2,::2]
            label = label[:,::2,::2,::2]

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label
    
    def to_primitive(self, rho, mx, my, mz, E):
        vx = mx / rho
        vy = my / rho
        vz = mz / rho
        P = 0.4 * (E - 0.5 * (mx ** 2 + my ** 2 + mz ** 2) / rho)
        
        return torch.cat((rho, vx, vy, vz, P), axis=0)

class ATMMSC3DMoistTimeDataset(BaseTimeDataset):
    def __init__(self,
                *args,
                time_offset: int = 9,
                max_z: Optional[int] = 192,
                embed_parameters: bool = True,
                **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = "/cluster/work/math/camlab-data/synthetic/ATM-MSC_3D_moist.nc"
        self.params_path = "/cluster/work/math/camlab-data/synthetic/_zsetup_ATM-MSC_3D_moist/ATM-MSC_3D_moist_input_parameters.npy"

        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        self.reader = nc.Dataset(self.file_path, "r")
        self.variable_names = ["u", "v", "w", "temperature", "qt"]
        self.output_variable_names = list(self.variable_names)
        self.base_channels = len(self.variable_names)
        self.time_offset = time_offset
        self.embed_parameters = embed_parameters
        self.max_z = max_z

        self.mean = torch.tensor([0.0, 0.0, 0.0, 290.0, 0.006], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor([1.0, 1.0, 1.0, 4.0, 0.004], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if self.masked_input is not None:
            self.mask = torch.tensor([1.0] * min(self.out_dim, self.base_channels) + [0.0] * max(self.in_dim - self.base_channels, 0), dtype=torch.float32)

        # Parameters that are not part of the initial conditions.
        self.params_idx = np.array([0, 1, 5, 6, 7, 8])
        self.params = np.load(self.params_path)[:, self.params_idx].astype(np.float32)
        self.param_mean = np.array([0.0, 0.0, 291.0, 0.00099, 0.0, 0.25], dtype=np.float32)
        self.param_std = np.array([14.7, 7.74e-06, 2.0, 0.00022, 0.2, 0.75], dtype=np.float32)

        any_var = self.reader[self.variable_names[0]]
        self.native_spatial_shape = tuple(int(x) for x in any_var.shape[-3:])
        self.spatial_shape = (
            self.native_spatial_shape[0],
            self.native_spatial_shape[1],
            min(int(self.max_z), self.native_spatial_shape[2]) if self.max_z is not None else self.native_spatial_shape[2],
        )
        
        self.total_time_steps = int(any_var.shape[1])
        self.N_max = int(any_var.shape[0])
        

        requested_val = min(26, max(1, self.N_max // 10))
        self.configure_tail_splits(requested_val=requested_val, requested_test=32)

        self.N_val = min(26, max(1, self.N_max // 10))
        self.N_test = 32
        if (
            type(self) is ATMMSC3DMoistTimeDataset
            and self.num_trajectories + self.N_val + self.N_test < self.N_max
        ):
            print(self.N_max, "N_MAX")
            self.num_trajectories = max(1, self.N_max - self.N_val - self.N_test)
            self.post_init()

    def _read_field(self, var_name: str, sample_index: int, time_index: int) -> np.ndarray:
        x = np.asarray(self.reader[var_name][sample_index, time_index], dtype=np.float32)
        if x.ndim != 3:
            x = np.squeeze(x)
        target_z = self.spatial_shape[2]
        if x.shape[-1] > target_z:
            x = x[..., :target_z]
        return x

    def _embed_parameters_as_channels(self, x: torch.Tensor, sample_index: int) -> torch.Tensor:
        if not self.embed_parameters:
            return x

        num_param_channels = max(self.in_dim - self.base_channels, 0)
        if num_param_channels == 0:
            return x

        params = (self.params[sample_index] - self.param_mean) / self.param_std
        d, h, w = x.shape[-3:]
        param_tensor = torch.zeros((num_param_channels, d, h, w), dtype=x.dtype)

        fill_channels = min(num_param_channels, len(params))
        for i in range(fill_channels):
            param_tensor[i].fill_(float(params[i]))

        return torch.cat((x, param_tensor), dim=0)

    def _match_output_channels(self, y: torch.Tensor) -> torch.Tensor:
        if self.out_dim <= self.base_channels:
            return y[:self.out_dim]

        d, h, w = y.shape[-3:]
        pad_channels = self.out_dim - self.base_channels
        y_pad = torch.zeros((pad_channels, d, h, w), dtype=y.dtype)
        return torch.cat((y, y_pad), dim=0)

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        t1, t2 = self.time_indices[_idx]
        t1 += self.time_offset
        t2 += self.time_offset

        if t2 >= self.total_time_steps:
            raise IndexError(
                f"Requested time index {t2} exceeds available ATM_MSC_3D_moist range [0, {self.total_time_steps - 1}]"
            )

        sample_index = i + self.start

        x_np = np.stack([self._read_field(v, sample_index, t1) for v in self.variable_names], axis=0)
        y_np = np.stack([self._read_field(v, sample_index, t2) for v in self.variable_names], axis=0)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = self._embed_parameters_as_channels(x, sample_index)
        y = self._match_output_channels(y)


        lead_time = float(t2 - t1) / 24.0
        if self.masked_input is not None:
            return lead_time, x, y, self.mask
        return lead_time, x, y


class ATMMSC3DDryTimeDataset(ATMMSC3DMoistTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
	
        self.file_path = "/cluster/work/math/camlab-data/synthetic/ATM-CBL_3D_dry.nc"
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        if hasattr(self, "reader") and self.reader is not None:
            self.reader.close()
        self.reader = nc.Dataset(self.file_path, "r")

        self.variable_names = ["u", "v", "w", "temperature"]
        self.output_variable_names = list(self.variable_names)
        self.base_channels = len(self.variable_names)

        self.mean = torch.tensor([0.0, 0.0, 0.0, 290.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor([1.0, 1.0, 1.0, 4.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if self.masked_input is not None:
            self.mask = torch.tensor(
                [1.0] * min(self.out_dim, self.base_channels) + [0.0] * max(self.in_dim - self.base_channels, 0),
                dtype=torch.float32,
            )

        _X = np.array(self.reader.variables["Ug"]).reshape(-1, 1)
        _Y = np.array(self.reader.variables["Qstar"]).reshape(-1, 1)
        self.params_idx = np.array([0, 1])
        self.params = np.concatenate((_X, _Y), axis=1)[:, self.params_idx].astype(np.float32)
        self.param_mean = np.array([2.5, 0.2], dtype=np.float32)
        self.param_std = np.array([1.5, 0.06], dtype=np.float32)

        any_var = self.reader[self.variable_names[0]]
        self.native_spatial_shape = tuple(int(x) for x in any_var.shape[-3:])
        self.spatial_shape = (
            self.native_spatial_shape[0],
            self.native_spatial_shape[1],
            min(int(self.max_z), self.native_spatial_shape[2]) if self.max_z is not None else self.native_spatial_shape[2],
        )
        self.total_time_steps = int(any_var.shape[1])
        # Keep dry splits tied to the configured cap, not full dataset size.
        self.N_max = 8500
        self.N_val = min(32, max(1, self.N_max // 10))
        self.N_test = 32

        # Ensure split counts are feasible for the actual file size.
        available_samples = int(any_var.shape[0])
        self.N_max = min(self.N_max, available_samples)
        print(available_samples, "DRY DRY DRY")
        if self.num_trajectories + self.N_val + self.N_test > self.N_max:
            self.num_trajectories = max(1, self.N_max - self.N_val - self.N_test)

        # Recompute multiplier/start/length with the dry split settings.
        self.post_init()


class ConditionalATMMSC3DMoistTimeDataset(BaseTimeDataset):
    def __init__(self,
                *args,
                dataset_path: Optional[str] = None,
                macro_id: int = 2,
                time_offset: int = 9,
                max_z: int = 192,
                embed_parameters: bool = True,
                macro_parameters: Optional[np.ndarray] = None,
                **kwargs):
        super().__init__(*args, **kwargs)

        if dataset_path is None:
            dataset_path = f"/cluster/work/math/camlab-data/synthetic/ATM-MSC_3D_moist_macro{int(macro_id)}.nc"
        
        self.file_path = dataset_path
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        self.reader = nc.Dataset(self.file_path, "r")
        self.variable_aliases = [
            ("u",),
            ("v",),
            ("w",),
            ("temperature", "s"),
            ("qt",),
        ]
        self.variable_names = [self._resolve_variable_name(names) for names in self.variable_aliases]
        self.output_variable_names = ["u", "v", "w", "temperature", "qt"]
        self.base_channels = len(self.variable_names)
        
        self.time_offset = time_offset
        self.max_z = max_z
        self.embed_parameters = embed_parameters

        self.mean = torch.tensor([0.0, 0.0, 0.0, 290.0, 0.006], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor([1.0, 1.0, 1.0, 4.0, 0.004], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if self.masked_input is not None:
            self.mask = torch.tensor([1.0] * min(self.out_dim, self.base_channels) + [0.0] * max(self.in_dim - self.base_channels, 0), dtype=torch.float32)

        self.params_idx = np.array([0, 1, 5, 6, 7, 8])
        self.param_mean = np.array([0.0, 0.0, 291.0, 0.00099, 0.0, 0.25], dtype=np.float32)
        self.param_std = np.array([14.7, 7.74e-06, 2.0, 0.00022, 0.2, 0.75], dtype=np.float32)

        if macro_parameters is None:
            macro_parameters = np.array([7.35, 3.87e-06, 777.0, 290.7, 0.00833, 292.0, 0.00110, 0.0, 0.625], dtype=np.float32)
        self.params_macro = np.asarray(macro_parameters, dtype=np.float32)[self.params_idx]

        any_var = self.reader[self.variable_names[0]]
        self.native_spatial_shape = tuple(int(x) for x in any_var.shape[-3:])

        self.spatial_shape = (
            self.native_spatial_shape[0],
            self.native_spatial_shape[1],
            min(int(self.max_z), self.native_spatial_shape[2]),
        )

        self.total_time_steps = int(any_var.shape[1])
        self.N_max = int(any_var.shape[0])

        requested_samples = min(self.num_trajectories, self.N_max)
        self.num_trajectories = requested_samples
        self.N_val = 0
        self.N_test = 0
        self.post_init()

    def _resolve_variable_name(self, candidates):
        for name in candidates:
            if name in self.reader.variables:
                return name
        raise KeyError(f"None of the variable aliases {candidates} were found in {self.file_path}")
    
    def _read_field(self, var_name: str, sample_index: int, time_index: int) -> np.ndarray:

        print(var_name, sample_index, time_index)
        x = np.asarray(self.reader[var_name][sample_index, time_index], dtype=np.float32)
        if x.ndim != 3:
            x = np.squeeze(x)
        target_z = self.spatial_shape[2]
        if x.shape[-1] > target_z:
            x = x[..., :target_z]
        return x

    def _embed_parameters_as_channels(self, x: torch.Tensor) -> torch.Tensor:
        if not self.embed_parameters:
            return x

        num_param_channels = max(self.in_dim - self.base_channels, 0)
        if num_param_channels == 0:
            return x

        #params = (self.params[sample_index] - self.param_mean) / self.param_std
        params = (self.params_macro - self.param_mean) / self.param_std
        d, h, w = x.shape[-3:]
        param_tensor = torch.zeros((num_param_channels, d, h, w), dtype=x.dtype)

        fill_channels = min(num_param_channels, len(params))
        for i in range(fill_channels):
            param_tensor[i].fill_(float(params[i]))

        return torch.cat((x, param_tensor), dim=0)

    def _match_output_channels(self, y: torch.Tensor) -> torch.Tensor:
        if self.out_dim <= self.base_channels:
            return y[:self.out_dim]

        d, h, w = y.shape[-3:]
        pad_channels = self.out_dim - self.base_channels
        y_pad = torch.zeros((pad_channels, d, h, w), dtype=y.dtype)
        return torch.cat((y, y_pad), dim=0)

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        t1, t2 = self.time_indices[_idx]
        t1 += self.time_offset
        t2 += self.time_offset

        if t2 >= self.total_time_steps:
            raise IndexError(
                f"Requested time index {t2} exceeds available ConditionalATM_MSC_3D_moist range [0, {self.total_time_steps - 1}]")

        sample_index = i + self.start
        x_np = np.stack([self._read_field(v, sample_index, t1) for v in self.variable_names], axis=0)
        y_np = np.stack([self._read_field(v, sample_index, t2) for v in self.variable_names], axis=0)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        #x = self._embed_parameters_as_channels(x, sample_index)
        x = self._embed_parameters_as_channels(x)
        y = self._match_output_channels(y)


        lead_time = float(t2 - t1) / 24.0
        if self.masked_input is not None:
            return lead_time, x, y, self.mask
        return lead_time, x, y

class KelvinHelmholtz3dTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 5000
        self.N_val = 40
        self.N_test = 240

        self.file_path = "/cluster/work/math/camlab-data/synthetic/CEU_3D_RiemannKelvinHelmholtz.nc"
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        if self.masked_input is None:
            self.mean = torch.tensor([2.494, 0.0, 0.0, 0.0, 1.735], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.534, 0.255, 0.255,  0.255, 0.438], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([2.494, 0.0, 0.0, 0.0, 1.735] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.534, 0.255, 0.255,  0.255, 0.438] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        '''
        WRONG THING I USED FOR RIEMANN....

            if self.masked_input is None:
            self.mean = torch.tensor([1.262, 0., 0., 0., 1.252], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.259, 0.255, 0.255, 0.255, 0.275], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([1.262, 0., 0., 0., 1.252] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.259, 0.255, 0.255, 0.255, 0.275] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        '''

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()

class RiemannEllipse3dTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 5000
        self.N_val = 40
        self.N_test = 240

        self.file_path = "/cluster/work/math/camlab-data/synthetic/CEU_3D_RiemannEllipse.nc"
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        if self.masked_input is None:
            self.mean = torch.tensor([2.500, 0., 0., 0., 1.532], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.377, 0.252, 0.307, 0.293, 0.340], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([2.500, 0., 0., 0., 1.532] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.377, 0.252, 0.307, 0.293, 0.340] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()


class Riemann3dTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 5000
        self.N_val = 40
        self.N_test = 240

        self.file_path = "/cluster/work/math/camlab-data/synthetic/CEU_3D_Riemann.nc"
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        if self.masked_input is None:
            self.mean = torch.tensor([1.262, 0., 0., 0., 1.252], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.259, 0.255, 0.255, 0.255, 0.275], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([1.262, 0., 0., 0., 1.252] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.259, 0.255, 0.255, 0.255, 0.275] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()

class RiemannCurved3dTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 5000
        self.N_val = 40
        self.N_test = 240

        self.file_path = "/cluster/work/math/camlab-data/synthetic/CEU_3D_CurvedRiemann.nc"
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        if self.masked_input is None:
            self.mean = torch.tensor([1.246, 0., 0., 0., 1.272], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.213, 0.240, 0.240, 0.240, 0.227], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([1.246, 0., 0., 0., 1.272] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.213, 0.240, 0.240, 0.240, 0.227] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()

class ShearLayer3dTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 10000
        self.N_val = 40
        self.N_test = 240
        self.is_compressible = False

        self.file_path = "/cluster/work/math/camlab-data/synthetic/IEU_3D_CylindricalShearFlowLowRes.nc"
        #if self.copy_to_local_scratch:
        #   self._copy_to_tmpdir()

        self.mean = torch.tensor([0., 0., 0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.661, 0.204, 0.204], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        if self.in_dim>3:
            self.mask = torch.tensor([0.,1.,1.,1.,0.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.8, 0., 0., 0., 0.] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [1.0, 0.240, 0.240, 0.240, 1.0] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()

class ShearLayer3dMicroMacroTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 1000
        self.N_val = 0
        self.N_test = 1000
        self.is_compressible = False

        self.file_path = "/cluster/work/math/camlab-data/synthetic/IEU_3D_MacroMicroCylindricalShearFlow.nc"

        self.mean = torch.tensor([0., 0., 0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor( [0.661, 0.204, 0.204], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        if self.in_dim>3:
            self.mask = torch.tensor([0.,1.,1.,1.,0.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.8, 0., 0., 0., 0.] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [1.0, 0.240, 0.240, 0.240, 1.0] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()

class CloudShock3dTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 9977
        self.N_val = 40
        self.N_test = 240

        self.file_path = "/cluster/work/math/camlab-data/synthetic/CEU_3D_Cloudshock.nc"
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        if self.masked_input is None:
            self.mean = torch.tensor([2.535, 5.356, 0.0, 0.0, 77.09], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [2.002, 5.654, 0.441, 0.305, 81.72], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([1.,1.,1.,1.,1.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([2.535, 5.356, 0.0, 0.0, 77.09] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [2.002, 5.654, 0.441, 0.305, 81.72] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()
    
class TaylorGreen3dTimeDataset(CompressibleEulerNavierStokes3dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 8776
        self.N_val = 40
        self.N_test = 240
        self.is_compressible = False

        self.file_path = "/cluster/work/math/camlab-data/synthetic/IEU_3D_TaylorGreenLowRes.nc"
        if self.copy_to_local_scratch:
           self._copy_to_tmpdir()

        if self.in_dim == 3:
            self.mean = torch.tensor([0., 0., 0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.292, 0.292, 0.193], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([0.,1.,1.,1.,0.] + (self.in_dim - 5)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.8, 0., 0., 0., 0.] + (self.in_dim - 5)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [1.0,0.292, 0.292, 0.193, 1.0] + (self.in_dim - 5)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()


class _WindowedAll2AllN32T50Base(CompressibleEulerNavierStokes3dTimeDataset):
    """
    Shared base for the 32^3, 51-snapshot, incompressible (u, v, w) windowed
    all-to-all loaders. Subclasses set FILE_PATH, REAL_DT_RATIO and provide
    `_set_stats()` (means/stds + optional mask).

    Two enumeration modes (selected by `random_train_sampling`):

    * random_train_sampling=True (default — matches the original loader):
        - spatial resolution 32, trajectory has 51 snapshots;
        - windowed all-to-all sampling: jumps t1 -> t2 with 1 <= t2 - t1 <= time_window;
        - train split samples (t1, dt) randomly on every __getitem__ call;
        - val/test deterministically enumerate every valid (t1, t2) in the window;
        - epoch length is `num_trajectories * train_multiplier`.

    * random_train_sampling=False (coarse, deterministic mode):
        - all splits (train/val/test) deterministically enumerate (t1, t2) pairs
          built from `time_step_size`, `max_num_time_steps` and
          `allowed_transitions` (same recipe as BaseTimeDataset.post_init);
        - `time_window` and `train_multiplier` are ignored;
        - epoch length is `num_trajectories * len(time_indices)`.

    The per-sample `time` returned to the model is rescaled by 1/REAL_DT_RATIO so
    the time-embedding matches the real-time displacement the pretrained model
    saw (REAL_DT_RATIO = pretraining_dt / new_dt).
    """

    N_TIME_TOTAL = 51
    REAL_DT_RATIO: float = 10.0
    FILE_PATH: str = None
    _MEMBER_MAX = 10000
    _DEFAULT_N_VAL = 100
    _DEFAULT_N_TEST = 400

    def __init__(self,
                 *args,
                 time_window: int = 8,
                 train_multiplier: int = 50,
                 random_train_sampling: bool = True,
                 N_val: Optional[int] = None,
                 N_test: Optional[int] = None,
                 **kwargs):
        CompressibleEulerNavierStokes3dTimeDataset.__init__(self, *args, **kwargs)

        self.time_window = int(time_window)
        assert self.time_window >= 1
        self.train_multiplier = int(train_multiplier)
        assert self.train_multiplier >= 1
        self.random_train_sampling = bool(random_train_sampling)

        self.N_max = self._MEMBER_MAX
        self.N_val = int(N_val) if N_val is not None else self._DEFAULT_N_VAL
        self.N_test = int(N_test) if N_test is not None else self._DEFAULT_N_TEST
        self.is_compressible = False

        assert self.FILE_PATH is not None, "Subclasses must set FILE_PATH."
        self.file_path = self.FILE_PATH
        if self.copy_to_local_scratch:
            self._copy_to_tmpdir()

        self._set_stats()

        self.reader = nc.Dataset(self.file_path, "r")
        self.post_init()

    def _set_stats(self) -> None:
        raise NotImplementedError

    def post_init(self) -> None:
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.which == "train":
            assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val >= 0 and self.N_test >= 0

        T = self.N_TIME_TOTAL

        if self.random_train_sampling:
            W = self.time_window
            # Deterministic enumeration of every valid pair (t1, t2) with
            # 1 <= t2 - t1 <= W and 0 <= t1, t2 <= T - 1.  Used as-is for val/test.
            self.time_indices = []
            for t1 in range(T):
                for dt in range(1, W + 1):
                    t2 = t1 + dt
                    if t2 <= T - 1:
                        self.time_indices.append((t1, t2))

            if self.which == "train":
                self.multiplier = self.train_multiplier
            else:
                self.multiplier = len(self.time_indices)
        else:
            # Coarse, deterministic enumeration controlled by
            # (time_step_size, max_num_time_steps, allowed_transitions). Same
            # recipe as BaseTimeDataset.post_init, used for ALL splits so train
            # and val/test see the same (t1, t2) pair distribution.
            assert self.allowed_transitions is not None, (
                "random_train_sampling=False requires `allowed_transitions` to "
                "be set."
            )
            self.time_indices = []
            for i in range(self.max_num_time_steps + 1):
                for j in range(i, self.max_num_time_steps + 1):
                    if (j - i) in self.allowed_transitions:
                        t1 = self.time_step_size * i
                        t2 = self.time_step_size * j
                        assert t2 <= T - 1, (
                            f"time_step_size={self.time_step_size} * "
                            f"max_num_time_steps={self.max_num_time_steps} "
                            f"would exceed N_TIME_TOTAL-1={T - 1}."
                        )
                        self.time_indices.append((t1, t2))
            assert len(self.time_indices) > 0, (
                "No valid (t1, t2) pairs generated; check time_step_size, "
                "max_num_time_steps, allowed_transitions."
            )
            self.multiplier = len(self.time_indices)
            print("time_indices (coarse)", self.time_indices)

        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.which == "train" and self.random_train_sampling:
            dt = random.randint(1, self.time_window)
            t1 = random.randint(0, self.N_TIME_TOTAL - 1 - dt)
            t2 = t1 + dt
        else:
            t1, t2 = self.time_indices[_idx]

        t = t2 - t1
        # Time normalization: rescale parent's `t / 20.0` by 1/REAL_DT_RATIO so
        # the time embedding matches the real-time displacement seen during
        # pretraining (one new index = 0.04 real units = 1/10 of one old step).
        time = float(t) / (20.0 * self.REAL_DT_RATIO)
        if time <= 0:
            time = 1e-6

        t1_indices = (i + self.start, t1)
        t2_indices = (i + self.start, t2)

        u = torch.tensor(self.reader['u'][t1_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
        v = torch.tensor(self.reader['v'][t1_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
        w = torch.tensor(self.reader['w'][t1_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
        inputs = torch.cat((u, v, w), axis=0)

        u = torch.tensor(self.reader['u'][t2_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
        v = torch.tensor(self.reader['v'][t2_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
        w = torch.tensor(self.reader['w'][t2_indices]).type(torch.float32).reshape(1, self.resolution, self.resolution, self.resolution)
        label = torch.cat((u, v, w), axis=0)

        # Pad to in_dim/out_dim with the same channel layout used by the
        # pretrained 3D mix: rho-slot = 1, (u, v, w), P-slot = 0, then zero
        # slots up to in_dim.
        if self.in_dim >= 5:
            ones_ = torch.ones((1, self.resolution, self.resolution, self.resolution), dtype=torch.float32)
            p_slot = torch.zeros((1, self.resolution, self.resolution, self.resolution), dtype=torch.float32)
            if self.perturb_b:
                p_slot = p_slot + 0.2
            inputs = torch.cat((ones_, inputs, p_slot), 0)
            label = torch.cat((ones_, label, p_slot), 0)

        if self.in_dim > 5:
            for c in range(5, self.in_dim):
                z = torch.zeros((1, self.resolution, self.resolution, self.resolution), dtype=torch.float32)
                inputs = torch.cat((inputs, z), 0)
                if c < self.out_dim:
                    label = torch.cat((label, z), 0)

        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        return time, inputs, label


class TaylorGreenN32T50TimeDataset(_WindowedAll2AllN32T50Base):
    """
    Dense, low-res TG3D: IEU_3D_TaylorGreenN32T50.nc.
    Time range [0, 2] with 51 snapshots (dt = 0.04). Pretraining cadence on
    the LowRes TG3D was dt = 0.4, hence REAL_DT_RATIO = 10.
    """

    FILE_PATH = "/cluster/work/math/camlab-data/synthetic/IEU_3D_TaylorGreenN32T50.nc"
    REAL_DT_RATIO = 10.0

    def _set_stats(self) -> None:
        if self.in_dim == 3:
            self.mean = torch.tensor([0., 0., 0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.292, 0.292, 0.193], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([0., 1., 1., 1., 0.] + (self.in_dim - 5) * [0.], dtype=torch.float32)
            self.mean = torch.tensor([0.8, 0., 0., 0., 0.] + (self.in_dim - 5) * [0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([1.0, 0.292, 0.292, 0.193, 1.0] + (self.in_dim - 5) * [1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)


class ShearLayer3dN32T50TimeDataset(_WindowedAll2AllN32T50Base):
    """
    Dense, low-res Cylindrical Shear Flow 3D: IEU_3D_CylindricalShearFlowN32T50.nc.
    Time range [0, 1] with 51 snapshots (dt = 0.02). Pretraining cadence on
    the LowRes ShearLayer3d was dt = 0.25, hence REAL_DT_RATIO = 12.5.
    Mean/std mirror the existing ShearLayer3dTimeDataset.
    """

    FILE_PATH = "/cluster/work/math/camlab-data/synthetic/IEU_3D_CylindricalShearFlowN32T50.nc"
    REAL_DT_RATIO = 12.5

    def _set_stats(self) -> None:
        if self.in_dim == 3:
            self.mean = torch.tensor([0., 0., 0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.661, 0.204, 0.204], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([0., 1., 1., 1., 0.] + (self.in_dim - 5) * [0.], dtype=torch.float32)
            self.mean = torch.tensor([0.8, 0., 0., 0., 0.] + (self.in_dim - 5) * [0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([1.0, 0.240, 0.240, 0.240, 1.0] + (self.in_dim - 5) * [1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)