# Copyright 2024 The CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains all 3D incompressible fluid flow datasets used for training and evaluating diffusion models.

### Overview:
- **Training Datasets:** Standard datasets used to train diffusion models on 3D incompressible flows.  
- **Conditional Datasets:** Datasets prefixed with *Conditional* represent perturbed ensembles, 
        designed for evaluating model generalization and robustness.

### Training Strategy:
All 3D datasets in this module utilize a *lead time* parameter, which conditions the diffusion model on the number 
of timesteps moving forward from the initial condition. An **All-to-All (A2A) training strategy** is applied, 
allowing the model to learn dynamics across all possible temporal pairs.

> ⚠️ *Explicit usage of `lead_time` normalization for specific datasets is defined in* `utils/gencfd_utils.py`.

### Available Datasets:

- **3D Shear Layer:**  
  - `ShearLayer3D`  
  - `ConditionalShearLayer3D`  

- **3D Taylor Green Vortex:**  
  - `TaylorGreen3D`  
  - `ConditionalTaylorGreen3D`  

- **3D Nozzle Flow:**  
  - `Nozzle3D`  
  - `ConditionalNozzle3D` 
"""

import numpy as np
import torch
import netCDF4 as nc

from GenCFD.dataloader.dataset import TrainingSetBase, ConditionalBase
from typing import Union, Tuple, Any, List, Dict

array = np.ndarray


class LeadTimeNormalizer:
    """
    Handles dataset-specific lead time normalization for 3D diffusion models.

    The model is conditioned on the lead time using an All2All training strategy.
    Different datasets require different normalization strategies to optimize performance.

    Attributes:
        dataset_name (str): The name of the dataset to determine the normalization strategy.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def normalize_lead_time(self, t_init: int, t_final: int):
        """
        Applies dataset-specific normalization to the lead time.

        Args:
            t_init (int): The initial timestep.
            t_final (int): The final timestep.

        Returns:
            float: The normalized lead time.
        """
        #print(self.dataset_name)

        if self.dataset_name in ["Nozzle3D", "ConditionalNozzle3D"]:
            return self.nozzle3d_normalization(t_init, t_final)

        elif self.dataset_name in ["ShearLayer3D", "ConditionalShearLayer3D", "ConditionalShearLayerSpectral3D"]:
            return self.shearlayer3d_normalization(t_init, t_final)

        elif self.dataset_name in ["TaylorGreen3D, ConditionalTaylorGreen3D"]:
            return self.taylorgreen_normalization(t_init, t_final)

        elif self.dataset_name in ["ATM_MSC_3D_moist", "ConditionalATM_MSC_3D_moist", "ATM_MSC_3D_dry", "ConditionalATM_MSC_3D_dry"]:
            return self.atm_normalization(t_init, t_final)
        else:
            return self.default_normalization(t_init, t_final)
        
    def atm_normalization(self, t_init: int, t_final: int):
        lead_time = float(t_final - t_init)
        return lead_time/24.
    
    def nozzle3d_normalization(self, t_init: int, t_final: int):
        """Lead time normalized to be in between 0.1 and 1.4"""
        lead_time = t_final - t_init
        return 0.1 * lead_time

    def shearlayer3d_normalization(self, t_init: int, t_final: int):
        """Normalize Values to be in between 0.25 and 1"""
        lead_time = float(t_final - t_init)
        return 0.25 * lead_time

    def taylorgreen_normalization(self, t_init: int, t_final: int):
        """Normalize lead time to be in between 0.6875 and 2"""
        lead_time = float(t_final - t_init)
        return 0.25 + 0.4375 * (lead_time - 1)

    def default_normalization(self, t_init: int, t_final: int):
        """Default is no normalization but rather computing the lead time"""
        lead_time = t_final - t_init
        return lead_time


class IncompressibleFlows3D(TrainingSetBase):
    """
    Dataset class for 3D incompressible fluid flow simulations used in training diffusion models.

    This class handles the loading, normalization, and preprocessing of 3D incompressible flow datasets
    for model training. It supports flexible data handling, including moving files to local scratch storage,
    custom normalization, and generating all possible (t_initial, t_final) time pairs for All-to-All (A2A)
    training strategies.

    Attributes:
        min_time (int): The starting timestep for data sampling.
        max_time (int): The final timestep for data sampling.
        time_pairs (List[Tuple[int, int]]): Precomputed (t_initial, t_final) pairs for lead time conditioning.
        total_pairs (int): Total number of (t_initial, t_final) time pairs.
        lead_time_normalizer (LeadTimeNormalizer): Normalizer for lead time values.

    Args:
        file_system (dict): File paths and metadata for dataset storage and retrieval.
        input_channel (int): Number of input channels (variables) for the model.
        output_channel (int): Number of output channels (variables) for the model.
        spatial_resolution (Tuple): Spatial resolution of the dataset (e.g., grid size).
        input_shape (Tuple): Shape of the input data tensors.
        output_shape (Tuple): Shape of the output data tensors.
        variable_names (List[str]): Names of physical variables in the dataset (e.g., velocity, pressure).
        min_time (int): Minimum timestep to include in the dataset.
        max_time (int): Maximum timestep to include in the dataset.
        ndim (int, optional): Number of spatial dimensions (default is 3).
        start (int, optional): Offset to start reading data samples (default is 0).
        training_samples (int, optional): Number of training samples to use. If None, uses all available samples.
        move_to_local_scratch (bool, optional): If True, moves dataset to local scratch storage for faster access.
        retrieve_stats_from_file (bool, optional): If True, loads normalization statistics from a file.
        mean_training_input (array, optional): Mean values for input normalization.
        std_training_input (array, optional): Standard deviation values for input normalization.
        mean_training_output (array, optional): Mean values for output normalization.
        std_training_output (array, optional): Standard deviation values for output normalization.
        get_values (bool, optional): If True, retrieves raw mean and std values from the dataset instead of computing them.

    Raises:
        ValueError: If `min_time` is greater than `max_time`.
        FileNotFoundError: If dataset files specified in `file_system` are not found.

    Example:
        ```python
        dataset = IncompressibleFlows3D(
            file_system = {
                "dataset_name": "TaylorGreen3D",
                'file_name': 'taylor_green.nc',
                'origin': '/cluster/data/taylor_green/'
                # Additional not relevant file_system settings in case the mean and std were accumulated
                'stats_file': 'GroundTruthStats_ConditionalTaylorGreen',
                'origin_stats': '/cluster/data/taylor_green/'
                # If the the single mean and std values in vector form are available
                'mean_file': 'mean.npy',
                'std_file': 'std.npy',
            }
        )
        sample = dataset.__getitem__(0)
        print(sample)
        ```
    """

    def __init__(
        self,
        file_system: dict,
        input_channel: int,
        output_channel: int,
        spatial_resolution: Tuple,
        input_shape: Tuple,
        output_shape: Tuple,
        variable_names: List[str],
        min_time: int,
        max_time: int,
        ndim: int = 3,
        start: int = 0,
        training_samples: int = None,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        mean_training_input: array = None,
        std_training_input: array = None,
        mean_training_output: array = None,
        std_training_output: array = None,
        get_values: bool = False,
    ) -> None:

        super().__init__(
            file_system=file_system,
            ndim=ndim,  # Always 3D dataset
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            training_samples=training_samples,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
            get_values=get_values,
        )

        self.min_time = min_time
        self.max_time = max_time

        # Precompute all possible (t_initial, t_final) pairs within the specified range.
        self.time_pairs = [
            (i, j)
            for i in range(self.min_time, self.max_time)
            for j in range(i + 1, self.max_time + 1)
        ]
        self.total_pairs = len(self.time_pairs)

        # get the correct normalization method for the lead time
        self.lead_time_normalizer = LeadTimeNormalizer(
            dataset_name=self.file_system["dataset_name"]
        )

        if "resolver_path" in file_system:
            self.spectral_resolver_path = file_system["resolver_path"]
        else:
            self.spectral_resolver_path = None
        
        if self.spectral_resolver_path is not None:
            self.spectral_resolver_file = nc.Dataset(self.spectral_resolver_path, mode="r")
            _time_i = self.spectral_resolver_file.variables["time_i"][:]
            if len(_time_i) != len(self.time_pairs):
                raise ValueError(f"You msut select min_time and max_time that matches the resolver's file")
        else:
            self.spectral_resolver_file = None
            
    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Uses the correct normalization scheme for both the Conditional and Training Dataset"""
        return self.lead_time_normalizer.normalize_lead_time(t_init, t_final)

    def __len__(self):
        # Return the total number of data points times the number of pairs.
        #print(self.training_samples * self.total_pairs, "HERE")
        return self.training_samples * self.total_pairs

    def __getitem__(self, index):
        # Determine the data point and the (t_initial, t_final) pair
        data_index = index // self.total_pairs
        pair_index = index % self.total_pairs
        t_init, t_final = self.time_pairs[pair_index]

        # List with all variables relevant for the given dataset will be stacked later
        ###print(index, data_index, pair_index, t_init, t_final)
        
        data_list_inp = [
            self.file.variables[var][data_index, t_init] for var in self.variable_names
        ]
        if self.spectral_resolver_file is not None:
            data_list_inp = data_list_inp + [self.spectral_resolver_file.variables[var][data_index, pair_index] for var in self.variable_names]

        data_list_out = [
            self.file.variables[var][data_index, t_final] for var in self.variable_names
        ]

        if self.file_system["dataset_name"] == "Nozzle3D":
            # Add an additional conditioning tensor is required for the Nozzle dataset
            vel_inject_inp = float(
                self.file.variables["injection_velocity"][data_index, t_init] * np.ones((192, 64, 64)))
            #vel_inject_out = float(
            #    self.file.variables["injection_velocity"][data_index, t_final] * np.ones((192, 64, 64)))

            data_list_inp = [np.broadcast_to(vel_inject_inp, (192, 64, 64))] + data_list_inp
            #data_list_out = [np.broadcast_to(vel_inject_inp, (192, 64, 64))] + data_list_out

        combined_data_inp = np.stack(data_list_inp, axis=-1)
        combined_data_out = np.stack(data_list_out, axis=-1)

        # Extract initial and final conditions
        initial_condition = self.normalize_input(combined_data_inp)

        if self.file_system["dataset_name"] == "Nozzle3D":
            combined_data_out = combined_data_out[..., 1:]  # get rid of the conditioning

        final_condition = self.normalize_output(combined_data_out)

        lead_time_normalized = self.normalize_lead_time(t_init=t_init, t_final=t_final)

        initial_cond = (
            torch.from_numpy(initial_condition).type(torch.float32).permute(3, 2, 1, 0)
        )

        target_cond = (
            torch.from_numpy(final_condition).type(torch.float32).permute(3, 2, 1, 0)
        )

        return {
            "lead_time": torch.tensor(lead_time_normalized, dtype=torch.float32),
            "initial_cond": initial_cond,
            "target_cond": target_cond,
        }


class ShearLayer3D(IncompressibleFlows3D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        start=0,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 3,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 64),
        input_shape: Tuple[int, ...] = (6, 64, 64, 64),
        output_shape: Tuple[int, ...] = (3, 64, 64, 64),
        variable_names: List[str] = ["u", "v", "w"],
        min_time: int = 0,
        max_time: int = 4,
        mean_training_input: array = np.array(
            [1.5445266e-08, 1.2003070e-08, -3.2182508e-09]
        ),
        mean_training_output: array = np.array(
            [-8.0223117e-09, -3.3674191e-08, 1.5241447e-08]
        ),
        std_training_input: array = np.array([0.20691067, 0.15985465, 0.15808222]),
        std_training_output: array = np.array([0.2706984, 0.24893111, 0.24169469]),
    ):

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            min_time=min_time,
            max_time=max_time,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
            training_samples = 144, #72, #36, #140, #282
        )

    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Normalize Values to be in between 0 and 1"""
        lead_time = float(t_final - t_init)
        return 0.25 * lead_time

class ShearLayerSpectral3D(IncompressibleFlows3D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        start=0,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 6,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 64),
        input_shape: Tuple[int, ...] = (9, 64, 64, 64),
        output_shape: Tuple[int, ...] = (3, 64, 64, 64),
        variable_names: List[str] = ["u", "v", "w"],
        min_time: int = 0,
        max_time: int = 4,
        mean_training_input: array = np.array(
            [1.5445266e-08, 1.2003070e-08, -3.2182508e-09, 0.0, 0.0, 0.0]
        ),
        mean_training_output: array = np.array(
            [-8.0223117e-09, -3.3674191e-08, 1.5241447e-08]
        ),
        std_training_input: array = np.array([0.20691067, 0.15985465, 0.15808222, 1.0, 1.0, 1.0]),
        std_training_output: array = np.array([0.2706984, 0.24893111, 0.24169469]),
    ):

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            min_time=min_time,
            max_time=max_time,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
            training_samples = 144, #72, #36, #140, #282
        )

    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Normalize Values to be in between 0 and 1"""
        lead_time = float(t_final - t_init)
        return 0.25 * lead_time

class TaylorGreen3D(IncompressibleFlows3D):

    def __init__(
        self,
        metadata: Dict[str, Any],
        start: int = 0,
        min_time: int = 0,
        max_time: int = 5,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 3,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 64),
        input_shape: Tuple[int, ...] = (6, 64, 64, 64),
        output_shape: Tuple[int, ...] = (3, 64, 64, 64),
        variable_names: List[str] = ["u", "v", "w"],
        #mean_training_input: array = np.array([0.0, 0.0, 0.0]),
        #std_training_input: array = np.array([0.292, 0.292, 0.193]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0]),
        #std_training_output: array = np.array([0.292, 0.292, 0.193]),
    ):

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            min_time=min_time,
            max_time=max_time,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            #mean_training_input=mean_training_input,
            #std_training_input=std_training_input,
            #mean_training_output=mean_training_output,
            #std_training_output=std_training_output,
        )


class Nozzle3D(IncompressibleFlows3D):

    def __init__(
        self,
        metadata: Dict[str, Any],
        start: int = 0,
        file: str = None,
        min_time: int = 0,
        max_time: int = 14,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        input_channel: int = 4,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 192),
        input_shape: Tuple[int, ...] = (4, 64, 64, 192),
        output_shape: Tuple[int, ...] = (3, 64, 64, 192),
        variable_names: List[str] = ["u", "v", "w"],
        mean_training_input: array = np.array([0.0, 0.00858, 0.0, 0.0]),
        std_training_input: array = np.array([1.0, 0.0727, 0.0266, 0.0252]),
        mean_training_output: array = np.array([0.00858, 0.0, 0.0]),
        std_training_output: array = np.array([0.0727, 0.0266, 0.0252]),
    ):

        super().__init__(
            file_system=metadata,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            min_time=min_time,
            max_time=max_time,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
        )

        # Overwrite pairs since here every second step should be taken
        self.time_pairs = [
            (2 * i, 2 * j)
            for i in range(0, self.max_time // 2 - 1)
            for j in range(i + 1, self.max_time // 2)
        ]
        self.total_pairs = len(self.time_pairs)


class ATM_MSC_3D_moist(TrainingSetBase):
    def __init__(
        self,
        file_system: dict,
        spatial_resolution: Tuple[int, ...] = (96, 96, 192),
        variable_names: List[str] = ["u", "v", "w", "temperature","qt"],# "ql"], ###
        input_shape: Tuple[int, ...] =  (5, 96, 96, 192),
        output_shape: Tuple[int, ...] = (5,  96, 96, 192),
        min_time: int = 9,
        max_time: int = 24,
        ndim: int = 3,
        start: int = 0,
        training_samples: int = 268,
        move_to_local_scratch: bool = True,
        retrieve_stats_from_file: bool = False,
        #mean_training_input: array = np.array([-0.23376483, 0.08835366, 0.0, 289.84653, 0.0058482527, 4.5011555e-05, 0.0,  0.0, 291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array = np.array([0.5305401, 0.42337054, 0.3987728, 3.9537246, 0.0037528432, 0.00012942153, 14.7, 7.74e-06, 2.0, 0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([-0.23376483, 0.08835366, 0.0, 289.84653, 0.0058482527, 4.5011555e-05]),
        #std_training_output: array = np.array([0.5305401, 0.42337054, 0.3987728, 3.9537246, 0.0037528432, 0.00012942153]),
        
        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0,   0.0,  0.0,     291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001,   14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001]),
        
        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0,   0.0,  0.0,      840.0,  290.2, 0.000, 291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001,   14.7, 7.74e-06, 50.0,   1.0,   0.009,  2.0,  0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001]),

        mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0,  0.006]),
        std_training_input: array  = np.array([1.0, 1.0, 1.0,  4.0,   0.004]),
        mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0, 0.006]),
        std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0,   0.004]),

        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.0,  0.0,     291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,    14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0]),
        get_values: bool = False,
        input_channel: int = 5,
        output_channel: int = 5,
    ) -> None:

        super().__init__(
            file_system=file_system,
            ndim=ndim,  # Always 3D dataset
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            training_samples=training_samples,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
            get_values=get_values,
        )

        self.min_time = min_time
        self.max_time = max_time

        # HARD CODED
        self._names = ["ug", "divergence", "zi", "tg", "qtg", "sst", "cm", "cs", "prt"]
        self.params = np.load("/cluster/work/math/camlab-data/synthetic/_zsetup_ATM-MSC_3D_moist/ATM-MSC_3D_moist_input_parameters.npy")

        #self.params_idx = np.array([]) # Parameters that we need during the training (non IC params)
        self.params_idx = np.array([0, 1, 5, 6, 7, 8])
        self.params =  self.params[:, self.params_idx]
        self.param_mean = np.array([0.0,  0.0,     291.0, 0.00099, 0.0, 0.25])
        self.param_std  = np.array([14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75])
        
        self.num_parameters = len(self.params_idx)
        self.param_embed_channels = 1 if self.num_parameters > 0 else 0
        
        #self.params_mins = np.array([0.0,   0.0,     756.0, 289.6, 0.0081, 291.0, 0.00099, 0.0, 0.25])[self.params_idx]
        #self.params_maxs = np.array([14.7, 7.74e-06, 924.0, 290.8, 0.0099, 293.0, 0.00121, 0.2, 1.0])[self.params_idx]
        #                             14.7, 7.74e-06,                        2.0,  0.00022, 0.2, 0.75
        # Precompute all possible (t_initial, t_final) pairs within the specified range.
        self.time_pairs = [
            (i, j)
            for i in range(self.min_time, self.max_time)
            for j in range(i + 1, self.max_time + 1)
        ]
        self.total_pairs = len(self.time_pairs)

        # get the correct normalization method for the lead time
        self.lead_time_normalizer = LeadTimeNormalizer(
            dataset_name=self.file_system["dataset_name"]
        )

        if "resolver_path" in file_system:
            self.spectral_resolver_path = file_system["resolver_path"]
        else:
            self.spectral_resolver_path = None
        
        if self.spectral_resolver_path is not None:
            self.spectral_resolver_file = nc.Dataset(self.spectral_resolver_path, mode="r")
            _time_i = self.spectral_resolver_file.variables["time_i"][:]
            if len(_time_i) != len(self.time_pairs):
                raise ValueError(f"You msut select min_time and max_time that matches the resolver's file")
        else:
            self.spectral_resolver_file = None
            
    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Uses the correct normalization scheme for both the Conditional and Training Dataset"""
        return self.lead_time_normalizer.normalize_lead_time(t_init, t_final)

    def __len__(self):
        # Return the total number of data points times the number of pairs.
        #print(self.training_samples * self.total_pairs, "HERE")
        return self.training_samples * self.total_pairs

    def __getitem__(self, index):
        # Determine the data point and the (t_initial, t_final) pair
        data_index = index // self.total_pairs
        pair_index = index % self.total_pairs
        t_init, t_final = self.time_pairs[pair_index]

        # List with all variables relevant for the given dataset will be stacked later
        ###print(index, data_index, pair_index, t_init, t_final)
        
        data_list_inp = [
            self.file.variables[var][data_index, t_init, ... ,:self.input_shape[-1]] for var in self.variable_names
        ]
        if self.spectral_resolver_file is not None:
            data_list_inp = data_list_inp + [self.spectral_resolver_file.variables[var][data_index, pair_index] for var in self.variable_names]

        data_list_out = [
            self.file.variables[var][data_index, t_final, ... ,:self.output_shape[-1]] for var in self.variable_names
        ]

        #for i in range(len(self.params_idx)):
        #    param = float(self.params[data_index, i])
        #    #print(param, data_index, i, self.params[data_index])
        #    data_list_inp = data_list_inp + [np.broadcast_to(param, (self.input_shape[1], self.input_shape[2], self.input_shape[3]))]

        parameters = None
        if len(self.params_idx) > 0:
            parameters = self.params[data_index].astype(np.float32)
            parameters = (parameters - self.param_mean)/self.param_std
        combined_data_inp = np.stack(data_list_inp, axis=-1)
        combined_data_out = np.stack(data_list_out, axis=-1)

        # Extract initial and final conditions
        initial_condition = self.normalize_input(combined_data_inp)
        final_condition = self.normalize_output(combined_data_out)

        lead_time_normalized = self.normalize_lead_time(t_init=t_init, t_final=t_final)

        initial_cond = (
            torch.from_numpy(initial_condition).type(torch.float32).permute(3, 0, 1, 2)
        )

        target_cond = (
            torch.from_numpy(final_condition).type(torch.float32).permute(3, 0, 1, 2)
        )

        return {
            "lead_time": torch.tensor(lead_time_normalized, dtype=torch.float32),
            "initial_cond": initial_cond,
            "target_cond": target_cond,
            "parameters": None if parameters is None else torch.from_numpy(parameters),
        }


class ATM_MSC_3D_dry(ATM_MSC_3D_moist):
    def __init__(
        self,
        file_system: dict,
        spatial_resolution: Tuple[int, ...] = (128, 128, 128),
        variable_names: List[str] = ["u", "v", "w", "temperature"], ###
        input_shape: Tuple[int, ...] =  (4, 128, 128, 128),
        output_shape: Tuple[int, ...] = (4,  128, 128, 128),
        min_time: int = 9,
        max_time: int = 24,
        ndim: int = 3,
        start: int = 0,
        training_samples: int = 9744,
        move_to_local_scratch: bool = False,
        retrieve_stats_from_file: bool = False,
        #mean_training_input: array = np.array([-0.23376483, 0.08835366, 0.0, 289.84653, 0.0058482527, 4.5011555e-05, 0.0,  0.0, 291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array = np.array([0.5305401, 0.42337054, 0.3987728, 3.9537246, 0.0037528432, 0.00012942153, 14.7, 7.74e-06, 2.0, 0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([-0.23376483, 0.08835366, 0.0, 289.84653, 0.0058482527, 4.5011555e-05]),
        #std_training_output: array = np.array([0.5305401, 0.42337054, 0.3987728, 3.9537246, 0.0037528432, 0.00012942153]),
        
        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0,   0.0,  0.0,     291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001,   14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001]),
        
        mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0]),
        std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,]),
        mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0 ]),
        std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0]),

        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.0,  0.0,     291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,    14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0]),
        get_values: bool = False,
        input_channel: int = 4,
        output_channel: int = 4,
    ) -> None:

        super().__init__(
            file_system=file_system,
            ndim=ndim,  # Always 3D dataset
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            training_samples=training_samples,
            move_to_local_scratch=move_to_local_scratch,
            retrieve_stats_from_file=retrieve_stats_from_file,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
            get_values=get_values,
        )

        self.min_time = min_time
        self.max_time = max_time

        _file_path = "/cluster/work/math/camlab-data/synthetic/ATM-CBL_3D_dry.nc"
        _ds = nc.Dataset(_file_path, mode="r")
        _X = np.array(_ds.variables["Ug"]).reshape(-1,1)
        _Y = np.array(_ds.variables["Qstar"]).reshape(-1,1)
        self.params = np.concatenate((_X,_Y), axis = 1)
        _ds.close()

        self.params_idx = np.array([0, 1])
        self.params =  self.params[:, self.params_idx]
        self.param_mean = np.array([2.5, 0.2])
        self.param_std  = np.array([1.5, 0.06])
        self.num_parameters = len(self.params_idx)
        self.param_embed_channels = 1 if self.num_parameters > 0 else 0

        self.time_pairs = [
            (i, j)
            for i in range(self.min_time, self.max_time)
            for j in range(i + 1, self.max_time + 1)
        ]
        self.total_pairs = len(self.time_pairs)

        # get the correct normalization method for the lead time
        self.lead_time_normalizer = LeadTimeNormalizer(
            dataset_name=self.file_system["dataset_name"]
        )

        if "resolver_path" in file_system:
            self.spectral_resolver_path = file_system["resolver_path"]
        else:
            self.spectral_resolver_path = None

        if self.spectral_resolver_path is not None:
            self.spectral_resolver_file = nc.Dataset(self.spectral_resolver_path, mode="r")
            _time_i = self.spectral_resolver_file.variables["time_i"][:]
            if len(_time_i) != len(self.time_pairs):
                raise ValueError(f"You msut select min_time and max_time that matches the resolver's file")
        else:
            self.spectral_resolver_file = None

class ConditionalATM_MSC_3D_moist(ConditionalBase):
    def __init__(
        self,
        file_system: dict,
        input_shape: Tuple[int, ...] =  (5, 96, 96, 192),
        output_shape: Tuple[int, ...] = (5,  96, 96, 192),
        micro_perturbations: int = 1000,
        macro_perturbations: int = 1,
        variable_names: List[str] = ["u", "v", "w", "temperature", "qt"], #, "ql"],
        input_channel: int = 5,
        output_channel: int = 5,
        spatial_resolution: Tuple =  (96, 96, 192),
        t_start: int = 12,
        t_final: int = 20,
        start: int = 0,
        ndim: int = 3,
        training_samples: int = None,
        move_to_local_scratch: bool = False,
        
        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0,   0.0,  0.0,     291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001,   14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0 , 0.006,   0.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0,    0.004, 0.001]),
        
        mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0,  0.006]),
        std_training_input: array  = np.array([1.0, 1.0, 1.0,  4.0,   0.004]),
        mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0, 0.006]),
        std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0,   0.004]),

        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0]),
        
        #mean_training_input: array = np.array([0.0, 0.0, 0.0, 290.0,  0.0,  0.0,     291.0, 0.00099, 0.0, 0.25]),
        #std_training_input: array  = np.array([1.0, 1.0, 1.0, 4.0,    14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75]),
        #mean_training_output: array = np.array([0.0, 0.0, 0.0, 290.0]),
        #std_training_output: array =  np.array([1.0, 1.0, 1.0, 4.0,]),
    ) -> None:

        super().__init__(
            file_system=file_system,
            ndim=ndim,  # Always 3D dataset
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            training_samples=training_samples,
            move_to_local_scratch=move_to_local_scratch,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
        )

        self.t_start = t_start
        self.t_final = t_final

        self._names = ["ug", "divergence", "zi", "tg", "qtg", "sst", "cm", "cs", "prt"]

        self.params_idx = np.array([0, 1, 5, 6, 7, 8]) # Parameters that we need during the training (non IC params)
        self.num_parameters = len(self.params_idx)
        self.param_embed_channels = 1 if self.num_parameters > 0 else 0

        self.param_mean = np.array([0.0,  0.0,     291.0, 0.00099, 0.0, 0.25])
        self.param_std  = np.array([14.7, 7.74e-06, 2.0,  0.00022, 0.2, 0.75])

        # Macro 2:
        ###self.params_macro = np.array([7.35, 3.87e-06,  777.0, 290.7, 0.00833, 292.0, 0.00110, 0.0, 0.625])[self.params_idx]
        self.params_macro = np.array([7.35, 3.87e-06,  777.0, 290.7, 0.00833, 292.0, 0.00110, 0.0, 0.625])[self.params_idx]

        if "resolver_path" in file_system:
            self.spectral_resolver_path = file_system["resolver_path"]
        else:
            self.spectral_resolver_path = None

        # get the correct normalization method for the lead time
        self.lead_time_normalizer = LeadTimeNormalizer(
            dataset_name=self.file_system["dataset_name"]
        )

    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Uses the correct normalization scheme for both the Conditional and Training Dataset"""
        return self.lead_time_normalizer.normalize_lead_time(t_init, t_final)

    def __getitem__(self, index):
        
        #/cluster/work/math/braonic/TrainedModels/OOD_Generalization/eul_ns3d_mix1/TURBO_MASK_scratch_Base_10ep_8gpus_bs3_4acc_10000/predictions_ns_shear3d_mm_test/sample_997_mm_0_pred.npy
        macro_idx = self.get_macro_index(index + self.start)
        micro_idx = self.get_micro_index(index + self.start)

        micro_idx = micro_idx % 50

        # Preload selector since some datasets have only 1 macro perturbation included
        idx_selector = (
            (macro_idx, micro_idx) if self.macro_perturbations > 4 else (micro_idx,)
        )

        data_initial = [
            self.file.variables[var][*idx_selector, self.t_start, ... ,:self.output_shape[-1]]
            for var in self.variable_names
        ]

        data_target = [
            self.file.variables[var][*idx_selector, self.t_final, ... ,:self.output_shape[-1]]
            for var in self.variable_names
        ]

        parameters = None
        if len(self.params_idx) > 0:
            parameters = self.params_macro.astype(np.float32)
            parameters = (parameters - self.param_mean)/self.param_std
        
        data_input = np.stack(data_initial, axis=-1)
        data_input = self.normalize_input(data_input)

        data_output = np.stack(data_target, axis=-1)
        data_output = self.normalize_output(data_output)

        initial_cond = (
            torch.from_numpy(data_input).type(torch.float32).permute(3, 0, 1, 2)
        )

        target_cond = (
            torch.from_numpy(data_output).type(torch.float32).permute(3, 0, 1, 2)
        )

        lead_time_normalized = self.normalize_lead_time(
            t_init=self.t_start, t_final=self.t_final
        )

        # Store indices for the CDF computation
        return {
            "lead_time": torch.tensor(lead_time_normalized, dtype=torch.float32),
            "initial_cond": initial_cond,
            "target_cond": target_cond,
            "parameters": None if parameters is None else torch.from_numpy(parameters),
        }



#--------------------
# Conditional:
#--------------------
class ConditionalIncompressibleFlows3D(ConditionalBase):
    """
    Conditional dataset class for 3D incompressible fluid flow simulations with perturbations.

    This class extends the functionality of `IncompressibleFlows3D` by introducing micro and macro
    perturbations for evaluating the robustness of diffusion models. It is designed for testing how
    the model handles perturbed initial conditions over a specified lead time.

    For general dataset handling and attributes, refer to `IncompressibleFlows3D`.

    Attributes:
        t_start (int): The initial timestep for sampling.
        t_final (int): The final timestep for sampling.
        lead_time_normalizer (LeadTimeNormalizer): Normalizer for the lead time between `t_start` and `t_final`.

    Args:
        micro_perturbations (int): Number of micro-scale perturbations applied to the dataset for fine-grained variability.
        macro_perturbations (int): Number of macro-scale perturbations applied to the dataset for large-scale variability.
        t_start (int): The starting timestep for conditional sampling.
        t_final (int): The ending timestep for conditional sampling.
        training_samples (int, optional): Number of training samples to use. If None, uses all available samples.

    Example:
        See IncompressibleFlows3D
    """

    def __init__(
        self,
        file_system: dict,
        input_channel: int,
        output_channel: int,
        spatial_resolution: Tuple,
        input_shape: Tuple,
        output_shape: Tuple,
        variable_names: List[str],
        micro_perturbations: int,
        macro_perturbations: int,
        t_start: int,
        t_final: int,
        start: int = 0,
        ndim: int = 3,
        training_samples: int = None,
        move_to_local_scratch: bool = False,
        mean_training_input: array = None,
        std_training_input: array = None,
        mean_training_output: array = None,
        std_training_output: array = None,
        
    ) -> None:

        super().__init__(
            file_system=file_system,
            ndim=ndim,  # Always 3D dataset
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            training_samples=training_samples,
            move_to_local_scratch=move_to_local_scratch,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            mean_training_input=mean_training_input,
            std_training_input=std_training_input,
            mean_training_output=mean_training_output,
            std_training_output=std_training_output,
        )

        self.t_start = t_start
        self.t_final = t_final

        if "resolver_path" in file_system:
            self.spectral_resolver_path = file_system["resolver_path"]
        else:
            self.spectral_resolver_path = None
        
        #if self.spectral_resolver_path is not None:
        #    self.spectral_resolver_file = nc.Dataset(self.spectral_resolver_path, mode="r")

        # get the correct normalization method for the lead time
        self.lead_time_normalizer = LeadTimeNormalizer(
            dataset_name=self.file_system["dataset_name"]
        )

    def normalize_lead_time(self, t_init: int, t_final: int) -> Union[int, float]:
        """Uses the correct normalization scheme for both the Conditional and Training Dataset"""
        return self.lead_time_normalizer.normalize_lead_time(t_init, t_final)

    def __getitem__(self, index):
        
        #/cluster/work/math/braonic/TrainedModels/OOD_Generalization/eul_ns3d_mix1/TURBO_MASK_scratch_Base_10ep_8gpus_bs3_4acc_10000/predictions_ns_shear3d_mm_test/sample_997_mm_0_pred.npy
        macro_idx = self.get_macro_index(index + self.start)
        micro_idx = self.get_micro_index(index + self.start)

        # Preload selector since some datasets have only 1 macro perturbation included
        idx_selector = (
            (macro_idx, micro_idx) if self.macro_perturbations >= 0 else (micro_idx,)
        )
        '''
            vrati posle idx_selector na > 0 --> SAMO 1 MACRO ZA SADA 
        '''

        #print(*idx_selector, self.t_start, self.file.variables['u'].shape)
        # Stack along the new last dimension (axis=-1) and dynamically load the data
        data_initial = [
            self.file.variables[var][*idx_selector, self.t_start]
            for var in self.variable_names
        ]

        if data_initial[0].shape[0] == 2*self.input_shape[1]:
            for i in range(len(data_initial)):
                data_initial[i] = data_initial[i][::2,::2,::2]

        data_target = [
            self.file.variables[var][*idx_selector, self.t_final]
            for var in self.variable_names
        ]

        if data_target[0].shape[0] == 2*self.output_shape[1]:
            for i in range(len(data_target)):
                data_target[i] = data_target[i][::2,::2,::2]
        
        if self.spectral_resolver_path is not None:
            data_spectral_inp = np.load(self.spectral_resolver_path + f"/sample_{micro_idx}_mm_0_pred.npy")
            
            if self.output_shape[0] == 3:
                spectral_start = 1
                spectral_end = 4
            elif self.output_shape[0] == 5:
                spectral_start = 0
                spectral_end = 5
            for i in range(spectral_start, spectral_end):
                data_initial = data_initial + [data_spectral_inp[i]]
            #self.spectral_resolver_file.variables[var][data_index, pair_index] for var in self.variable_names]
        
        data_input = np.stack(data_initial, axis=-1)
        data_input = self.normalize_input(data_input)

        '''
            If you ever check nozzle, think what is the exact location of the channel
        '''
        if self.file_system["dataset_name"] == "ConditionalNozzle3D":
            # Additional Conditioning for the Nozzle Dataset
            vel_inject = float(
                self.file.variables["injection_velocity"][micro_idx]
            ) * np.ones((192, 64, 64))
            data_initial.insert(0, vel_inject)

        
        data_output = np.stack(data_target, axis=-1)
        data_output = self.normalize_output(data_output)

        initial_cond = (
            torch.from_numpy(data_input).type(torch.float32).permute(3, 2, 1, 0)
        )

        target_cond = (
            torch.from_numpy(data_output).type(torch.float32).permute(3, 2, 1, 0)
        )

        lead_time_normalized = self.normalize_lead_time(
            t_init=self.t_start, t_final=self.t_final
        )

        # Store indices for the CDF computation
        return {
            "lead_time": torch.tensor(lead_time_normalized, dtype=torch.float32),
            "initial_cond": initial_cond,
            "target_cond": target_cond,
        }


class ConditionalShearLayer3D(ConditionalIncompressibleFlows3D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        move_to_local_scratch: bool = False,
        input_channel: int = 3,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 64),
        input_shape: Tuple[int, ...] = (6, 64, 64, 64),
        output_shape: Tuple[int, ...] = (3, 64, 64, 64),
        variable_names: List[str] = ["u", "v", "w"],
        macro_perturbations: int = 10,
        micro_perturbations: int = 1000,
        t_start: int = 0,
        t_final: int = 4,
        start: int = 0,
    ):

        super().__init__(
            file_system=metadata,
            training_samples=micro_perturbations * macro_perturbations,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            t_start=t_start,
            t_final=t_final,
            mean_training_input = np.array(
            [1.5445266e-08, 1.2003070e-08, -3.2182508e-09]
        ),
        mean_training_output = np.array(
            [-8.0223117e-09, -3.3674191e-08, 1.5241447e-08]
        ),
        std_training_input = np.array([0.20691067, 0.15985465, 0.15808222]),
        std_training_output = np.array([0.2706984, 0.24893111, 0.24169469]),
        )

class ConditionalShearLayerSpectral3D(ConditionalIncompressibleFlows3D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        move_to_local_scratch: bool = False,
        input_channel: int = 6,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 64),
        input_shape: Tuple[int, ...] = (6, 64, 64, 64),
        output_shape: Tuple[int, ...] = (3, 64, 64, 64),
        variable_names: List[str] = ["u", "v", "w"],
        macro_perturbations: int = 1,
        micro_perturbations: int = 1000,
        t_start: int = 0,
        t_final: int = 4,
        start: int = 0,
    ):

        super().__init__(
            file_system=metadata,
            training_samples=micro_perturbations * macro_perturbations,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            start=start,
            move_to_local_scratch=move_to_local_scratch,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            t_start=t_start,
            t_final=t_final,
            mean_training_input = np.array(
            [1.5445266e-08, 1.2003070e-08, -3.2182508e-09, 0.0, 0.0, 0.0]
        ),
        mean_training_output = np.array(
            [-8.0223117e-09, -3.3674191e-08, 1.5241447e-08]
        ),
        std_training_input = np.array([0.20691067, 0.15985465, 0.15808222, 1.0, 1.0, 1.0]),
        std_training_output = np.array([0.2706984, 0.24893111, 0.24169469]),
        )


class ConditionalTaylorGreen3D(ConditionalIncompressibleFlows3D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        move_to_local_scratch: bool = False,
        input_channel: int = 3,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 64),
        input_shape: Tuple[int, ...] = (6, 64, 64, 64),
        output_shape: Tuple[int, ...] = (3, 64, 64, 64),
        variable_names: List[str] = ["u", "v", "w"],
        macro_perturbations: int = 10,
        micro_perturbations: int = 1000,
        t_start: int = 0,
        t_final: int = 5,
        start: int = 0,
    ):

        super().__init__(
            training_samples=micro_perturbations * macro_perturbations,
            file_system=metadata,
            move_to_local_scratch=move_to_local_scratch,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            t_start=t_start,
            t_final=t_final,
            start=start,
            #mean_training_input = np.array([0.0, 0.0, 0.0]),
            #std_training_input = np.array([0.292, 0.292, 0.193]),
            #mean_training_output = np.array([0.0, 0.0, 0.0]),
            #std_training_output = np.array([0.292, 0.292, 0.193]),
        )


class ConditionalNozzle3D(ConditionalIncompressibleFlows3D):
    def __init__(
        self,
        metadata: Dict[str, Any],
        move_to_local_scratch: bool = False,
        input_channel: int = 4,
        output_channel: int = 3,
        spatial_resolution: Tuple[int, ...] = (64, 64, 192),
        input_shape: Tuple[int, ...] = (4, 64, 64, 192),
        output_shape: Tuple[int, ...] = (3, 64, 64, 192),
        variable_names: List[str] = ["u", "v", "w"],
        macro_perturbations: int = 1,
        micro_perturbations: int = 4000,
        t_start: int = 0,
        t_final: int = 10,
        start: int = 0,
    ):

        super().__init__(
            training_samples=micro_perturbations * macro_perturbations,
            file_system=metadata,
            move_to_local_scratch=move_to_local_scratch,
            input_channel=input_channel,
            output_channel=output_channel,
            spatial_resolution=spatial_resolution,
            input_shape=input_shape,
            output_shape=output_shape,
            variable_names=variable_names,
            micro_perturbations=micro_perturbations,
            macro_perturbations=macro_perturbations,
            t_start=t_start,
            t_final=t_final,
            start=start,
        )

    def get_mask(self):
        """Used to mask the output"""
        mask = self.file.variables["mask"][:]
        return torch.from_numpy(mask).type(torch.float32).permute(2, 1, 0)
