from dataloader.dataloader import BrainDataset, MNSIT_Dataset, CIFAR10_Dataset, Wave2d_Dataset, RPB_NavierStokes_Dataset, CustomDataset, Merra2Dataset
from dataloader.dataloader_poseidon import ConditionalATMMSC3DMoistTimeDataset, ATMMSC3DDryTimeDataset, ATMMSC3DMoistTimeDataset, ShearLayer3dMicroMacroTimeDataset, ShearLayerGenCFDMicroMacroTimeDataset,ShearLayerGenCFDTimeDataset, TaylorGreen3dTimeDataset, TaylorGreenN32T50TimeDataset, ShearLayer3dN32T50TimeDataset, CloudShock3dTimeDataset,MERRA2Dataset,RiemannEllipse3dTimeDataset, Riemann3dTimeDataset, RiemannCurved3dTimeDataset, ShearLayer3dTimeDataset, KelvinHelmholtz3dTimeDataset, OrszagTang8TimeDataset, CahnEquations,PoissonGaussians,WaveSeismic,WaveGaussians,RichtmyerMeshkov,BrownianBridgeTimeDataset, VortexSheetTimeDataset, SinesTimeDataset, SinesEasyTimeDataset, PiecewiseConstantsTimeDataset, GaussiansTimeDataset, ComplicatedShearLayerTimeDataset, EulerGaussTimeDataset, RiemannTimeDataset,RiemannKHTimeDataset, RiemannCurvedTimeDataset, KelvinHelmholtzTimeDataset, Helmholtz
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Subset, DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import json
from pathlib import Path
import netCDF4
import numpy as np
import wandb
import torch
import time
import ast
import albumentations as A

def save_data(D, filename="data.json"):
    with open(filename, "w") as f:
        json.dump(D, f, indent=4)

def load_data(filename="data.json"):
    with open(filename, "r") as f:
        return json.load(f)

def save_errors(file_name,
                error,
                rel_error,
                likelihood,
                p = 2):
    
    with netCDF4.Dataset(file_name, "w", format="NETCDF4") as ncfile:
        # Define a dimension (same for all arrays)
        
        if rel_error is not None:
            n_samples = rel_error.shape[0]
        else:
            n_samples = likelihood.shape[0]
        
        ncfile.createDimension("samples", n_samples)
        ncfile.createDimension("p", p)

        # Create variables with the same dimension
        if error is not None:
            error_var = ncfile.createVariable("error", "f4", ("samples",))
            error_rel_var = ncfile.createVariable("error_rel", "f4", ("samples",))
            error_var.description = "Lp testing errors of samples"
            error_rel_var.description = "Relative Lp errors of samples"
            
            error_var[:] = error
            error_rel_var[:] = rel_error
            
            mean_lp = ncfile.createVariable("mean_error", "f4")
            median_lp = ncfile.createVariable("median_error", "f4")
            std_lp = ncfile.createVariable("std_error", "f4")
            mean_lp.assignValue(np.mean(error))
            median_lp.assignValue(np.median(error))
            std_lp.assignValue(np.std(error))

            mean_rel_lp = ncfile.createVariable("mean_rel_error", "f4")
            median_rel_lp = ncfile.createVariable("median_rel_error", "f4")
            std_red_lp = ncfile.createVariable("std_rel_error", "f4")
            mean_rel_lp.assignValue(np.mean(rel_error))
            median_rel_lp.assignValue(np.median(rel_error))
            std_red_lp.assignValue(np.std(rel_error))
        
        if likelihood is not None:
            likelihood_var = ncfile.createVariable("likelihood", "f4", ("samples",))
            likelihood_var.description = "Likelihoods of the samples"

            # Store data in variables
            likelihood_var[:] = likelihood

            mean_likelihood = ncfile.createVariable("mean_likelihood", "f4")
            median_likelihood = ncfile.createVariable("median_likelihood", "f4",)
            std_likelihood = ncfile.createVariable("std_likelihood", "f4")
            mean_likelihood.assignValue(np.mean(likelihood))
            median_likelihood.assignValue(np.median(likelihood))
            std_likelihood.assignValue(np.std(likelihood))

def load_errors(file_name, only_likelihood = False, only_errors = False):
    with netCDF4.Dataset(file_name, "r", format="NETCDF4") as ncfile:
        # Read data from variables
        if not only_likelihood:
            error = np.array(ncfile.variables["error"][:])
            error_rel = np.array(ncfile.variables["error_rel"][:])
        else:
            error = None
            error_rel = None
        
        if not only_errors:
            likelihood =  np.array(ncfile.variables["likelihood"][:])
        else:
            likelihood = None
        
    return error, error_rel, likelihood

def find_files_with_extension(folder_path, extension = "pth", tags = [], is_pl = False):
    real_paths = []
    potential_paths =  list(Path(folder_path).glob(f"*.{extension}"))
    print(potential_paths)
    for path in potential_paths:
        is_valid = True
        for tag in tags:
            is_valid = is_valid and (tag in str(path))
        if is_valid:
            real_paths.append(str(path))
    
    print(real_paths)
    print(" ")
    
    if is_pl:
        max_step = 0
        max_path = None
        for path in real_paths:
            #print(path)
            split_path = path.split("=")[-1].split(".")[0]
            if "v"  in split_path:
                split_path = split_path.split("-")[0]
            step = int(split_path)

            if step>max_step:
                max_path = path
                max_step = step
        return [max_path]
    else:
        return real_paths


def read_cli_regression(parser):
    """Reads command line arguments."""
    # Existing arguments
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--device", type=str, default = 'cuda')
    
    parser.add_argument("--config_arch", type = str, default = "/cluster/home/braonic/ViT_FM/configs/data_regression/config_regression_small_vit3_rkh.json")

    parser.add_argument("--which_model", type=str, default = 'basic_vit3')
    parser.add_argument("--tag", type=str, default = '')
    parser.add_argument("--loss", type=int, default = 1)

    parser.add_argument("--workdir", type=str, default = None)

    parser.add_argument("--epochs", type=int, default = 100)
    parser.add_argument("--warmup_epochs", type=int, default = 2)
    parser.add_argument("--batch_size", type=int, default = 20)
    parser.add_argument("--peak_lr", type=float, default = 1e-4)
    parser.add_argument("--end_lr", type=float, default = 1e-6)
    
    parser.add_argument("--is_time", type=bool, default = False)
    #parser.add_argument("--is_masked", type=bool, default = False)

    parser.add_argument("--is_masked", type=lambda x: x.lower() == "true" or str(x).lower() == "true")
    

    parser.add_argument("--which_data", type=str, default = 'wave')
    parser.add_argument("--in_dim", type=int, default = 1)
    parser.add_argument("--out_dim", type=int, default = 1)
    parser.add_argument("--N_train", type=int, default = 1000)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--s", type=int, default = 128)

    parser.add_argument("--is_fourier_emb", type=bool, default = True)

    parser.add_argument(
        "--allowed_transitions",
        nargs="+",
        type=int,
        default=[1,2,3,4,5,6,7],
        help="list of allowed transitions, e.g. `--allowed_transitions 1 2 3 4 5 6 7`"
    )
    parser.add_argument("--max_num_time_steps", type=int, default=7)
    parser.add_argument("--time_step_size", type=int, default=2)
    parser.add_argument("--fix_input_to_time_step", type=bool, default = None)

    parser.add_argument("--wandb_run_name", type=str, required=False, default=None, help="Name of the run in wandb")
    parser.add_argument("--wandb_project_name", type=str, default="diffusion-project", help="Name of the wandb project")

    return parser

def read_cli_finetune(parser):

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--which_model", type=str, default='basic_vit3')
    parser.add_argument("--tag", type=str, default = "")

    parser.add_argument("--N_train", type=int, default = 128)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--end_lr", type=float, default=1e-6)
    parser.add_argument("--loss_type", type=str, default="rel")

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--which_data", type=str, default="ns_pwc")
    parser.add_argument("--in_dim", type=int, default=4)
    parser.add_argument("--out_dim", type=int, default=4)

    parser.add_argument(
        "--err_group",
        nargs="+",
        type=int,
        default=[1, 1],
        help="error-group sizes, e.g. `--err_group 1 2 1`"
    )
    parser.add_argument(
        "--err_mask_group",
        nargs="+",
        type=int,
        default=[1, 0],
        help="mask group flags, e.g. `--err_mask_group 0 1 0`"
    )
    parser.add_argument(
        "--allowed_transitions",
        nargs="+",
        type=int,
        default=[1,2,3,4,5,6,7],
        help="list of allowed transitions, e.g. `--allowed_transitions 1 2 3 4 5 6 7`"
    )
    
    parser.add_argument("--is_time", type=lambda x: x == "True", default=True)
    parser.add_argument("--is_masked", type=lambda x: x == "True", default=True)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--accumulate_grad", type=int, default=2)

    parser.add_argument("--reinit_ft", type=lambda x: x == "True", default=True)
    parser.add_argument("--is_post_trained", type=lambda x: x == "True", default=False)
    parser.add_argument("--is_3d_scratch", type=lambda x: x == "True", default=False)
    parser.add_argument("--is_precision_16", type=lambda x: x == "True", default=False)
    parser.add_argument("--init_new", type=lambda x: x == "True", default=False)

    parser.add_argument("--rescale_time", type=lambda x: x == "True", default=False)
    parser.add_argument("--ar_train", type=lambda x: x == "True", default=False)
    
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_sdpa", type=lambda x: x == "True", default=True)
    parser.add_argument("--attention_chunk_size", type=int, default=0)

    parser.add_argument("--loss", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--ft_encoder_decoder_warmup_steps",
        type=int,
        default=0,
        help=(
            "Number of optimizer/gradient steps at the beginning of fine-tuning "
            "where only the replaced encoder/decoder I/O modules are trainable. "
            "After these steps, the full model is unfrozen."
        ),
    )
    parser.add_argument("--max_num_time_steps", type=int, default=7)
    parser.add_argument("--time_step_size", type=int, default=2)
    parser.add_argument("--s", type=int, default=128)
    parser.add_argument("--config_regression", type=str, default=None)
    parser.add_argument("--wandb-project-name", type=str, default="foundation-model")
    parser.add_argument("--wandb-run-name", type=str, default = "")

    parser.add_argument("--s_new", type=ast.literal_eval, default=None)
    parser.add_argument("--patch_size_new", type=ast.literal_eval, default=None)
    parser.add_argument(
        "--interpolate_patch_weights",
        type=lambda x: x == "True",
        default=False,
        help=(
            "If True, when finetuning with a different patch size from the "
            "pretrained checkpoint, transfer the pretrained patch-embed and "
            "depatchify linear weights by spatially adaptive_avg_pool3d-ing "
            "them from the pretrained cubic patch grid down to the new one. "
            "The launcher will then load the checkpoint at the pretrained "
            "geometry and re-resize the patch modules afterwards."
        ),
    )

    return parser


def read_cli_diffusion(parser):
    """Reads command line arguments."""
    # Existing arguments
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--device", type=str, default = 'cuda')
    
    parser.add_argument("--tag", type=str, default = '')

    parser.add_argument("--epochs", type=int, default = 100)
    parser.add_argument("--warmup_epochs", type=int, default = 2)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--peak_lr", type=float, default = 0.005)
    parser.add_argument("--end_lr", type=float, default = 1e-5)
    
    parser.add_argument("--which_data", type=str, default = 'wave')
    parser.add_argument("--which_type", type=str, default = 'xy')
    
    parser.add_argument("--sigma", type=float, default = 25.0)
    parser.add_argument("--in_dim", type=int, default = 1)
    parser.add_argument("--out_dim", type=int, default = 1)
    parser.add_argument("--N_train", type=int, default = 1000)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--s", type=int, default = 128)

    parser.add_argument("--unet_param", type=list, default = [64, 128, 256, 512])

    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Name of the run in wandb")
    parser.add_argument("--wandb-project-name", type=str, default="diffusion-project", help="Name of the wandb project")

    return parser

def read_config(parser):
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    return parser

def read_cli_diffusion_gencfd(parser):
    """Reads command line arguments."""
    # Existing arguments
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--config_arch", type = str, default = "/cluster/home/braonic/ood_generalization/configs/architectures/config_unet_small.json")

    parser.add_argument("--device", type=str, default = 'cuda')
    
    parser.add_argument("--tag", type=str, default = '')

    parser.add_argument("--epochs", type=int, default = 100)
    parser.add_argument("--warmup_epochs", type=int, default = 2)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--peak_lr", type=float, default = 0.005)
    parser.add_argument("--end_lr", type=float, default = 1e-5)
    
    parser.add_argument("--which_data", type=str, default = 'wave')
    parser.add_argument("--which_type", type=str, default = 'xy')
    parser.add_argument("--is_time", type=bool, default = False)
    parser.add_argument("--is_masked", type=bool, default = False)

    parser.add_argument("--sigma", type=float, default = 25.0)
    parser.add_argument("--in_dim", type=int, default = 1)
    parser.add_argument("--out_dim", type=int, default = 1)
    parser.add_argument("--N_train", type=int, default = 1000)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--s", type=int, default = 128)
    parser.add_argument("--ema_param", type=float, default = 0.999)

    parser.add_argument("--unet_param", type=list, default = [64, 128, 256, 512])
    parser.add_argument("--is_log_uniform", type=bool, default = False)
    parser.add_argument("--is_exploding", type=bool, default = False)
    parser.add_argument("--log_uniform_frac", type=float, default = 3)

    parser.add_argument("--skip", type=bool, default = False)

    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Name of the run in wandb")
    parser.add_argument("--wandb-project-name", type=str, default="diffusion-project", help="Name of the wandb project")

    return parser

def read_cli_inference(parser):
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--config_regression", type=str, default = None, help="Path to regression model")
    parser.add_argument("--config_diffusion", type=str, default = None, help="Path to diffusion model")
    parser.add_argument("--which_data", type=str, default = None, help="which testing dataset")
    parser.add_argument("--tag_data", type=str, default = None, help="tag for testing dataset")
    parser.add_argument("--device", type=str, default = "cuda")
    parser.add_argument("--N_samples", type=int, default = 128)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--save_data", type=bool, default = False)
    parser.add_argument("--inference_tag", type=str, default = "")

    parser.add_argument("--is_log_uniform", type=bool, default = False)
    parser.add_argument("--is_exploding", type=bool, default = False)
    parser.add_argument("--log_uniform_frac", type=float, default = 3)

    return parser

def save_id(run_, filepath):
    print(f"Sweep ID: {run_.sweep_id}")
    print(f"Run ID: {run_.id}")
    with open(filepath + '/ids.txt', 'w') as file:
        file.write(f"Sweep ID: {run_.sweep_id}\n")
        file.write(f"Run ID: {run_.id}")

def get_loader(which_data: str,
            which_type: str,
            N_samples:int,
            batch_size: int,
            ood_tag: int = None,
            ood_share:float = 0.0,
            num_workers:int = 4,
            in_dim: int = None,
            out_dim: int = None,
            use_generated: str = None,
            masked_input: list = None,
            is_time: bool = False,
            max_num_time_steps: int = None,
            time_step_size: int = None,
            fix_input_to_time_step: int = None,
            allowed_transitions: list = None,
            return_loader: bool = True,
            rel_time: bool = True,
            transpose_shear: bool = False,
            ar_train: bool = False,
            shuffle_: bool = None,
            set_batch_to_full_traj: bool = False,
            is_spectral_resolver: bool = False,
            spectral_file: str = None,
            curr_macro: int = 0,
            N_micro: int = 128,
            macro_id: int = 2,
            random_train_sampling: bool = True,
            ):
    
    if which_type == "train":
        shuffle = True
    else:
        shuffle = False

    if shuffle_ is not None:
        shuffle = shuffle_

    if which_data == "mnist":
        dataset = MNIST(root = "/cluster/work/math/camlab-data/MNIST", train=True, transform=transforms.ToTensor(), download=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    elif which_data == "cifar_diff":
        dataset = CIFAR10_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "cifar_class":
        dataset = CIFAR10_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    elif which_data == "mnist_diff":
        dataset = MNSIT_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "mnist_class":
        dataset = MNSIT_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    elif which_data == "wave":
        dataset = Wave2d_Dataset(which = which_type, N_samples = N_samples, is_time = is_time)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "wave_ood":
        dataset = Wave2d_Dataset(which = which_type, N_samples = N_samples, is_ood=ood_tag, is_time = is_time)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    elif which_data == "shear_layer_rpb":
        dataset = RPB_NavierStokes_Dataset(which = which_type, N_samples = N_samples, is_ood=False, use_generated = use_generated)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "shear_layer_rpb_ood":
        dataset = RPB_NavierStokes_Dataset(which = which_type, N_samples = N_samples, is_ood=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "custom":
        dataset = CustomDataset(folder = which_type, N_samples = N_samples, in_dim = in_dim, out_dim=out_dim)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    elif which_data == "brain":
        
        trasnform = None
        if which_type == "train": 
            transform = A.Compose([
                                A.Resize(width=128, height=128, p=1.0),
                                A.HorizontalFlip(p=0.5)])
        elif which_type == "val":
            transform = A.Compose([
            A.Resize(width=128, height=128, p=1.0),
            A.HorizontalFlip(p=0.5),])

        dataset = BrainDataset("/cluster/work/math/braonic/data/brats2018_HGG_t1.nc", which = which_type, transform=transform, is_ood = False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    elif which_data == "ns_shear":
        dataset = ComplicatedShearLayerTimeDataset(max_num_time_steps = max_num_time_steps, 
                                                    time_step_size = time_step_size,
                                                    fix_input_to_time_step = fix_input_to_time_step,
                                                    which = which_type,
                                                    resolution = 128,
                                                    in_dist = True,
                                                    num_trajectories = N_samples,
                                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                                    time_input = True,
                                                    masked_input = masked_input,
                                                    allowed_transitions = allowed_transitions,
                                                    rel_time = rel_time,
                                                    is_time = is_time)
    
    elif which_data == "ns_shear_gencfd":
        dataset = ShearLayerGenCFDTimeDataset(max_num_time_steps = 1, 
                                            time_step_size = 1.0,
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_type,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = N_samples,
                                            data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                            time_input = True,
                                            masked_input = masked_input,
                                            allowed_transitions = [1],
                                            rel_time = rel_time,
                                            is_time = is_time,
                                            in_dim = in_dim,
                                            out_dim = out_dim,
                                            is_spectral_resolver = is_spectral_resolver,
                                            spectral_file=spectral_file)
    
    elif which_data == "ns_shear_gencfd_mm":
        dataset = ShearLayerGenCFDMicroMacroTimeDataset(max_num_time_steps = 1, 
                                            time_step_size = 1.0,
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_type,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = N_samples,
                                            data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                            time_input = True,
                                            masked_input = masked_input,
                                            allowed_transitions = [1],
                                            is_time = is_time,
                                            in_dim = in_dim,
                                            out_dim = out_dim,
                                            is_spectral_resolver = is_spectral_resolver,
                                            spectral_file=spectral_file,
                                            curr_macro = curr_macro,
                                            N_micro = N_micro)

    elif which_data == "ns_vortex":
        dataset = VortexSheetTimeDataset(max_num_time_steps = max_num_time_steps, 
                                                    time_step_size = time_step_size,
                                                    fix_input_to_time_step = fix_input_to_time_step,
                                                    which = which_type,
                                                    resolution = 128,
                                                    in_dist = True,
                                                    num_trajectories = N_samples,
                                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                                    time_input = True,
                                                    masked_input = masked_input,
                                                    allowed_transitions = allowed_transitions,
                                                    rel_time = rel_time,
                                                    is_time = is_time)
    elif which_data == "ns_brownian":
        dataset = BrownianBridgeTimeDataset(max_num_time_steps = max_num_time_steps, 
                                                    time_step_size = time_step_size,
                                                    fix_input_to_time_step = fix_input_to_time_step,
                                                    which = which_type,
                                                    resolution = 128,
                                                    in_dist = True,
                                                    num_trajectories = N_samples,
                                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                                    time_input = True,
                                                    masked_input = masked_input,
                                                    allowed_transitions = allowed_transitions,
                                                    rel_time = rel_time,
                                                    is_time = is_time)
        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif which_data == "ns_sin":
        dataset = SinesTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time,
                                    in_dim = in_dim,
                                    out_dim = out_dim,
                                    copy_to_local_scratch = False)
        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    elif which_data == "ns_sin_easy":
        dataset = SinesEasyTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = N_samples,
                                        data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        rel_time = rel_time,
                                        is_time = is_time)
        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif which_data == "ns_pwc":
        dataset = PiecewiseConstantsTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dim = in_dim,
                                    out_dim = out_dim,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time,
                                    copy_to_local_scratch = False)
    
    elif which_data == "mhd_orszag8":
        dataset = OrszagTang8TimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dim = in_dim,
                                        out_dim = out_dim,
                                        in_dist = True,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        rel_time = rel_time,
                                        is_time = is_time,
                                        copy_to_local_scratch = False,
                                        rescale_time = True)
    elif which_data == "mhd_orszag8_long":
        print(in_dim, out_dim)
        dataset = OrszagTang8TimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dim = in_dim,
                                        out_dim = out_dim,
                                        in_dist = True,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        rel_time = rel_time,
                                        is_time = is_time,
                                        copy_to_local_scratch = False,
                                        rescale_time = False,
                                        ar_training = ar_train)

    elif which_data == "ns_gauss":
        dataset = GaussiansTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    in_dim = in_dim,
                                    out_dim = out_dim,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)

    elif which_data == "wave_seismic":
        dataset = WaveSeismic(max_num_time_steps = max_num_time_steps, 
                            time_step_size = time_step_size,
                            fix_input_to_time_step = fix_input_to_time_step,
                            in_dim=in_dim,
                            out_dim = out_dim,
                            which = which_type,
                            resolution = 128,
                            in_dist = True,
                            num_trajectories = N_samples,
                            data_path = "",
                            time_input = True,
                            masked_input = masked_input,
                            allowed_transitions = allowed_transitions)

    elif which_data == "ns_mix1":
        dataset1 = SinesTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)
        dataset2 = ComplicatedShearLayerTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)
        dataset3 = GaussiansTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = N_samples,
                                        data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        rel_time = rel_time,
                                        is_time = is_time)
        dataset = [dataset1, dataset2, dataset3]
        dataset = torch.utils.data.ConcatDataset(dataset)

    elif which_data == "eul_ns_mix1":
        dataset1 = SinesTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 19640),
                                    data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)

        dataset2 = GaussiansTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 19640),
                                        data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        rel_time = rel_time,
                                        is_time = is_time)
        
        dataset3 = EulerGaussTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 9640),
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions)
        
        dataset4 = RiemannTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 9640),
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions)
        
        dataset5 = RiemannCurvedTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 9640),
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions)
        
        dataset6 = KelvinHelmholtzTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 9640),
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions)

        dataset = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]
        dataset = torch.utils.data.ConcatDataset(dataset)


    elif which_data == "helmholtz":
        dataset = Helmholtz(
                            which = which_type,
                            resolution = 128,
                            in_dist = True,
                            num_trajectories = N_samples,
                            data_path = "",
                            time_input = True,
                            augment = True,
                            masked_input = masked_input,
                            in_dim = 9,
                            out_dim = 9,
                            oversample = 1,
                           )

    elif which_data == "eul_gauss":
        dataset = EulerGaussTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "",
                                    in_dim = in_dim,
                                    out_dim=out_dim,
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions)
    elif which_data == "allen_cahn":
        dataset = CahnEquations(is_allen_cahn = True,
                                max_num_time_steps = max_num_time_steps, 
                                time_step_size = time_step_size,
                                fix_input_to_time_step = fix_input_to_time_step,
                                in_dim = in_dim,
                                out_dim = out_dim,
                                which = which_type,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = N_samples,
                                data_path = "",
                                time_input = True,
                                masked_input = masked_input,
                                allowed_transitions = allowed_transitions,
                                copy_to_local_scratch = False)

    elif which_data == "eul_riemann_kh":
        dataset = RiemannKHTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)
    elif which_data == "eul_riemann_curved":
        dataset = RiemannCurvedTimeDataset(max_num_time_steps = max_num_time_steps, 
                                            time_step_size = time_step_size,
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_type,
                                            resolution = 128,
                                            in_dist = True,
                                            in_dim = in_dim,
                                            out_dim = out_dim,
                                            num_trajectories = N_samples,
                                            data_path = "",
                                            time_input = True,
                                            masked_input = masked_input,
                                            allowed_transitions = allowed_transitions,
                                            copy_to_local_scratch = True,
                                            is_spectral_resolver = is_spectral_resolver,
                                            spectral_file=spectral_file)
    elif which_data == "eul_riemann_kh3d":
        dataset = KelvinHelmholtz3dTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)
    elif which_data == "eul_riemann_ellipse3d":
        dataset = RiemannEllipse3dTimeDataset(max_num_time_steps = max_num_time_steps, 
                                            time_step_size = time_step_size,
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_type,
                                            resolution = 64,
                                            in_dist = True,
                                            in_dim = in_dim,
                                            out_dim=out_dim,
                                            num_trajectories = N_samples,
                                            data_path = "",
                                            time_input = True,
                                            masked_input = masked_input,
                                            allowed_transitions = allowed_transitions,
                                            copy_to_local_scratch = False)
    elif which_data == "ns_shear3d":
        dataset = ShearLayer3dTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)

    elif which_data == "ns_shear3d_mm":
        dataset = ShearLayer3dMicroMacroTimeDataset(max_num_time_steps = max_num_time_steps, 
                                            time_step_size = time_step_size,
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_type,
                                            resolution = 64,
                                            in_dist = True,
                                            in_dim = in_dim,
                                            out_dim=out_dim,
                                            num_trajectories = N_samples,
                                            data_path = "",
                                            time_input = True,
                                            masked_input = masked_input,
                                            allowed_transitions = allowed_transitions,
                                            copy_to_local_scratch = True,
                                            curr_macro = curr_macro)

    elif which_data == "eul_riemann3d":
        dataset = Riemann3dTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)
    elif which_data == "eul_riemann_curved3d":
        dataset = RiemannCurved3dTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)
    elif which_data == "tg3d":
        dataset = TaylorGreen3dTimeDataset(max_num_time_steps = 5,
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 8450),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5],
                                        copy_to_local_scratch = False,
                                        perturb_p = False)

    elif which_data == "tg3d_n32t50":
        # Windowed all-to-all finetune loader on the dense, 32^3, 51-snapshot TG3D file.
        # - random_train_sampling=True (default): the dataset ignores
        #   max_num_time_steps / time_step_size / allowed_transitions and uses
        #   time_window=8 / train_multiplier=50; train draws (t1, dt) randomly.
        # - random_train_sampling=False: the dataset deterministically enumerates
        #   (t1, t2) pairs from time_step_size / max_num_time_steps /
        #   allowed_transitions for ALL splits (coarse mode).
        _tg3d_max_steps = max_num_time_steps if max_num_time_steps is not None else 8
        _tg3d_step_size = time_step_size if time_step_size is not None else 1
        _tg3d_allowed = allowed_transitions if allowed_transitions is not None else list(range(1, 9))
        dataset = TaylorGreenN32T50TimeDataset(max_num_time_steps = _tg3d_max_steps,
                                        time_step_size = _tg3d_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 32,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim = out_dim,
                                        num_trajectories = min(N_samples, 9500),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = _tg3d_allowed,
                                        copy_to_local_scratch = True,
                                        perturb_p = False,
                                        time_window = 8,
                                        train_multiplier = 50,
                                        random_train_sampling = random_train_sampling)

    elif which_data == "shear3d_n32t50":
        # Windowed all-to-all loader on the dense, 32^3, 51-snapshot
        # Cylindrical Shear Flow 3D file. Same plumbing as `tg3d_n32t50`, only
        # the underlying class differs (different file path, stats and
        # REAL_DT_RATIO so the time embedding matches pretraining).
        _sh3d_max_steps = max_num_time_steps if max_num_time_steps is not None else 8
        _sh3d_step_size = time_step_size if time_step_size is not None else 1
        _sh3d_allowed = allowed_transitions if allowed_transitions is not None else list(range(1, 9))
        dataset = ShearLayer3dN32T50TimeDataset(max_num_time_steps = _sh3d_max_steps,
                                        time_step_size = _sh3d_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 32,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim = out_dim,
                                        num_trajectories = min(N_samples, 9500),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = _sh3d_allowed,
                                        copy_to_local_scratch = True,
                                        perturb_p = False,
                                        time_window = 8,
                                        train_multiplier = 50,
                                        random_train_sampling = random_train_sampling)

    elif which_data == "eul3d_mix1":
        dataset1 = Riemann3dTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)

        dataset2 = RiemannCurved3dTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)
        
        dataset = [dataset1, dataset2]
        dataset = torch.utils.data.ConcatDataset(dataset)

    elif which_data == "atm_msc_3d_moist":
        dataset = ATMMSC3DMoistTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 96,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = True)

    elif which_data == "atm_msc_3d_dry":
        dataset = ATMMSC3DDryTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False)

    elif which_data == "conditional_atm_msc_3d_moist":
        dataset = ConditionalATMMSC3DMoistTimeDataset(max_num_time_steps = max_num_time_steps,
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 96,
                                        in_dist = False,
                                        in_dim = in_dim,
                                        out_dim = out_dim,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        copy_to_local_scratch = False,
                                        macro_id = macro_id)

    elif which_data == "eul_ns3d_mix1":
        dataset1 = Riemann3dTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 4700),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        copy_to_local_scratch = False,
                                        perturb_p = False) 

        dataset2 = RiemannCurved3dTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 4700),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        copy_to_local_scratch = False,
                                        perturb_p = False)   
        
        dataset3 = CloudShock3dTimeDataset(max_num_time_steps = 4, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 7500),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5],
                                        copy_to_local_scratch = False,
                                        perturb_p = False) 
        
        dataset4 = TaylorGreen3dTimeDataset(max_num_time_steps = 5, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 8450),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5],
                                        copy_to_local_scratch = False,
                                        perturb_p = False)   

        dataset5 = ShearLayer3dTimeDataset(max_num_time_steps = 4, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 9000),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5],
                                        copy_to_local_scratch = False,
                                        perturb_p = False)                        
        
        dataset = [dataset1, dataset2, dataset3, dataset4, dataset5]
        dataset = torch.utils.data.ConcatDataset(dataset)

    elif which_data == "ns3d_mix1":
        dataset1 = Riemann3dTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 4700),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        copy_to_local_scratch = False)

        dataset2 = RiemannCurved3dTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 4700),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        copy_to_local_scratch = False)
        
        dataset4 = TaylorGreen3dTimeDataset(max_num_time_steps = 5, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 8450),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,6],
                                        copy_to_local_scratch = False)

        dataset5 = ShearLayer3dTimeDataset(max_num_time_steps = 4, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 64,
                                        in_dist = True,
                                        in_dim = in_dim,
                                        out_dim=out_dim,
                                        num_trajectories = min(N_samples, 9000),
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = [1,2,3,4,5],
                                        copy_to_local_scratch = False)                        
        
        dataset = [dataset1, dataset2, dataset4, dataset5]
        dataset = torch.utils.data.ConcatDataset(dataset)

    elif which_data == "rich_mesh":
        dataset = RichtmyerMeshkov(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions)

    elif which_data == "poisson":
        dataset = PoissonGaussians(which = which_type,
                                    in_dim=in_dim,
                                    out_dim=out_dim,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    masked_input = masked_input)

    elif which_data == "pdegym_plus":
        
        dataset1 = SinesTimeDataset(max_num_time_steps = 10, 
                                    time_step_size = 2,
                                    fix_input_to_time_step = None,
                                    in_dim = 9,
                                    out_dim = 9,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 19640),
                                    data_path = '',
                                    time_input = True,
                                    masked_input = True,
                                    allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                    augment = False,
                                    copy_to_local_scratch = True)

        dataset2 = GaussiansTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 2,
                                        fix_input_to_time_step = None,
                                        in_dim = 9,
                                        out_dim = 9,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 19640),
                                        data_path = '',
                                        time_input = True,
                                        masked_input = True,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        augment = False,
                                        copy_to_local_scratch = True)
        
        dataset3 = EulerGaussTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 2,
                                        fix_input_to_time_step = None,
                                        in_dim = 9,
                                        out_dim = 9,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 9640),
                                        data_path = '',
                                        time_input = True,
                                        masked_input = True,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        augment = False,
                                        copy_to_local_scratch = True)
        
        dataset4 = RiemannTimeDataset(max_num_time_steps = 10, 
                                    time_step_size = 2,
                                    fix_input_to_time_step = None,
                                    in_dim = 9,
                                    out_dim = 9,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 9640),
                                    data_path = '',
                                    time_input = True,
                                    masked_input = True,
                                    allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                    augment = False,
                                    copy_to_local_scratch = True)
        
        dataset5 = RiemannCurvedTimeDataset(max_num_time_steps = 10, 
                                            time_step_size = 2,
                                            fix_input_to_time_step = None,
                                            in_dim = 9,
                                            out_dim = 9,
                                            which = which_type,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = min(N_samples, 9640),
                                            data_path = '',
                                            time_input = True,
                                            masked_input = True,
                                            allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                            augment = False,
                                            copy_to_local_scratch = True)
        
        dataset6 = KelvinHelmholtzTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 2,
                                        fix_input_to_time_step = None,
                                        in_dim = 9,
                                        out_dim = 9,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 9640),
                                        data_path = '',
                                        time_input = True,
                                        masked_input = True,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        augment = False,
                                        copy_to_local_scratch = True)
        
        dataset7 = WaveGaussians(max_num_time_steps = 15, 
                                time_step_size = 1,
                                fix_input_to_time_step = None,
                                in_dim = 9,
                                out_dim = 9,
                                which = which_type,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = min(N_samples, 10512 - 60 - 240),
                                data_path = '',
                                time_input = True,
                                masked_input = True,
                                allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                augment = False,
                                copy_to_local_scratch = True)
        
        dataset8 = CahnEquations(is_allen_cahn = False,
                                max_num_time_steps = 20, 
                                time_step_size = 1,
                                fix_input_to_time_step = None,
                                in_dim = 9,
                                out_dim = 9,
                                which = which_type,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = min(N_samples, 4096 - 60 - 120),
                                data_path = '',
                                time_input = True,
                                masked_input = True,
                                allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                augment = False,
                                copy_to_local_scratch = True)
        
        dataset9 = Helmholtz(which = which_type,
                            resolution = 128,
                            in_dist = True,
                            num_trajectories = 19675 - 128 - 512,
                            data_path = "",
                            time_input = True,
                            augment = True,
                            masked_input = True,
                            in_dim = 9,
                            out_dim = 9,
                            oversample = 4,
                            copy_to_local_scratch = True)

        dataset = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9]
        dataset = torch.utils.data.ConcatDataset(dataset)
    
    elif which_data == "pdegym_giga":
        
        dataset1 = SinesTimeDataset(max_num_time_steps = 20, 
                                    time_step_size = 1,
                                    fix_input_to_time_step = None,
                                    in_dim = 9,
                                    out_dim = 9,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 19640),
                                    data_path = '',
                                    time_input = True,
                                    masked_input = True,
                                    allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                    augment = False,
                                    copy_to_local_scratch = True)

        dataset2 = GaussiansTimeDataset(max_num_time_steps = 20, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = None,
                                        in_dim = 9,
                                        out_dim = 9,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 19640),
                                        data_path = '',
                                        time_input = True,
                                        masked_input = True,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                        augment = False,
                                        copy_to_local_scratch = True)
        
        dataset3 = EulerGaussTimeDataset(max_num_time_steps = 20, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = None,
                                        in_dim = 9,
                                        out_dim = 9,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 9640),
                                        data_path = '',
                                        time_input = True,
                                        masked_input = True,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                        augment = False,
                                        copy_to_local_scratch = True)
        
        dataset4 = RiemannTimeDataset(max_num_time_steps = 20, 
                                    time_step_size = 1,
                                    fix_input_to_time_step = None,
                                    in_dim = 9,
                                    out_dim = 9,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 9640),
                                    data_path = '',
                                    time_input = True,
                                    masked_input = True,
                                    allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                    augment = False,
                                    copy_to_local_scratch = True)
        
        dataset5 = RiemannCurvedTimeDataset(max_num_time_steps = 20, 
                                            time_step_size = 1,
                                            fix_input_to_time_step = None,
                                            in_dim = 9,
                                            out_dim = 9,
                                            which = which_type,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = min(N_samples, 9640),
                                            data_path = '',
                                            time_input = True,
                                            masked_input = True,
                                            allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                            augment = False,
                                            copy_to_local_scratch = True)
        
        dataset6 = KelvinHelmholtzTimeDataset(max_num_time_steps = 20, 
                                        time_step_size = 1,
                                        fix_input_to_time_step = None,
                                        in_dim = 9,
                                        out_dim = 9,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 9640),
                                        data_path = '',
                                        time_input = True,
                                        masked_input = True,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                        augment = False,
                                        copy_to_local_scratch = True)

        dataset7 = WaveGaussians(max_num_time_steps = 15, 
                                time_step_size = 1,
                                fix_input_to_time_step = None,
                                in_dim = 9,
                                out_dim = 9,
                                which = which_type,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = min(N_samples, 10512 - 60 - 240),
                                data_path = '',
                                time_input = True,
                                masked_input = True,
                                allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                augment = False,
                                copy_to_local_scratch = True)
        
        dataset8 = CahnEquations(is_allen_cahn = False,
                                max_num_time_steps = 20, 
                                time_step_size = 1,
                                fix_input_to_time_step = None,
                                in_dim = 9,
                                out_dim = 9,
                                which = which_type,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = min(N_samples, 4096 - 60 - 120),
                                data_path = '',
                                time_input = True,
                                masked_input = True,
                                allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                augment = False,
                                copy_to_local_scratch = True)
        
        dataset9 = Helmholtz(which = which_type,
                            resolution = 128,
                            in_dist = True,
                            num_trajectories = 19675 - 128 - 512,
                            data_path = "",
                            time_input = True,
                            augment = True,
                            masked_input = True,
                            in_dim = 9,
                            out_dim = 9,
                            oversample = 2,
                            copy_to_local_scratch = True)

        dataset = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9]
        dataset = torch.utils.data.ConcatDataset(dataset)

    elif which_data == "pdegym_mini":
        
        dataset4 = RiemannTimeDataset(max_num_time_steps = 10, 
                                    time_step_size = 2,
                                    fix_input_to_time_step = None,
                                    in_dim = 9,
                                    out_dim = 9,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = min(N_samples, 9640),
                                    data_path = '',
                                    time_input = True,
                                    masked_input = True,
                                    allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                    augment = True,
                                    copy_to_local_scratch = True)
        
        dataset5 = RiemannCurvedTimeDataset(max_num_time_steps = 10, 
                                            time_step_size = 2,
                                            fix_input_to_time_step = None,
                                            in_dim = 9,
                                            out_dim = 9,
                                            which = which_type,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = min(N_samples, 9640),
                                            data_path = '',
                                            time_input = True,
                                            masked_input = True,
                                            allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                            augment = True,
                                            copy_to_local_scratch = True)
        
        dataset6 = KelvinHelmholtzTimeDataset(max_num_time_steps = 10, 
                                        time_step_size = 2,
                                        fix_input_to_time_step = None,
                                        in_dim = 9,
                                        out_dim = 9,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = min(N_samples, 9640),
                                        data_path = '',
                                        time_input = True,
                                        masked_input = True,
                                        allowed_transitions = [1,2,3,4,5,6,7,8,9,10],
                                        augment = True,
                                        copy_to_local_scratch = True)

        dataset7 = WaveGaussians(max_num_time_steps = 15, 
                                time_step_size = 1,
                                fix_input_to_time_step = None,
                                in_dim = 9,
                                out_dim = 9,
                                which = which_type,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = min(N_samples, 10512 - 60 - 240),
                                data_path = '',
                                time_input = True,
                                masked_input = True,
                                allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                augment = True,
                                copy_to_local_scratch = True)
        
        dataset8 = CahnEquations(is_allen_cahn = False,
                                max_num_time_steps = 20, 
                                time_step_size = 1,
                                fix_input_to_time_step = None,
                                in_dim = 9,
                                out_dim = 9,
                                which = which_type,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = min(N_samples, 4096 - 60 - 120),
                                data_path = '',
                                time_input = True,
                                masked_input = True,
                                allowed_transitions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                                augment = True,
                                copy_to_local_scratch = True)
        
        dataset9 = Helmholtz(which = which_type,
                            resolution = 128,
                            in_dist = True,
                            num_trajectories = 19675 - 128 - 512,
                            data_path = "",
                            time_input = True,
                            augment = True,
                            masked_input = True,
                            in_dim = 9,
                            out_dim = 9,
                            oversample = 2,
                            copy_to_local_scratch = True)

        dataset = [dataset4, dataset5, dataset6, dataset7, dataset8, dataset9]
        dataset = torch.utils.data.ConcatDataset(dataset)

    elif which_data == "merra2":

        dataset = MERRA2Dataset(which = which_type,
                                in_dim = in_dim,
                                out_dim = out_dim,
                                resolution = 128,
                                in_dist = True,
                                num_trajectories = N_samples,
                                data_path = "/cluster/work/math/camlab-data/incompressible_fluids",
                                masked_input = masked_input,
                                allowed_transitions = allowed_transitions,)

    if which_type == "test":
        num_workers = 1
    else:
        num_workers = num_workers
    
    if hasattr(dataset, 'time_indices') and set_batch_to_full_traj:
        _batch_size = len(dataset.time_indices)
        print(dataset.time_indices, _batch_size)
    else:
        _batch_size = batch_size


    if which_type == "val":
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)  # Shuffle indices once
        shuffled_dataset = Subset(dataset, indices)
        loader = DataLoader(shuffled_dataset, 
                            batch_size=_batch_size,
                            shuffle=False, 
                            pin_memory=False, 
                            num_workers=2, 
                            persistent_workers=False)
    else:
        if "mix" in which_data:
            is_train = (which_type == "train")
            loader = DataLoader(dataset, 
                                batch_size=_batch_size,
                                shuffle=is_train, 
                                pin_memory=False, 
                                num_workers=2,
                                prefetch_factor=1,
                                persistent_workers=False)
        else:
            loader = DataLoader(dataset, batch_size=_batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    if return_loader:
        return loader
    else:
        return dataset

def select_variable_condition(input_batch,
                              output_batch,
                              which_type = "x&y",
                              mask = None):
  if which_type == "xy":
    variable = input_batch
    condition = output_batch
  elif which_type == "yx":
    variable = output_batch
    condition = input_batch
  elif which_type == "x":
    variable = input_batch
    condition = None
  elif which_type == "y":
    variable = output_batch
    condition = None
  elif which_type == "x&y":
    variable = torch.cat((input_batch, output_batch), axis = 1)
    condition = None

    if mask is not None:
        mask = torch.cat((mask, mask), dim = 1)
  
  return variable, condition, mask