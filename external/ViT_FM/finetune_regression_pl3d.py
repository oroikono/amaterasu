import torch
import os
import functools
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing
import torch._dynamo

from pytorch_lightning.strategies import FSDPStrategy
import torch.nn as nn

import tqdm
import argparse
import copy
import wandb
import ast
import warnings
import pickle

import matplotlib.pyplot as plt

import time 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from utils.utils_data import get_loader, save_data, load_data, read_cli_finetune, read_cli_regression, save_errors, read_config, select_variable_condition, find_files_with_extension
from regression.loss_fn import relative_lp_loss_fn_3d

from utils.utils_finetune_3d import initialize_FT3d
from regression.ViTModulev2 import  Vit3_pl
from regression.ViTModulev2 import Attention, FeedForward  # adjust if namespaced differently



os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/ood_wandb_logs"

def load_checkpoint_compat(path, map_location='cpu'):
    """
    PyTorch 2.6 changed torch.load default to weights_only=True.
    Older Lightning checkpoints may contain pickled metadata and fail to load
    unless we explicitly fall back to weights_only=False.
    """
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as e:
        if "Weights only load failed" in str(e):
            warnings.warn(
                "Falling back to torch.load(..., weights_only=False) for a trusted checkpoint.",
                RuntimeWarning,
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        raise



def load_state_dict_shape_compat(model, state_dict, strict=True):
    if strict:
        model.load_state_dict(state_dict)
        return

    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if len(skipped) > 0:
        warnings.warn(
            f"Skipped {len(skipped)} incompatible checkpoint tensors (e.g. {skipped[:3]}).",
            RuntimeWarning,
        )
    if len(missing) > 0 or len(unexpected) > 0:
        warnings.warn(
            f"Non-strict load: missing={len(missing)}, unexpected={len(unexpected)}",
            RuntimeWarning,
        )

def is_checkpointable(module: torch.nn.Module) -> bool:
    return isinstance(module, (Attention, FeedForward))

def is_checkpointable_mlp(module: torch.nn.Module) -> bool:
    return isinstance(module, (FeedForward, ))

if __name__ == "__main__":

    torch._dynamo.config.optimize_ddp = False  # disable broken DDP graph fusion
    torch._dynamo.config.suppress_errors = True

    warnings.filterwarnings(
    "ignore",
    message=".*None of the inputs have requires_grad=True.*",
    module="torch.utils.checkpoint"
    )

    #parser = argparse.ArgumentParser(description="load parameters for training")
    #params = read_config(parser).parse_args()    
    #config_ft = argparse.Namespace(**load_data(params.config))
    parser = argparse.ArgumentParser(description="Finetuning regression model")
    parser = read_cli_finetune(parser)
    config_ft = parser.parse_args()
    
    if config_ft.config is not None:
        config_ft = argparse.Namespace(**load_data(config_ft.config))

    workdir_base = config_ft.config_regression
    base_model_path = str(find_files_with_extension(config_ft.config_regression + "/model", "ckpt", [], is_pl = True)[0])
    base_config_path = str(find_files_with_extension(config_ft.config_regression, "json", ["param"])[0])
    config_base = argparse.Namespace(**load_data(base_config_path))

    device = 'cuda'
    tag_ft = config_ft.tag
    p = int(config_ft.loss)
    which_data = config_ft.which_data

    reinit_ft = config_ft.reinit_ft
    init_new = config_ft.init_new
    if reinit_ft:
        type_ft = "non_native"
    else:
        type_ft = "native"

    config_train = vars(config_ft)

    _arch_to_load = config_base.config_arch

    if isinstance(_arch_to_load, dict):
        config_arch = _arch_to_load
    else:
        config_arch = load_data(config_base.config_arch)

    config_train["config_arch"] = config_arch
    config_train["fix_input_to_time_step"] = None
    config_train["num_workers"] = min(max(0, int(getattr(config_ft, "num_workers", 2))), 2)

    workdir = f"{workdir_base}/finetuned/{which_data}/{tag_ft}_FT_{which_data}_{type_ft}_" + str(config_train["N_train"])
    config_train["workdir"] = workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    save_data(vars(config_ft), f"{workdir}/param_regression_{tag_ft}" + str(config_train["N_train"]) + ".json")

    loss = functools.partial(relative_lp_loss_fn_3d, p=p)

    dim0 = config_ft.in_dim
    dim1 = config_ft.out_dim
    
    if "pdegym/" in workdir_base:
        dim0 = 4
        dim1 = 4
    elif "ns_mix" in workdir_base:
        dim0 = 2
        dim1 = 2
    elif "pdegym_" in workdir_base:
        dim0 = 9
        dim1 = 9
    elif "eul_ns3d_mix" in workdir_base: 
        dim0 = 5
        dim1 = 5

    if not config_ft.is_masked:
        config_train["is_masked"] = None 
    
    is_post_trained = config_ft.is_post_trained
    if hasattr(config_ft, "is_3d_scratch"):
        is_3d_scratch = config_ft.is_3d_scratch
    else:
        is_3d_scratch = False

    model = Vit3_pl(in_dim = dim0, 
                    out_dim = dim1,
                    loss_fn = loss,
                    config_train = config_train,
                    config_arch = config_arch)

    #checkpoint = torch.load(base_model_path, map_location = device)
    #model.load_state_dict(checkpoint["state_dict"])
    
    if not is_post_trained and not is_3d_scratch:
        #ckpt = torch.load(base_model_path, map_location='cpu')
        #ckpt = load_checkpoint_compat(base_model_path, map_location='cpu')
        #model.load_state_dict(ckpt['state_dict'])
        ckpt = load_checkpoint_compat(base_model_path, map_location='cpu')
        load_state_dict_shape_compat(model, ckpt['state_dict'], strict=True)

    if not "eul_ns3d_mix1" in which_data:
        model.is_ft = True
    
    model._plot_epoch = 100000

    target_s = config_ft.s_new if hasattr(config_ft, "s_new") else config_ft.s
    if hasattr(config_ft, "patch_size_new") and config_ft.patch_size_new is not None:
        patch_size = config_ft.patch_size_new
    elif isinstance(target_s, (list, tuple)):
        patch_size = [8, 8, 8]
    elif target_s == 64:
        patch_size = 4
    else:
        patch_size = 8

    if is_3d_scratch or not hasattr(config_base, "init_new"):
        init_new_load = False
    else:
        init_new_load = config_base.init_new

    # If we want to transfer pretrained patch weights to a *different* patch
    # geometry, build the model at the pretrained geometry first so the
    # checkpoint loads with strict-matching shapes; we then resize the patch
    # modules in a second pass with interpolation.
    interpolate_patch = bool(getattr(config_ft, "interpolate_patch_weights", False)) and is_post_trained

    if interpolate_patch:
        # Derive pretraining patch geometry from config_base (same rule the
        # launcher uses above, applied to the pretraining `s`).
        pretrain_s = config_base.s
        if isinstance(pretrain_s, (list, tuple)):
            pretrain_patch_size = [8, 8, 8]
        elif pretrain_s == 64:
            pretrain_patch_size = 4
        else:
            pretrain_patch_size = 8
        init_s = pretrain_s
        init_patch_size = pretrain_patch_size
    else:
        init_s = target_s
        init_patch_size = patch_size

    model = initialize_FT3d(model = model,
                            new_in_dim = config_ft.in_dim,
                            new_out_dim = config_ft.out_dim,
                            new_s = init_s,
                            new_patch_size = init_patch_size,
                            dims = config_arch["dims"],
                            latent_channels = config_arch["latent_channels"],
                            init_new = init_new_load)

    if is_post_trained or is_3d_scratch:
        ckpt = load_checkpoint_compat(base_model_path, map_location='cpu')
        load_state_dict_shape_compat(model, ckpt['state_dict'], strict=False)
        model.save_hyperparameters(ignore=['model'])

    # Second pass: re-resize patch_embed / depatchify to the target geometry,
    # carrying the pretrained weights through by spatial adaptive_avg_pool3d.
    if interpolate_patch and (init_s != target_s or init_patch_size != patch_size):
        model = initialize_FT3d(model = model,
                                new_in_dim = config_ft.in_dim,
                                new_out_dim = config_ft.out_dim,
                                new_s = target_s,
                                new_patch_size = patch_size,
                                dims = config_arch["dims"],
                                latent_channels = config_arch["latent_channels"],
                                init_new = init_new_load,
                                interpolate_patch_weights = True)

    if not init_new_load and init_new:
        model = initialize_FT3d(model = model, 
                            new_in_dim = config_ft.in_dim, 
                            new_out_dim = config_ft.out_dim, 
                            new_s = target_s,
                            new_patch_size = patch_size,
                            dims = config_arch["dims"],
                            latent_channels = config_arch["latent_channels"],
                            init_new = init_new)


    if model.is_ft:
        model.configure_ft_encoder_decoder_warmup()
    
    model.model.train()

    is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_rank0:
        wandb.require("service")
        run = wandb.init(entity="bogdanraonic", 
                  project="foundation-model", 
                  name= tag_ft + f"_3d_" + str(config_train["N_train"]) + "_" + which_data + "_" + type_ft + "_" + config_base.which_data, 
                  group=which_data,
                  tags = [tag_ft, config_base.which_data, type_ft],
                  config=config_ft)
    else:
        os.environ["WANDB_MODE"] = "disabled"  # no network/spawn on workers
    

    if "mix" in which_data or "merra" in which_data:
        check_interval = 0.1
        limit_val_batches = 0.25
    elif "tg" in which_data:
        check_interval = 0.2
        limit_val_batches = 0.2
    elif "atm" in which_data:
        if "moist" in which_data:
            check_interval = 0.2
        else:
            check_interval = 0.05
        limit_val_batches = 1.0 #0.5
    else:
        check_interval = 1.0
        limit_val_batches = 0.5
    

    check_val_every_n_epoch = 1
    if "poisson" in which_data or "helmholtz" in which_data or "airfoil" in which_data:
        check_val_every_n_epoch = 30


    checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model", monitor='val_loss', save_top_k=3)
    logger = TensorBoardLogger(save_dir=workdir, version=123, name="logs")

    nonreentrant = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    
    if config_ft.is_precision_16:

        apply_activation_checkpointing(
        model.model.transformers,
        checkpoint_wrapper_fn=nonreentrant,
        check_fn=is_checkpointable_mlp)

        precision = 16
        strategy = DDPStrategy(find_unused_parameters=False)
        limit_val_batches = 1.0
        model.lift_project_first = True
        model.model.transformers = torch.compile(model.model.transformers)

    else:
        apply_activation_checkpointing(
        model.model.transformers,
        checkpoint_wrapper_fn=nonreentrant,
        check_fn=is_checkpointable
        )
        precision = "bf16-mixed"
        strategy = DDPStrategy(
                process_group_backend="nccl",
                find_unused_parameters=False,
                static_graph=False,            
                gradient_as_bucket_view=True,
                bucket_cap_mb=25,
            )
    
    trainer = Trainer(accelerator='gpu',
                    devices = -1,
                    precision=precision, 
                    accumulate_grad_batches=config_ft.accumulate_grad,
                    max_epochs = config_ft.epochs,
                    callbacks = [checkpoint_callback],
                    strategy = strategy,
                    val_check_interval = check_interval,
                    num_sanity_val_steps = 0,
                    check_val_every_n_epoch = check_val_every_n_epoch,
                    limit_val_batches = limit_val_batches,
                    log_every_n_steps=30,
                    logger = logger)       

    trainer.fit(model)    


