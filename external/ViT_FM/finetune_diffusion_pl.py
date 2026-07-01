import torch
import os
import functools
import tqdm
import argparse
import copy
import wandb
import matplotlib.pyplot as plt

from diffusion.loss_fn import loss_fn, loss_fn_denoised
from diffusion.variance_fn import marginal_prob_std_1, diffusion_coeff_1, marginal_prob_std_2, diffusion_coeff_2
import time 

from GenCFD.model.lightning_wrap.pl_conditional_denoiser import PreconditionedDenoiser_pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils.utils_data import get_loader, save_data, load_data, read_cli_regression, save_errors, read_config, select_variable_condition, find_files_with_extension
from torch.optim.swa_utils import AveragedModel

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/ood_wandb_logs"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="load parameters for training")
    params = read_config(parser).parse_args()    
    config_ft = argparse.Namespace(**load_data(params.config))

    workdir_base = config_ft.config_diffusion
    base_model_path = str(find_files_with_extension(workdir_base + "/model", "ckpt", [], is_pl = True)[0])
    base_config_path = str(find_files_with_extension(workdir_base, "json", ["param"])[0])
    config_base = argparse.Namespace(**load_data(base_config_path))

    device = config_ft.device
    tag_ft = config_ft.tag
    
    is_log_uniform = config_base.is_log_uniform
    log_uniform_frac = config_base.log_uniform_frac
    is_exploding = config_base.is_exploding
    sigma =  config_base.sigma
    ema_param = config_base.ema_param

    if is_exploding:
        marginal_prob_std_fn = functools.partial(marginal_prob_std_2, sigma_min = 0.001, sigma_max=sigma, device = device)
        diffusion_coeff_fn = functools.partial(diffusion_coeff_2, sigma_min = 0.001, sigma_max=sigma, device = device)
    else:
        marginal_prob_std_fn = functools.partial(marginal_prob_std_1, sigma=sigma, device = device)
        diffusion_coeff_fn = functools.partial(diffusion_coeff_1, sigma=sigma, device = device)

    which_data = config_ft.which_data
    which_type = config_base.which_type

    workdir = f"{workdir_base}/finetuned/{which_data}/{tag_ft}_FT_{which_data}_diffusion_gencfd"
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    print(workdir)
    config_arch = load_data(config_base.config_arch)
    config_train = vars(config_ft)
    config_train["workdir"] = workdir
    config_train["which_type"] = config_base.which_type
    config_train["s"] = config_base.s
    config_train["ema_param"] = config_base.ema_param
    config_train["skip"] = config_base.skip
    
    is_time_diff = config_ft.is_time
    if which_type == "xy":
        dim = config_ft.in_dim
        dim_cond = config_ft.out_dim 
    elif which_type == "yx":
        dim = config_ft.out_dim
        dim_cond = config_ft.in_dim
    elif which_type == "x":
        dim = config_ft.in_dim
        dim_cond = 0
    elif which_type == "y":
        dim = config_ft.out_dim
        dim_cond = 0
    elif which_type == "x&y":
        dim = config_ft.out_dim + config_ft.in_dim
        dim_cond = 0

    loss = functools.partial(loss_fn_denoised, is_log_uniform = is_log_uniform, log_uniform_frac = log_uniform_frac, weighting = "edm", sigma_data = 0.5, consistent_weight = 0.1, channel_weight = None)

    model = PreconditionedDenoiser_pl(dim = dim, 
                                    dim_cond = dim_cond,
                                    loss_fn = loss,
                                    marginal_prob_std_fn = marginal_prob_std_fn,
                                    diffusion_coeff_fn = diffusion_coeff_fn,
                                    config_train = config_train,
                                    config_arch = config_arch,
                                    is_inference = True
                                    )
    
    checkpoint = torch.load(base_model_path, map_location = device)
    model.load_state_dict(checkpoint["state_dict"])
    model.best_model_ema = None

    model.ema_model = AveragedModel(model.model, avg_fn=model.ema_update)

    for param in model.ema_model.parameters():
        param.requires_grad = False  # Prevent updates from optimizer

    for param in model.model.parameters():
        param.requires_grad = True

    run = wandb.init(entity="bogdanraonic", 
                  project=config_base.wandb_project_name, 
                  name=tag_ft + "_diff_" + which_data + config_base.wandb_run_name, 
                  config=config_ft)

    config_ft["config_arch"] = config_arch
    config_ft["is_log_uniform"] = config_base.is_log_uniform
    config_ft["log_uniform_frac"] = config_base.log_uniform_frac
    config_ft["is_exploding"] = config_base.is_exploding
    config_ft["sigma"] =  config_base.sigma
    config_ft["ema_param"] = config_base.ema_param

    save_data(vars(config_ft), f"{workdir}/param_diffusion_gencfd_{tag_ft}.json")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model", monitor='val_loss_ema', save_top_k=2)
    logger = TensorBoardLogger(save_dir=workdir, version=1, name="logs")

    check_interval = 1.0

    trainer = Trainer(devices = -1,
                    max_epochs = config_ft.epochs,
                    callbacks = [checkpoint_callback],
                    logger=logger,
                    val_check_interval = check_interval,
                    num_sanity_val_steps = 0)
    trainer.fit(model)
    trainer.validate(model)
    