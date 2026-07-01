import torch
import os
import functools
import tqdm
import argparse
import copy
import wandb
import ast

import matplotlib.pyplot as plt

import time 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from utils.utils_data import get_loader, save_data, load_data, read_cli_finetune, read_cli_regression, save_errors, read_config, select_variable_condition, find_files_with_extension
from regression.loss_fn import relative_lp_loss_fn, relative_lp_loss_separate_fn

from utils.utils_finetune import initialize_FT
from regression.ViTModulev2 import MultiVit3_pl, MultiVit2_pl, Vit3_pl

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/ood_wandb_logs"

if __name__ == "__main__":

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

    if "ar_train" in config_train:
        ar_train = config_train["ar_train"]
    else:
        ar_train = False
        
    #if ar_train:
    #    assert len(config_train["allowed_transitions"]) == 1

    _arch_to_load = config_base.config_arch

    if isinstance(_arch_to_load, dict):
        config_arch = _arch_to_load
        already_ft = True
    else:
        config_arch = load_data(config_base.config_arch)
        already_ft = False
    
    config_train["config_arch"] = config_arch
    config_train["fix_input_to_time_step"] = None

    workdir = f"{workdir_base}/finetuned/{which_data}/{tag_ft}_FT_{which_data}_{type_ft}_" + str(config_train["N_train"])
    config_train["workdir"] = workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    save_data(vars(config_ft), f"{workdir}/param_regression_{tag_ft}" + str(config_train["N_train"]) + ".json")
    
    err_group = config_ft.err_group
    err_mask_group = config_ft.err_mask_group
    groups = [0]

    num_unmasked = 0
    for i,g in enumerate(err_group):
        groups.append(groups[-1]+g)
        if err_mask_group[i]>0:
            num_unmasked+=1
    num_groups = len(err_group)

    if config_ft.loss_type == "rel":
        loss = functools.partial(relative_lp_loss_fn, p=p)
    elif config_ft.loss_type == "rel_g":
        loss = functools.partial(relative_lp_loss_separate_fn, p=p, separate_dim = groups)

    if "pdegym/" in workdir_base:
        dim0 = 4
        dim1 = 4
    elif "ns_mix" in workdir_base:
        dim0 = 2
        dim1 = 2
    elif "pdegym_" in workdir_base:
        dim0 = 9
        dim1 = 9

    if not config_ft.is_masked:
        config_train["is_masked"] = None 
    model = Vit3_pl(in_dim = dim0, 
                    out_dim = dim1,
                    loss_fn = loss,
                    config_train = config_train,
                    config_arch = config_arch)

    
    checkpoint = torch.load(base_model_path, map_location = device)

    if not already_ft:
        model.load_state_dict(checkpoint["state_dict"])

    model.is_ft = True
    if reinit_ft:
        model = initialize_FT(model, config_ft.in_dim, config_ft.out_dim, latent_channels = config_arch["latent_channels"], init_new = init_new, ar_train = ar_train)
    
    if already_ft:
        model.load_state_dict(checkpoint["state_dict"])
    
    if model.is_ft:
        model.configure_ft_encoder_decoder_warmup()
    
    model.model = model.model.to(device).train()
    
    if ar_train:
        print(model.model.time_scale, model.model.time_shift)
    #time.sleep(100)
    """    
    optimizer_state = checkpoint["optimizer_states"]
    print(optimizer_state)
    time.sleep(100)"""

    
    if "is_wandb" not in config_train or ("is_wandb" in config_train and config_train["is_wandb"]):
        run = wandb.init(entity="bogdanraonic", 
                    project="foundation-model", 
                    name= tag_ft + f"_" + str(config_train["N_train"]) + "_" + which_data + "_" + type_ft + "_" + config_base.which_data, 
                    group=which_data,
                    tags = [tag_ft, config_base.which_data, type_ft],
                    config=config_ft)

    if "mix" in which_data or "merra" in which_data:
        check_interval = 0.1
    else:
        check_interval = 1.0
    
    limit_val_batches = 1.0
    if config_train["N_train"] <= 16:
        check_val_every_n_epoch = 3
    else: 
        check_val_every_n_epoch = 5

    if "poisson" in which_data or "helmholtz" in which_data or "airfoil" in which_data:
        check_val_every_n_epoch = 30

    if "long" in which_data:
        check_val_every_n_epoch = 1
        limit_val_batches = 0.5
        check_interval = 1.0
    
    if "merra" in which_data:
        check_val_every_n_epoch = 1
        limit_val_batches = 1.0
        check_interval = 1.0

    checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model", monitor='val_loss', save_top_k=3)
    logger = TensorBoardLogger(save_dir=workdir, version=123, name="logs")
    trainer = Trainer(devices = -1,
                    max_epochs = config_ft.epochs,
                    callbacks = [checkpoint_callback],
                    strategy = DDPStrategy(find_unused_parameters=False), #IMPORTANT!!!
                    val_check_interval = check_interval,
                    num_sanity_val_steps = 0,
                    check_val_every_n_epoch = check_val_every_n_epoch,
                    limit_val_batches = limit_val_batches)
    trainer.fit(model)
    trainer.validate(model)
    


