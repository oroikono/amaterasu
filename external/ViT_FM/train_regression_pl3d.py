import torch
import os
import functools
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
import argparse
import warnings
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing
import torch._dynamo

from pytorch_lightning.strategies import FSDPStrategy
from utils.utils_finetune_3d import initialize_FT3d
from regression.ViTModulev2 import Vit3_pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from utils.utils_data import get_loader, save_data, load_data, read_cli_regression, save_errors
import time 
import json
import torch.nn as nn

from regression.loss_fn import relative_lp_loss_fn_3d
from regression.ViTModulev2 import Attention, FeedForward  # adjust if namespaced differently

import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/ood_wandb_logs"

def is_checkpointable(module: torch.nn.Module) -> bool:
    return isinstance(module, (Attention, FeedForward))

if __name__ == "__main__":

  torch._dynamo.config.optimize_ddp = False  # disable broken DDP graph fusion
  torch._dynamo.config.suppress_errors = True

  warnings.filterwarnings(
  "ignore",
  message=".*None of the inputs have requires_grad=True.*",
  module="torch.utils.checkpoint"
  )

  parser = argparse.ArgumentParser(description="load parameters for training")
  params = read_cli_regression(parser).parse_args()

  if params.config is None:
    config = params
  else:
    config = argparse.Namespace(**load_data(params.config))

  device = config.device
  tag = config.tag
  p = int(config.loss)
  which_data = config.which_data
  
  if hasattr(config, "workdir") and config.workdir is not None:
    workdir = f"{config.workdir}/{tag}_{config.N_train}"
  else:
    workdir = f"/cluster/work/math/braonic/TrainedModels/OOD_Generalization/{which_data}/{tag}_regression"
  
  if not os.path.exists(workdir):
    os.makedirs(workdir)
  
  save_data(vars(config), f"{workdir}/param_regression_{tag}.json")

  config_arch = load_data(config.config_arch)
  config_train = vars(config)
  config_train["workdir"] = workdir
  print(config_train["is_masked"], "PRE")
  if "is_masked" not in config_train or not config_train["is_masked"]:
    config_train["is_masked"] = None
  if "fix_input_to_time_step" not in config_train:
    config_train["fix_input_to_time_step"] = None

  print(config.config_arch, config_arch)


  loss = functools.partial(relative_lp_loss_fn_3d, p=p)
  model = Vit3_pl(in_dim = config.in_dim, 
                  out_dim = config.out_dim,
                  loss_fn = loss,
                  config_train = config_train,
                  config_arch = config_arch)
  
  if config.s == 64:
    patch_size = 4
  else:
    patch_size = 8
    
  model = initialize_FT3d(model = model, 
                          new_in_dim = config.in_dim, 
                          new_out_dim = config.out_dim, 
                          new_s = config.s,
                          new_patch_size = patch_size,
                          dims = config_arch["dims"],
                          latent_channels = config_arch["latent_channels"])
  is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0
  if is_rank0:
      wandb.require("service")
      run = wandb.init(entity="bogdanraonic", 
                  project=config.wandb_project_name, 
                  name= tag + "_" + str(config.N_train) + "_" +config.wandb_run_name, 
                  tags = [tag, "scratch", which_data],
                  group=which_data,
                  config=config)
  else:
        os.environ["WANDB_MODE"] = "disabled"  # no network/spawn on workers
  
  model.save_hyperparameters(ignore=['model'])

  if hasattr(config, "accumulate_grad"):
    accumulate_grad = config.accumulate_grad
  else:
    accumulate_grad = 3

  if "mix" in which_data or "merra" in which_data:
    check_interval = 0.1
    limit_val_batches = 0.25
  else:
    check_interval = 1.0
    limit_val_batches = 0.5
  

  check_val_every_n_epoch = 1
  if "poisson" in which_data or "helmholtz" in which_data or "airfoil" in which_data:
      check_val_every_n_epoch = 30

  
  checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model", monitor='val_loss', save_top_k=3)
  logger = TensorBoardLogger(save_dir=workdir, version=1, name="logs")
  
  nonreentrant = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
  apply_activation_checkpointing(
      model.model.transformers,
      checkpoint_wrapper_fn=nonreentrant,
      check_fn=is_checkpointable
  )

  strategy = DDPStrategy(
                process_group_backend="nccl",
                find_unused_parameters=False,
                static_graph=False,            
                gradient_as_bucket_view=True,
                bucket_cap_mb=25,
            )

  trainer = Trainer(accelerator='gpu',
                  devices = -1,
                  max_epochs = config.epochs,
                  precision="bf16-mixed",                # for float16 (on CUDA)
                  accumulate_grad_batches= accumulate_grad,
                  callbacks = [checkpoint_callback],
                  strategy= strategy,
                  val_check_interval = check_interval,
                  num_sanity_val_steps = 0,
                  check_val_every_n_epoch = check_val_every_n_epoch,
                  limit_val_batches = limit_val_batches,
                  log_every_n_steps=30,
                  logger = logger)
  """
  gradient_clip_val = 1.0,            # try 0.5–1.0 first
  gradient_clip_algorithm="norm",   # "norm" (global-norm) or "value"
  detect_anomaly=False )       
  """
  trainer.fit(model)
  trainer.validate(model)