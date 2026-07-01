import torch
import os
import functools
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
import argparse

from regression.ViTModulev2 import MultiVit3_pl, MultiVit2_pl, Vit3_pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from utils.utils_data import get_loader, save_data, load_data, read_cli_regression, save_errors
import time 
import json
import torch.nn as nn

from regression.loss_fn import relative_lp_loss_fn

import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/ood_wandb_logs"

if __name__ == "__main__":
  #parser = argparse.ArgumentParser(description="load parameters for training")
  #params = argparse.Namespace(**read_cli_regression(parser).parse_args())
  
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
  
  print(config_train["is_masked"], "POST")
  print(" ")
  print(" ")
  print(config.config_arch, config_arch)


  loss = functools.partial(relative_lp_loss_fn, p=p)  
  model = Vit3_pl(in_dim = config.in_dim, 
                  out_dim = config.out_dim,
                  loss_fn = loss,
                  config_train = config_train,
                  config_arch = config_arch)
    
  run = wandb.init(entity="bogdanraonic", 
                  project=config.wandb_project_name, 
                  name= tag + "_" + str(config.N_train) + "_" +config.wandb_run_name, 
                  tags = [tag, "scratch", which_data],
                  group=which_data,
                  config=config)

  limit_val_batches = 1.0
  check_val_every_n_epoch = 1
  check_interval = 1.0

  if "mix" in which_data or "merra" in which_data or "pdegym" in which_data:
    check_interval = 0.1
  
  if "long" in which_data:
    check_val_every_n_epoch = 1
    limit_val_batches = 0.5
    check_interval = 1.0

  checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model", monitor='val_loss', save_top_k=3)
  logger = TensorBoardLogger(save_dir=workdir, version=1, name="logs")

  trainer = Trainer(devices = -1,
                  max_epochs = config.epochs,
                  callbacks = [checkpoint_callback],
                  strategy=DDPStrategy(find_unused_parameters=False), #IMPORTANT!!!
                  logger=logger,
                  val_check_interval = check_interval,
                  check_val_every_n_epoch = check_val_every_n_epoch,
                    limit_val_batches = limit_val_batches,
                  num_sanity_val_steps = 0)
  trainer.fit(model)
  trainer.validate(model)