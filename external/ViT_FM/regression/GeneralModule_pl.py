import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from regression.lr_schedulers import CosineLinearWarmupCustomScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer

import wandb
import time
import math
import gc, os, psutil
from functools import partial

import matplotlib.pyplot as plt

from visualization.plot import plot_prediction
from utils.utils_data import get_loader
import torch.nn.functional as F

# Define GeneralModel_pl in pytorch_lightning framework.

class GeneralModel_pl(pl.LightningModule):
    def __init__(self,  
                in_dim, 
                out_dim,
                config_train: dict = dict()
                ):
        super(GeneralModel_pl, self).__init__()

        '''
            -- For example, in the child class, there must me smth like this:
            self.model = FNO2d(in_dim = in_dim, 
                             out_dim = out_dim,
                             n_layers = n_layers,
                             width = width,
                             modes = modes,
                             hidden_dim = hidden_dim,
                             use_conv = use_conv,
                             conv_filters = conv_filters,
                             padding = padding,
                             include_grid = include_grid,
                             is_time = is_time)
        '''

        self.model = None # TO BE DEFINED IN THE CHILD CLASS !!!
        self.loss_fn = None
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        #--------------------
        # Training parameters
        #--------------------
        
        self.peak_lr = config_train["peak_lr"]
        self.end_lr = config_train["end_lr"]
        self.warmup_epochs = config_train["warmup_epochs"]
        self.epochs = config_train["epochs"]
        self.batch_size = config_train["batch_size"]
        self.N_train = config_train["N_train"]

        self.is_time = config_train["is_time"]
        self.is_masked = config_train["is_masked"]
        self.max_num_time_steps = config_train["max_num_time_steps"]
        self.time_step_size = config_train["time_step_size"]
        self.fix_input_to_time_step = config_train["fix_input_to_time_step"]
        self.allowed_transitions = config_train["allowed_transitions"]
        self.num_workers = config_train.get("num_workers", 2)
        # Only consumed by the n32t50 windowed all-to-all loaders; defaults to
        # True to preserve the existing 50-random-pairs-per-trajectory behavior.
        self.random_train_sampling = config_train.get("random_train_sampling", True)

        self._curr_epoch = -1
        self._cur_step   = 0
        self._wandb_aggregation = 32
        self._plot_epoch = 30
        self.best_val_loss = 1000.0

        self._which_benchmark = config_train["which_data"]
        self._res = config_train["s"]
        self._workdir = config_train["workdir"]

        if "ns" in self._which_benchmark or "eul" in self._which_benchmark:
            self.interval = "step"
            _num_gpus = torch.cuda.device_count()//4+1
            self.lr_step_per_epoch = math.floor(self.N_train * self.max_num_time_steps / (self.batch_size * _num_gpus)) + 1
        else:
            self.interval = "epoch"
            self.lr_step_per_epoch = 1

        """ 
          - If we traing the model to predict different physical quantities (velocity + pressure + ...)
          - For example, if the variables are [rho, vx, vy, p], then "separate_dim" should be [1,2,1]
          - 2 means that vx and vy are grouped together!
        """
        # Are the physical quantities separated in the loss function?

        '''
        if  ("separate" in self.config) and self.config["separate"]:
            self._is_separate = True
            if "separate_dim" in self.config:
                self._separate_dim = self.config["separate_dim"]
            else:
                self._separate_dim = [out_dim]
        else:
            self._is_separate = False
        '''
         
        self._spatial_mask = False
        self.is_ft = False
        self.ar_train = False
        self.lift_project_first = False
        
        self.ft_encoder_decoder_warmup_steps = int(
            config_train.get(
                "ft_encoder_decoder_warmup_steps",
                config_train.get("ft_warmup_steps", 0),
            ) or 0
        )
        self.ft_warmup_trainable_patterns = tuple(
            config_train.get(
                "ft_warmup_trainable_patterns",
                ("model.lift", "model.project"),
            )
        )
        self._ft_encoder_decoder_warmup_enabled = False
        self._ft_encoder_decoder_warmup_optimizer_steps = 0
        self._ft_encoder_decoder_warmup_last_global_step = 0

        if "is_wandb" in config_train:
            self.is_wandb = config_train["is_wandb"]
            print("NO WANDB")
        else:
            self.is_wandb = True



        '''
        # Are we interested in all the channels or we want to predict just a few of them and ignore others?
        self._is_masked = "is_masked" in self.config and self.config["is_masked"] is not None

        # Is there a spatial mask, like in the airfoil benchmark?
        self._spatial_mask = "spatial_mask" in self.config and self.config["spatial_mask"] is not None and self.config["spatial_mask"]
        self._spatial_mask = self._spatial_mask or self._which_benchmark == "airfoil"
        '''


    def _is_ft_warmup_trainable_parameter(self, name):
        return any(pattern in name for pattern in self.ft_warmup_trainable_patterns)

    def count_trainable_parameters(self):
        total_params = 0
        trainable_params = 0
        for param in self.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, total_params

    def print_trainable_parameters(self, phase):
        trainable_params, total_params = self.count_trainable_parameters()
        percent_trainable = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"Fine-tuning trainable parameters ({phase}): "
                f"{trainable_params:,}/{total_params:,} "
                f"({percent_trainable:.2f}% trainable)"
            )

        return trainable_params, total_params

    def _set_ft_encoder_decoder_warmup_requires_grad(self, warmup_only, phase):
        for name, param in self.named_parameters():
            param.requires_grad = (not warmup_only) or self._is_ft_warmup_trainable_parameter(name)

        trainable_params, total_params = self.print_trainable_parameters(phase)

        if warmup_only and trainable_params == 0:
            raise ValueError(
                "ft_encoder_decoder_warmup_steps was set, but no parameters matched "
                f"ft_warmup_trainable_patterns={self.ft_warmup_trainable_patterns}."
            )

        return trainable_params, total_params

    def configure_ft_encoder_decoder_warmup(self):
        if self.ft_encoder_decoder_warmup_steps <= 0:
            self._ft_encoder_decoder_warmup_enabled = False
            self._set_ft_encoder_decoder_warmup_requires_grad(
                warmup_only=False,
                phase="fine-tuning start (no encoder/decoder warmup)",
            )
            return

        self._ft_encoder_decoder_warmup_enabled = True
        self._ft_encoder_decoder_warmup_optimizer_steps = 0
        self._ft_encoder_decoder_warmup_last_global_step = 0
        self._set_ft_encoder_decoder_warmup_requires_grad(
            warmup_only=True,
            phase="before encoder/decoder warmup starts",
        )

    def on_fit_start(self):
        # This is a safety net for callers that set ft_encoder_decoder_warmup_steps
        # but forget to call configure_ft_encoder_decoder_warmup() before Trainer.fit.
        if self.ft_encoder_decoder_warmup_steps > 0 and not self._ft_encoder_decoder_warmup_enabled:
            self.configure_ft_encoder_decoder_warmup()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self._ft_encoder_decoder_warmup_enabled:
            return

        current_global_step = int(self.global_step)
        if current_global_step <= self._ft_encoder_decoder_warmup_last_global_step:
            return

        self._ft_encoder_decoder_warmup_optimizer_steps += (
            current_global_step - self._ft_encoder_decoder_warmup_last_global_step
        )
        self._ft_encoder_decoder_warmup_last_global_step = current_global_step

        if self._ft_encoder_decoder_warmup_optimizer_steps >= self.ft_encoder_decoder_warmup_steps:
            self._ft_encoder_decoder_warmup_enabled = False
            self._set_ft_encoder_decoder_warmup_requires_grad(
                warmup_only=False,
                phase="after encoder/decoder warmup finishes",
            )
    
    def forward(self, x, time = None, rollout_steps = None):
        
        if rollout_steps is None:
            return self.model(x, time)
        else:
            '''
            assert isinstance(rollout_steps, int)
            out = x
            for k in range(rollout_steps):
                out = self.model(out, time)
            return out
            '''
            assert rollout_steps.dim() == 1, "rollout_steps must be [B]"
            B = x.shape[0]
            max_steps = int(rollout_steps.max().item())

            out = x
            for k in range(max_steps):
                # forward one step
                out_next = self.model(out, time)

                # mask: only update entries where rollout_steps > k
                
                mask = (rollout_steps > k).view(B, *([1] * (out.ndim - 1)))
                out = torch.where(mask, out_next, out)

            return out

    # --- Group 2 (film): cosine warmup 6–10, then decay ---
    @staticmethod
    def lr_lambda_film(epoch, e_max = 4):
        
        if epoch < e_max:
            return 1e-3    # tiny lr in warmup freeze
        elif epoch < 2*e_max:
            # cosine ramp from 0 -> 1 over 5 epochs
            progress = (epoch-1.*e_max)/e_max
            return 0.5 * (1 - math.cos(math.pi * progress))
        else:
            return 1.0

    # --- Group 3 (backbone): same schedule as film ---
    @staticmethod
    def lr_lambda_bb(epoch, e_max = 4):
        e_max = 4
        if epoch < e_max:
            return 1e-3
        elif epoch < 2*e_max:
            progress = (epoch - 1.*e_max)/e_max
            return 0.5 * (1 - math.cos(math.pi * progress))
        else:
            return 1.0

    def configure_optimizers(self):

        if not self.is_ft:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.end_lr)
            scheduler = CosineLinearWarmupCustomScheduler(optimizer, 
                                                        warmup_epochs = self.lr_step_per_epoch*self.warmup_epochs, 
                                                        total_epochs = self.lr_step_per_epoch*self.epochs, 
                                                        peak_lr = self.peak_lr, 
                                                        end_lr = self.end_lr)

            return [optimizer], [{"scheduler": scheduler, "interval": self.interval}]
        else:
            
            params_1 = [param for name, param in self.named_parameters() if (("project" in name) or ("lift" in name) or ("pe_proj" in name))]
            params_2 = [param for name, param in self.named_parameters() if ((("project" not in name) and ("lift" not in name) and ("pe_proj" not in name)) and (("film" in name) or ("adapt" in name)))]
            params_3 = [param for name, param in self.named_parameters() if ("project" not in name) and ("lift" not in name) and ("pe_proj" not in name) and ("film" not in name) and ("adapt" not in name)]
            
            lr_liftprojecttime = self.peak_lr
            lr_bb = self.end_lr
            lr_film = self.peak_lr/2.
            
            print(" ")
            print("----------")  
            print("FINETUNNING - lift/project", lr_liftprojecttime, "   OTHER: ", lr_bb) 
            print("----------")  
            print(" ")
            
            optimizer = torch.optim.AdamW([ {'params': params_1, 'lr': lr_liftprojecttime},  # lift/project
                                            {'params': params_2, 'lr': lr_film},             # film/adapt
                                            {'params': params_3, 'lr': lr_bb},])
                
            if not self.lift_project_first:
                
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
            else:

                ep_warmup = 2
                scheduler_lift = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs-ep_warmup, eta_min=1e-6)
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda e: 1.0, partial(self.lr_lambda_film, e_max = ep_warmup), partial(self.lr_lambda_bb, e_max = ep_warmup)])
                decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs-2*ep_warmup, eta_min=1e-6)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[2*ep_warmup])

            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def training_step(self, batch, batch_idx):
        
        optimizer = self.trainer.optimizers[0]  # if you have one optimizer
        #current_lr = 
        #print(f"Step {batch_idx}, LR: {optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr']}")
        if self.is_masked is not None:
            if not self.ar_train:
                t_batch, input_batch, output_batch, masked_dim = batch
                rollout_steps = None
            else:
                t_batch, rollout_steps, input_batch, output_batch, masked_dim = batch
                #rollout_steps = int(rollout_steps[0])
        else:
            if self.is_time:
                if not self.ar_train:
                    t_batch, input_batch, output_batch = batch
                    rollout_steps = None
                else:
                    t_batch, rollout_steps, input_batch, output_batch = batch
                    rollout_steps = int(rollout_steps[0])
            else:
                input_batch, output_batch = batch
                t_batch = None
                rollout_steps = None

            masked_dim = None
        
        if t_batch is not None:
            t_batch = t_batch.type(torch.float32)

        # Predict:
        output_pred_batch = self(input_batch, t_batch, rollout_steps = rollout_steps)
        
        # If spatial mask, as in airfoil, mask it
        if self._spatial_mask:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0

        loss = self.loss_fn(output_batch, 
                            output_pred_batch,
                            reduction = True,
                            mask = masked_dim)

        '''
            wandb logs:
        '''
        if batch_idx == 0:
            self._curr_epoch+=1
            self.avg_loss = float(loss.detach().cpu().item()) * output_batch.shape[0]
            self.num_train_items = output_batch.shape[0]
        else:
            self.avg_loss += float(loss.detach().cpu().item()) * output_batch.shape[0]
            self.num_train_items += output_batch.shape[0]

        if self._cur_step % self._wandb_aggregation == 0:
            self.log('loss', loss, prog_bar=True)
            dict_log = {'train/loss': float(loss.detach().cpu().item()), 'train/step': self._cur_step, 'train/epoch': self._curr_epoch}
            
            if batch_idx >= 10:
                loss_log = self.avg_loss/self.num_train_items
                dict_log['train/loss_avg'] =  loss_log
            if self.global_rank ==0 and self.is_wandb:
                wandb.log(dict_log, step=self._cur_step)

            self.avg_loss = 0.0
            self.num_train_items = 0
        self._cur_step+=1
        
        if batch_idx % 1000 == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        
        return loss
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    def validation_step(self, batch, batch_idx):
        
        if self.is_masked is not None:
            if not self.ar_train:
                t_batch, input_batch, output_batch, masked_dim = batch
                rollout_steps = None
            else:
                t_batch, rollout_steps, input_batch, output_batch, masked_dim = batch
        else:
            if self.is_time:
                if not self.ar_train:
                    t_batch, input_batch, output_batch = batch
                    rollout_steps = None
                else:
                    t_batch, rollout_steps, input_batch, output_batch = batch
                    rollout_steps = int(rollout_steps[0])
            else:
                input_batch, output_batch = batch
                t_batch = None
                rollout_steps = None

            masked_dim = None
            
        if t_batch is not None:
            t_batch = t_batch.type(torch.float32)

        # Predict:
        output_pred_batch = self(input_batch, t_batch, rollout_steps = rollout_steps)
        
        if batch_idx==0 and self._curr_epoch%self._plot_epoch == 0 and self._curr_epoch>=self._plot_epoch and self.model is not None:

            if self.global_rank == 0 and self._curr_epoch%50 == 0:
                batch_size_plot = min(20, input_batch.shape[0])
                fig = plot_prediction(batch_size_plot, (1,1), input_batch[:batch_size_plot], output_batch[:batch_size_plot], output_pred_batch[:batch_size_plot], f"{self._workdir}/train_plot_ep_{self._curr_epoch}.png")
                if self.is_wandb:
                    wandb.log({f"fig_train/train_plot_ep_{self._curr_epoch+1}": wandb.Image(fig)})
                plt.close()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # If spatial mask, as in airfoil, mask it
        if self._spatial_mask:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0

        ########output_pred_batch = input_batch
        loss = self.loss_fn(output_batch, 
                            output_pred_batch,
                            reduction = False,
                            mask = masked_dim)
        
        # Save validation errs:

        if batch_idx==0:
            self.validation_times = t_batch
            self.validation_errs = loss

            '''
            if self._curr_epoch%self._plot_epoch == 0 and self.model is not None:
                batch_size_plot = 8
                fig = plot_prediction(batch_size_plot, (1,1), input_batch[:batch_size_plot], output_batch[:batch_size_plot], output_pred_batch[:batch_size_plot], f"{self._workdir}/train_plot_ep_{self._curr_epoch}.png")
                wandb.log({f"fig_train/train_plot_ep_{self._curr_epoch+1}": wandb.Image(fig)})
                plt.close()
                #self.best_model_ema.to("cpu")
                torch.cuda.empty_cache()
            '''
        else:
            self.validation_times = torch.cat((self.validation_times, t_batch))
            self.validation_errs = torch.cat((self.validation_errs, loss))
        
        return loss

    def on_validation_epoch_end(self):

        _max_time = torch.max(self.validation_times)
        _validation_errs_last = self.validation_errs[self.validation_times==_max_time]

        _min_time = torch.min(self.validation_times)
        _validation_errs_first = self.validation_errs[self.validation_times==_min_time]

        median_loss = torch.median(self.validation_errs).item()
        mean_loss = torch.mean(self.validation_errs).item() 
        median_loss_last = torch.median(_validation_errs_last).item()
        mean_loss_last = torch.mean(_validation_errs_last).item() 
        median_loss_first = torch.median(_validation_errs_first).item()
        mean_loss_first = torch.mean(_validation_errs_first).item() 
        
        self.log("median_val_loss", float(median_loss), prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_loss",  float(mean_loss), prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        # Save the best loss
        if mean_loss < self.best_val_loss:
            self.best_val_loss = mean_loss
        
        self.log("best_val_loss",float(self.best_val_loss),on_step=False, on_epoch=True,sync_dist=True)
        
        if self.global_rank == 0 and self.is_wandb:
            wandb.log({'val/best_val_loss': float(self.best_val_loss), 'val/mean_val_all': float(mean_loss), 'val/med_val_all': float(median_loss), 'val/med_val_last': float(median_loss_last), 'val/mean_val_last': float(mean_loss_last), 'val/mean_val_first': float(mean_loss_first), 'val/median_val_first': float(median_loss_first)}, step=self._cur_step)
        
        return {"val_loss": mean_loss,} 

    def train_dataloader(self):
        
        _rel_time = True

        train_loader = get_loader(which_data = self._which_benchmark,
                                which_type = "train",
                                in_dim = self.in_dim,
                                out_dim = self.out_dim,
                                N_samples = self.N_train,
                                batch_size = self.batch_size,
                                masked_input = self.is_masked,
                                is_time = self.is_time,
                                max_num_time_steps = self.max_num_time_steps,
                                time_step_size = self.time_step_size,
                                fix_input_to_time_step = self.fix_input_to_time_step,
                                allowed_transitions = self.allowed_transitions,
                                rel_time = _rel_time,
                                ar_train = self.ar_train,
                                num_workers = self.num_workers,
                                random_train_sampling = self.random_train_sampling)
        return train_loader

    def val_dataloader(self):

        _rel_time = True

        val_loader = get_loader(which_data = self._which_benchmark,
                            which_type = "val",
                            in_dim = self.in_dim,
                            out_dim = self.out_dim,
                            N_samples = 1,
                            batch_size = self.batch_size,
                            masked_input = self.is_masked,
                            is_time = self.is_time,
                            max_num_time_steps = self.max_num_time_steps,
                            time_step_size = self.time_step_size,
                            fix_input_to_time_step = self.fix_input_to_time_step,
                            allowed_transitions = self.allowed_transitions,
                            rel_time = _rel_time,
                            ar_train = self.ar_train,
                            num_workers = self.num_workers,
                            random_train_sampling = self.random_train_sampling)

        return val_loader
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        def move(x):
            return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        if len(batch) == 5:
            t, steps, inp, lbl, m = batch
            return move(t), move(steps), move(inp), move(lbl), move(m)
        elif len(batch) == 4:
            t, inp, lbl, m = batch
            return move(t), move(inp), move(lbl), move(m)
        else:
            t, inp, lbl = batch
            return move(t), move(inp), move(lbl)