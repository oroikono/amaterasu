# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
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

"""Function that runs training."""

import os
from typing import Any, Sequence, Optional
from collections.abc import Mapping
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from GenCFD.utils import callbacks as cb
from GenCFD.train import trainers
from GenCFD.utils import train_utils

import wandb

def _is_finite_value(value: Any) -> bool:
    """Returns if a metric value is finite.

    Supports tensor and nested mapping structures.
    """
    if isinstance(value, torch.Tensor):
        return torch.isfinite(value).all().item()
    if isinstance(value, Mapping):
        return all(_is_finite_value(v) for v in value.values())
    return True


def _adapt_state_dict_keys_for_model(
    state_dict: dict,
    model_is_compiled: bool,
    model_is_ddp: bool,
    checkpoint_is_compiled: bool,
    checkpoint_is_ddp: bool,
) -> dict:
    """Align checkpoint key prefixes for compiled and DDP variations."""

    keyword_compiled = "_orig_mod."
    keyword_ddp = "module."

    adapted = state_dict
    if not model_is_compiled and checkpoint_is_compiled:
        adapted = {
            key.replace(keyword_compiled, ""): value for key, value in adapted.items()
        }
    if model_is_compiled and not checkpoint_is_compiled:
        adapted = {keyword_compiled + key: value for key, value in adapted.items()}
    if not model_is_ddp and checkpoint_is_ddp:
        adapted = {key.replace(keyword_ddp, ""): value for key, value in adapted.items()}
    if model_is_ddp and not checkpoint_is_ddp:
        adapted = {keyword_ddp + key: value for key, value in adapted.items()}

    return adapted


def _to_float_scalar(value: Any) -> Optional[float]:
    """Converts a scalar-like value to python float if finite, else None."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if torch.isfinite(torch.tensor(value)).item() else None


def _extract_eval_loss(metrics: Any) -> Optional[float]:
    """Extracts a checkpoint eval-loss proxy from stored eval metrics."""

    if not isinstance(metrics, Mapping) or not metrics:
        return None

    scalar_values = []
    for value in metrics.values():
        scalar = _to_float_scalar(value)
        if scalar is not None:
            scalar_values.append(scalar)

    if not scalar_values:
        return None

    return float(sum(scalar_values) / len(scalar_values))

def _get_current_learning_rate(optimizer: torch.optim.Optimizer) -> Optional[float]:
    """Returns current optimizer learning rate from the first param group."""

    if not optimizer.param_groups:
        return None

    lr = optimizer.param_groups[0].get("lr")
    return _to_float_scalar(lr)

def _find_latest_valid_checkpoint(
    checkpoint_dir: str,
    ema_decay: float = 0.9,
    spike_factor: float = 1.5,
) -> Optional[str]:
    """Finds latest valid checkpoint based on eval-loss EMA spike detection."""

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))

    ema_eval_loss = None
    latest_valid_path = None

    for ckpt_name in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        eval_loss = _extract_eval_loss(checkpoint.get("metrics", {}))
        if eval_loss is None:
            continue

        is_valid = True
        if ema_eval_loss is not None and eval_loss > spike_factor * ema_eval_loss:
            is_valid = False

        if is_valid:
            latest_valid_path = checkpoint_path

        if ema_eval_loss is None:
            ema_eval_loss = eval_loss
        else:
            ema_eval_loss = ema_decay * ema_eval_loss + (1.0 - ema_decay) * eval_loss

    return latest_valid_path


def _restore_latest_valid_checkpoint(
    trainer: trainers.BaseTrainer,
    workdir: str,
    world_size: int,
    local_rank: int,
) -> bool:
    """Restores latest valid checkpoint and optimizer state.

    Validity is determined via eval-loss moving-average spike detection.
    """

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        return False

    checkpoint_path = _find_latest_valid_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_state_dict = _adapt_state_dict_keys_for_model(
        state_dict=checkpoint["model_state_dict"],
        model_is_compiled=getattr(trainer, "is_compiled", False),
        model_is_ddp=getattr(trainer, "is_parallelized", False),
        checkpoint_is_compiled=checkpoint.get("is_compiled", False),
        checkpoint_is_ddp=checkpoint.get("is_parallelized", False),
    )

    #trainer.model.denoiser.load_state_dict(model_state_dict)
    trainer.model.denoiser.load_state_dict(model_state_dict, strict=False)
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.train_state.step = checkpoint["step"]
    
    if hasattr(trainer, "_lr_update_step"):
        trainer._lr_update_step = checkpoint.get(
            "lr_update_step", trainer.train_state.step
        )
    if hasattr(trainer, "_grad_accum_counter"):
        trainer._grad_accum_counter = checkpoint.get("grad_accum_counter", 0)
    if (
        getattr(trainer, "use_mixed_precision", False)
        and getattr(trainer, "scaler", None) is not None
        and checkpoint.get("scaler_state_dict") is not None
    ):
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if (world_size > 1 and local_rank == 0) or world_size == 1:
        print(
            "Detected non-finite training metrics. "
            f"Restored latest valid checkpoint: {checkpoint_path}"
        )

    return True

def run(
    *,
    train_dataloader: DataLoader,
    trainer: trainers.BaseTrainer,
    workdir: str,
    # DDP configs
    world_size: int,
    local_rank: int,
    # training configs
    total_train_steps: int,
    metric_aggregation_steps: int = 10,
    # evaluation configs
    eval_dataloader: Optional[DataLoader] = None,
    eval_every_steps: int = 100,
    num_batches_per_eval: int = 10,
    run_sanity_eval_batch: bool = True,
    # other configs
    metric_writer: Optional[SummaryWriter] = None,
    callbacks: Sequence[cb.Callback] = (),
) -> None:
    """Runs trainer for a training task.

    This function runs a trainer in batches of "metric aggregation" steps, where
    the step-wise metrics obtained within the same batch are aggregated
    (i.e. by computing the average and/or std based on the metric defined in
    the trainer class). The aggregated metrics are then automatically saved to a
    tensorflow event file in `workdir`. Evaluation runs periodically, i.e. once
    every `eval_every_steps` steps, if an eval dataloader is provided.

    Args:
      train_dataloader: A dataloader emitting training data in batches.
      trainer: A trainer object hosting the train and eval logic.
      workdir: The working directory where results (e.g. train & eval metrics) and
        progress (e.g. checkpoints) are saved.
      world_size: describes if model is in ddp mode trained (world_size > 1)
      total_train_steps: Total number of training steps to run.
      metric_aggregation_steps: The trainer runs this number of steps at a time,
        after which training metrics are aggregated and logged.
      eval_dataloader: An evaluation dataloader (optional). If set to `None`, no
        evaluation will run.
      eval_every_steps: The period, in number of train steps, at which evaluation
        runs. Must be an integer multiple of `metric_aggregation_steps`.
      num_batches_per_eval: The number of batches to step through every time
        evaluation is run (resulting metrics are aggregated).
      run_sanity_eval_batch: Whether to step through sanity check eval batch
        before training starts. This helps expose runtime issues early, without
        having to wait until evaluation is first triggered (i.e. after
        `eval_every_steps`).
      metric_writer: A metric writer that writes scalar metrics to disc. It is
        also accessible to callbacks for custom writing in other formats.
      callbacks: A sequence of self-contained programs executing non-essential
        logic (e.g. checkpoint saving, logging, timing, profiling etc.).
    """
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    train_iter = iter(train_dataloader)
    eval_iter = None
    run_evaluation = eval_dataloader is not None
    if run_evaluation:
        if eval_every_steps % metric_aggregation_steps != 0:
            raise ValueError(
                f"`eval_every_steps` ({eval_every_steps}) "
                f"must be an integer multiple of "
                f"`metric_aggregation_steps` ({metric_aggregation_steps})"
            )

        eval_iter = iter(eval_dataloader)
        if run_sanity_eval_batch and not trainer.is_compiled:
            trainer.eval(eval_iter, num_steps=1)

    for callback in callbacks:
        callback.metric_writer = metric_writer if (local_rank in [0, -1] and metric_writer) else None
        callback.on_train_begin(trainer)

    cur_step = trainer.train_state.int_step

    # setup for reinitializing iterator for training and evaluation
    if run_evaluation:
        epoch_eval = 1
        step_diff_eval = 1 if run_sanity_eval_batch else 0
        eval_steps_per_epoch = (
            len(eval_dataloader) // num_batches_per_eval * eval_every_steps
        )
        epochs_eval_steps = epoch_eval * eval_steps_per_epoch - step_diff_eval

    epoch_train = 1
    step_diff_train = 0
    epochs_train_steps = epoch_train * len(train_dataloader) - step_diff_train

    # Barrier before training
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    non_finite_recovery_attempts = 0

    while cur_step < total_train_steps:
        for callback in callbacks:
            callback.on_train_batches_begin(trainer)

        num_steps = min(total_train_steps - cur_step, metric_aggregation_steps)
        
        # evaluate if training dataset reinitialization is necessary
        if cur_step + num_steps > epochs_train_steps:
            epoch_train += 1  # increase epoch for training dataset

            if world_size > 1:
                # Reset for random shuffling
                train_dataloader.sampler.set_epoch(epoch_train)

            train_iter = iter(train_dataloader)
            step_diff_train += epochs_train_steps - cur_step
            epochs_train_steps = epoch_train * len(train_dataloader) - step_diff_train

        train_metrics = trainer.train(train_iter, num_steps).compute()
        
        
        if not _is_finite_value(train_metrics) and cur_step > 1000:
            recovered = _restore_latest_valid_checkpoint(
                trainer=trainer,
                workdir=workdir,
                world_size=world_size,
                local_rank=local_rank,
            )
            if not recovered:
                raise RuntimeError(
                    "Non-finite training metrics detected, but no valid checkpoint was "
                    "found for automatic recovery in '<workdir>/checkpoints'. "
                    "Check that evaluation has run and checkpoints include eval metrics."
                )

            # continue training from restored step without logging or callback updates
            cur_step = trainer.train_state.int_step
            continue
        
        cur_step += num_steps
        current_lr = _get_current_learning_rate(trainer.optimizer)

        if local_rank in [0, -1] and metric_writer:
            #metric_writer.add_scalars("train", train_metrics, cur_step)
            train_metrics_to_log = dict(train_metrics)
            if current_lr is not None:
                train_metrics_to_log["learning_rate"] = current_lr
            metric_writer.add_scalars("train", train_metrics_to_log, cur_step)

            _log_dict = {}
            for k in train_metrics_to_log:
                _log_dict[f"train/{k}"] = _to_float_scalar(train_metrics_to_log[k])
            _log_dict = {k: v for k, v in _log_dict.items() if v is not None}
            wandb.log(_log_dict, step=cur_step)

        # At train/eval batch end, callbacks are called in reverse order so that
        # they are last-in-first-out, loosely resembling nested python contexts.
        for callback in reversed(callbacks):
            callback.on_train_batches_end(trainer, train_metrics)

        if run_evaluation:
            if cur_step == total_train_steps or cur_step % eval_every_steps == 0:
                for callback in callbacks:
                    callback.on_eval_batches_begin(trainer)

                assert eval_iter is not None

                # evaluate if evaluation iterator needs to be reinitialized
                if cur_step + num_batches_per_eval > epochs_eval_steps:
                    epoch_eval += 1  # increase epoch for evaluation dataset

                    if world_size > 1:
                        # Reset for random shuffling
                        eval_dataloader.sampler.set_epoch(epoch_eval)

                    eval_iter = iter(eval_dataloader)
                    step_diff_eval += epochs_eval_steps - cur_step
                    epochs_eval_steps = (
                        epoch_eval * eval_steps_per_epoch - step_diff_eval
                    )

                eval_metrics = trainer.eval(eval_iter, num_batches_per_eval).compute()
                eval_metrics_to_log = {
                    k: v for k, v in eval_metrics.items() if train_utils.is_scalar(v)
                }

                if local_rank in [0, -1] and metric_writer:
                    metric_writer.add_scalars("eval", eval_metrics_to_log, cur_step)

                    _log_dict = {}
                    for k in eval_metrics_to_log:
                        _log_dict[f"eval/{k}"] = eval_metrics_to_log[k].detach().cpu().item()
                    wandb.log(_log_dict, step=cur_step)

                for callback in reversed(callbacks):
                    callback.on_eval_batches_end(trainer, eval_metrics)

    for callback in reversed(callbacks):
        callback.on_train_end(trainer)

    if local_rank in [0, -1] and metric_writer:
        metric_writer.flush()
