# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich
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

"""Trainer classes for use in gradient descent mini-batch training."""

import abc
import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torchmetrics import MetricCollection, MeanMetric

from GenCFD.utils.train_utils import StdMetric, compute_memory
from GenCFD.train import train_states
import GenCFD.diffusion as dfn_lib

Tensor = torch.Tensor
BatchType = Mapping[str, Tensor]
Metrics = MetricCollection

M = TypeVar("M")  # Model
S = TypeVar("S", bound=train_states.BasicTrainState)
D = TypeVar("D", bound=dfn_lib.DenoisingModel)
SD = TypeVar("SD", bound=train_states.DenoisingModelTrainState)


class BaseTrainer(Generic[M, S], metaclass=abc.ABCMeta):
    """Abstract base trainer for gradient descent mini-batch training."""

    def __init__(
        self,
        model: M,
        device: torch.device = None,
        track_memory: bool = False,
        world_size: int = 1,
        local_rank: int = -1,
    ):
        self.model = model
        self.device = device
        self.train_state = self.initialize_train_state()
        self.track_memory = track_memory
        self.world_size = world_size
        self.local_rank = local_rank

    @property
    @abc.abstractmethod
    def train_step(self) -> Metrics:
        """Returns the train step function."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def eval_step(self) -> Callable[[S, BatchType], Metrics]:
        """Returns the evaluation step function."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_train_state(self) -> S:
        """Instantiate the initial train state."""
        raise NotImplementedError

    def train(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
        """Runs training for a specified number of steps."""

        train_metrics = self.TrainMetrics(
            device=self.device,
            track_memory=self.track_memory,
            world_size=self.world_size,
        )
        self.model.denoiser.train()

        for step in range(num_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            metrics_update = self.train_step(batch)

            train_metrics["loss"].update(metrics_update["loss"])
            train_metrics["loss_std"].update(metrics_update["loss"])

            if "grad_norm" in metrics_update:
                train_metrics["grad_norm"].update(metrics_update["grad_norm"])

            if self.track_memory and "mem" in metrics_update:
                train_metrics["mem"].update(metrics_update["mem"])
        
        self.flush_grad_accumulation()

        if self.track_memory and self.device.type != "cuda":
            print(f"Warning: Memory tracking is skipped. CUDA device is not available.")

        if self.world_size > 1:
            # Barrier / Synchronization before training aggregation
            dist.barrier(device_ids=[self.local_rank])

        return train_metrics

    def eval(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
        """Runs evaluation for a specified number of steps."""
        eval_metrics = self.EvalMetrics(
            self.device, self.model.num_eval_noise_levels, self.world_size
        )
        self.model.denoiser.eval()

        with torch.no_grad():
            for _ in range(num_steps):
                batch = next(batch_iter)
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
                update_metrics = self.eval_step(
                    batch
                )  # self.train_state as first entry
                for key, value in update_metrics.items():
                    eval_metrics[key].update(value)

        if self.world_size > 1:
            # Barrier / Synchronization before evaluation aggregation
            dist.barrier(device_ids=[self.local_rank])

        return eval_metrics
    
    def flush_grad_accumulation(self) -> None:
        """Flushes pending gradient accumulation if implemented by subclass."""
        return None

class BasicTrainer(BaseTrainer[M, S]):
    """Basic Trainer implementing the training/evaluation steps."""

    def _resolve_debug_param_name(self) -> str | None:
        """Finds a model parameter matching the requested debug name."""

        if not self.debug_track_param_name:
            return None

        named_params = list(self.model.denoiser.named_parameters())
        requested = self.debug_track_param_name

        # 1) exact match
        for name, _ in named_params:
            if name == requested:
                return name

        # 2) match after removing optional DDP prefix
        for name, _ in named_params:
            if name.replace("module.", "") == requested:
                return name

        # 3) suffix fallback for convenience
        for name, _ in named_params:
            if name.endswith(requested):
                return name

        if self.local_rank in [0, -1]:
            print(
                "`debug_track_param_name` was set but no matching parameter was "
                f"found: {requested}"
            )
        return None

    def _maybe_log_debug_param_update(self, optimizer_step: int, before_val: float | None) -> None:
        """Logs tracked-parameter update information at a configurable interval."""

        if not self._debug_tracked_param_resolved_name:
            return
        if optimizer_step % self.debug_track_param_log_every != 0:
            return

        tracked_param = dict(self.model.denoiser.named_parameters()).get(
            self._debug_tracked_param_resolved_name
        )
        if tracked_param is None:
            return
        if tracked_param.numel() == 0:
            return

        after_val = tracked_param.detach().reshape(-1)[0].item()
        delta = (after_val - before_val) if before_val is not None else float("nan")
        grad_val = float("nan")
        if tracked_param.grad is not None and tracked_param.grad.numel() > 0:
            grad_val = tracked_param.grad.detach().reshape(-1)[0].item()

        if self.local_rank in [0, -1]:
            print(
                "[debug-param] "
                f"opt_step={optimizer_step} "
                f"param={self._debug_tracked_param_resolved_name} "
                f"first_value={after_val:.8e} "
                f"delta={delta:.8e} "
                f"grad_first={grad_val:.8e} "
                f"requires_grad={tracked_param.requires_grad}")

    class TrainMetrics(Metrics):
        """Training metrics based on the model outputs."""

        # Example usage:
        # train_loss = MeanMetric()
        # train_acc = torchmetrics.Accuracy()
        # memory tracer if set to True
        def __init__(self, device, track_memory: bool = False):
            metrics = {
                # 'train_loss': MeanMetric(),
                #'train_acc': torchmetrics.Accuracy()
            }
            super().__init__(metrics)

    class EvalMetrics(Metrics):
        """Evaluation metrics based on model outputs."""

        # Example usage:
        # eval_loss = torchmetrics.MeanSquaredError()
        # eval_acc = torchmetrics.Accuracy()

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = None,
        track_memory: bool = False,
        world_size: int = 1,
        local_rank: int = -1,
    ):
        super().__init__(
            model=model,
            device=device,
            track_memory=track_memory,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.optimizer = optimizer

    def train_step(self, batch: BatchType) -> Metrics:

        self.model.train()
        output = self.model(batch)
        loss, metrics = self.model.loss_fn(output, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_train_state()

        train_metrics = self.TrainMetrics()
        train_metrics.update(torch.tensor(metrics["loss"]))

        return train_metrics

    def eval_step(self, batch: BatchType) -> Callable[[S, BatchType], Metrics]:
        with torch.no_grad():
            metrics = self.model.eval_fn(batch)

        eval_metrics = self.EvalMetrics(
            self.device, self.model.num_eval_noise_levels, self.world_size
        )
        for key, value in metrics.items():
            eval_metrics[key](value)

        return eval_metrics.compute()

    def initialize_train_state(self) -> S:
        """Initializes the training state, including optimizer and parameters."""
        return train_states.BasicTrainState(
            model=self.model,
            optimizer=self.optimizer,
            params=self.model.state_dict(),
            opt_state=self.optimizer.state_dict(),
        )

    def update_train_state(self) -> S:
        """Update the training state, including optimizer and parameters."""
        next_step = self.train_state.step + 1
        if isinstance(next_step, Tensor):
            next_step = next_step.item()

        return self.train_state.replace(
            step=next_step,
            opt_state=self.optimizer.state_dict(),
            params=self.model.state_dict(),
        )


class BasicDistributedTrainer(BasicTrainer[M, S]):
    """Distributed Trainer for DDP (DistributedDataParallel) training."""

    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, device: torch.device
    ):
        super().__init__(model, optimizer, device)
        self.model = DDP(self.model, device_ids=[device])

    def train_step(self, batch: BatchType) -> Metrics:
        return super().train_step(batch)

    def eval_step(self, batch: BatchType) -> Metrics:
        return super().eval_step(batch)


class DenoisingTrainer(BasicTrainer[M, SD]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        ema_decay: float = 0.999,
        store_ema: bool = False,
        track_memory: bool = False,
        use_mixed_precision: bool = False,
        is_compiled: bool = False,
        world_size: int = 1,
        local_rank: int = -1,
        grad_accum_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        total_train_steps: int = 1,
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
        freeze_param_weight: bool = False,
        warmup_finetune_scratch_only: bool = False,
        finetune_scratch_only_entire_training: bool = False,
        finetune_unfreeze_ustack3d_output_after_warmup: bool = False,
        debug_track_param_name: str | None = None,
        debug_track_param_log_every: int = 100,
    ):

        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.store_ema = store_ema
        self.track_memory = track_memory
        # Mixed precision training with Grad scaler to avoid overflow and underflow during backprop.
        self.compute_dtype = torch.float16 if use_mixed_precision else torch.float32
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.amp.GradScaler(device.type) if use_mixed_precision else None

        if grad_accum_steps < 1:
            raise ValueError("`grad_accum_steps` must be >= 1")
        self.grad_accum_steps = grad_accum_steps
        self._grad_accum_counter = 0

        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("`max_grad_norm` must be > 0 when provided")
        self.max_grad_norm = max_grad_norm

        if total_train_steps < 1:
            raise ValueError("`total_train_steps` must be >= 1")
        self.total_train_steps = total_train_steps
        self.warmup_steps = max(0, warmup_steps)
        self.min_lr = min_lr
        self._peak_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self._lr_update_step = 0
        self._set_optimizer_lrs([self.min_lr for _ in self.optimizer.param_groups])
        
        self.freeze_param_weight = freeze_param_weight
        self.warmup_finetune_scratch_only = warmup_finetune_scratch_only
        self.finetune_scratch_only_entire_training = finetune_scratch_only_entire_training
        self.finetune_unfreeze_ustack3d_output_after_warmup = finetune_unfreeze_ustack3d_output_after_warmup

        self.debug_track_param_name = (
            debug_track_param_name.strip() if debug_track_param_name else None
        )
        self.debug_track_param_log_every = max(1, debug_track_param_log_every)

        if self.freeze_param_weight and self.warmup_finetune_scratch_only:
            raise ValueError(
                "`freeze_param_weight` and `warmup_finetune_scratch_only` cannot be "
                "enabled together."
            )
        if (
            self.warmup_finetune_scratch_only
            and self.finetune_scratch_only_entire_training
        ):
            raise ValueError(
                "`warmup_finetune_scratch_only` and "
                "`finetune_scratch_only_entire_training` cannot be enabled together."
            )
        if (
            self.finetune_unfreeze_ustack3d_output_after_warmup
            and not self.finetune_scratch_only_entire_training
        ):
            raise ValueError(
                "`finetune_unfreeze_ustack3d_output_after_warmup` requires "
                "`finetune_scratch_only_entire_training`."
            )

        self._param_weight_modules: list[nn.Module] = []
        self._param_weights_frozen = False
        self._scratch_param_names: set[str] = set()
        self._scratch_only_mode_active = False
        self._scratch_only_entire_training_active = False
        self._ustack3d_output_param_names: set[str] = set()

        self._debug_tracked_param_resolved_name: str | None = None

        # Store status if the model is compiled and / or parallellized
        self.is_compiled = is_compiled
        self.is_parallelized = True if world_size > 1 else False

        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            track_memory=track_memory,
            world_size=world_size,
            local_rank=local_rank,
        )

        if self.freeze_param_weight:
            self._param_weight_modules = self._collect_param_weight_modules()
            self._set_param_weights_frozen(True)

        if self.debug_track_param_name:
            self._debug_tracked_param_resolved_name = self._resolve_debug_param_name()

    def configure_warmup_scratch_only(self, scratch_param_names: list[str]) -> None:
        """Configures warmup/finetune modes that target scratch-initialized parameters."""

        if not (
            self.warmup_finetune_scratch_only
            or self.finetune_scratch_only_entire_training
        ):
            return

        available_param_names = {
            name for name, _ in self.model.denoiser.named_parameters()
        }
        self._scratch_param_names = set(scratch_param_names) & available_param_names

        if not self._scratch_param_names:
            self._scratch_only_mode_active = False
            if self.local_rank in [0, -1]:
                print(
                    "Scratch-only fine-tuning enabled, but no scratch parameters "
                    "were detected. Training all parameters."
                )
            return

        if self.finetune_unfreeze_ustack3d_output_after_warmup:
            self._ustack3d_output_param_names = self._collect_ustack3d_output_param_names()
            if not self._ustack3d_output_param_names and self.local_rank in [0, -1]:
                print(
                    "`finetune_unfreeze_ustack3d_output_after_warmup` enabled, but "
                    "no UStack3D output parameters were found."
                )

        self._scratch_only_entire_training_active = (
            self.finetune_scratch_only_entire_training
        )

        self._scratch_only_mode_active = True
        self._set_scratch_only_mode(warmup_active=True)

    def _collect_ustack3d_output_param_names(self) -> set[str]:
        """Returns parameter names for the 3D UStack output conv and res-skip layers."""

        conv_layer_indices = []
        named_params = list(self.model.denoiser.named_parameters())
        for name, _ in named_params:
            normalized_name = name.replace("module.", "")
            if "UStack.conv_layers." not in normalized_name:
                continue
            try:
                idx = int(normalized_name.split("UStack.conv_layers.", 1)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                continue
            conv_layer_indices.append(idx)

        if not conv_layer_indices:
            return set()

        output_conv_index = max(conv_layer_indices)
        selected = set()
        for name, _ in named_params:
            normalized_name = name.replace("module.", "")
            if "UStack.res_skip_layer." in normalized_name:
                selected.add(name)
            if f"UStack.conv_layers.{output_conv_index}." in normalized_name:
                selected.add(name)
        return selected

    def _set_scratch_only_mode(self, warmup_active: bool) -> None:
        for name, param in self.model.denoiser.named_parameters():
            should_train = name in self._scratch_param_names
            if (
                self._scratch_only_entire_training_active
                and not warmup_active
                and self.finetune_unfreeze_ustack3d_output_after_warmup
            ):
                should_train = should_train or (name in self._ustack3d_output_param_names)
            param.requires_grad = should_train

    def _maybe_update_scratch_only_mode(self) -> None:
        if not self._scratch_only_mode_active:
            return

        #keep_scratch_only = self._lr_update_step < self.grad_accum_steps * self.warmup_steps
        #self._set_scratch_only_mode(keep_scratch_only)

        if self._scratch_only_entire_training_active:
            keep_scratch_only = True
            warmup_active = self._lr_update_step < self.warmup_steps
        else:
            keep_scratch_only = self._lr_update_step < self.warmup_steps
            warmup_active = keep_scratch_only

        self._set_scratch_only_mode(warmup_active=warmup_active)

        if not keep_scratch_only:
            self._scratch_only_mode_active = False

    def _collect_param_weight_modules(self) -> list[nn.Module]:
        modules = []
        for module in self.model.denoiser.modules():
            if hasattr(module, "param_scale") and hasattr(module, "param_shift"):
                modules.append(module)
        return modules

    def _set_param_weights_frozen(self, frozen: bool) -> None:
        for module in self._param_weight_modules:
            if getattr(module, "param_scale", None) is not None:
                module.param_scale.weight.requires_grad = not frozen
            if getattr(module, "param_shift", None) is not None:
                module.param_shift.weight.requires_grad = not frozen
        self._param_weights_frozen = frozen

    def _maybe_update_param_weight_freeze(self, step: int) -> None:
        if not self.freeze_param_weight or not self._param_weight_modules:
            return

        should_freeze = step <= self.warmup_steps
        if should_freeze != self._param_weights_frozen:
            self._set_param_weights_frozen(should_freeze)


    def _set_optimizer_lrs(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = float(lr)

    def _lr_for_step(self, step: int, peak_lr: float) -> float:
        if step < 1:
            return self.min_lr

        warmup_steps = min(self.warmup_steps, self.total_train_steps)
        #if warmup_steps > 1 and step <= warmup_steps:
        #    progress = (step - 1) / (warmup_steps - 1)
        warmup_constant_steps = warmup_steps // 2
        warmup_linear_steps = warmup_steps - warmup_constant_steps

        if step <= warmup_constant_steps:
            return self.min_lr

        if step <= warmup_steps:
            if warmup_linear_steps <= 1:
                return peak_lr
            linear_step = step - warmup_constant_steps - 1
            progress = linear_step / (warmup_linear_steps - 1)
            return self.min_lr + progress * (peak_lr - self.min_lr)
        
        #if warmup_steps == 1 and step == 1:
        #    return peak_lr

        decay_steps = self.total_train_steps - warmup_steps
        if decay_steps <= 0:
            return peak_lr

        decay_progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.min_lr + (peak_lr - self.min_lr) * cosine_term

    def _apply_lr_for_next_optimizer_step(self) -> None:
        next_step = min(self._lr_update_step + 1, self.total_train_steps)
        self._set_optimizer_lrs(
            [self._lr_for_step(next_step, peak_lr) for peak_lr in self._peak_lrs]
        )

    class TrainMetrics(Metrics):
        """Train metrics including mean and std of loss and if required
        computes the mean of the memory profiler."""

        def __init__(self, device, track_memory: bool = False, world_size: int = 1):
            train_metrics = {
                "loss": MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device),
                
                "grad_norm": MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device),
                
                "loss_std": StdMetric().to(
                    device
                ),  # uses already reduction clauses thus no sync
            }
            if track_memory:
                train_metrics["mem"] = MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device)

            super().__init__(metrics=train_metrics)

    class EvalMetrics(Metrics):
        """Evaluation metrics based on the model output, using noise level"""

        def __init__(self, device, num_eval_noise_levels: int, world_size: int = 1):
            eval_metrics = {
                f"denoise_lvl{i}": MeanMetric(
                    sync_on_compute=True if world_size > 1 else False
                ).to(device)
                for i in range(num_eval_noise_levels)
            }
            super().__init__(metrics=eval_metrics)

    def initialize_train_state(self) -> SD:
        """Initializes the train state with EMA and model params

        Those states are tracked at every iteration step
        """
        return train_states.DenoisingModelTrainState(
            # Further parameters can be added here to track
            model=self.model.denoiser if self.store_ema else None,
            step=0,
            ema_decay=self.ema_decay,
            store_ema=self.store_ema,
        )

    @compute_memory
    def train_step(self, batch: BatchType) -> Metrics:
        
        step = self.train_state.step + 1
        self._maybe_update_param_weight_freeze(step)
        self._maybe_update_scratch_only_mode()

        with torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype):
            loss, metrics = self.model.loss_fn(batch)

        ###self.optimizer.zero_grad(set_to_none=True)
        
        def _compute_grad_norm() -> Tensor:
            total_norm = torch.zeros((), device=self.device, dtype=torch.float32)
            for param in self.model.denoiser.parameters():
                if param.grad is None:
                    continue
                grad_norm = torch.linalg.vector_norm(param.grad.detach().float(), ord=2)
                total_norm = total_norm + grad_norm.square()
            return torch.sqrt(total_norm)

        if not torch.isfinite(loss.detach()).all():
            safe_loss = torch.ones((), device=loss.device, dtype=loss.dtype)
            metrics["loss"] = safe_loss.detach()
            metrics["grad_norm"] = torch.ones((), device=self.device, dtype=torch.float32)
            self.optimizer.zero_grad(set_to_none=True)
            self._grad_accum_counter = 0
            self.update_train_state()
            return metrics

        if self._grad_accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        scaled_loss = loss / self.grad_accum_steps
        self._grad_accum_counter += 1
        should_step = self._grad_accum_counter >= self.grad_accum_steps

        #if step == 1 or step == self.warmup_steps :
        #trainable_params = sum(
        #    param.numel() for param in self.model.denoiser.parameters() if param.requires_grad
        #)
        #if step == 1:
        #    print(f"Trainable parameters; Start of the warmup {trainable_params}")
        #else:
        #    print(f"Trainable parameters; End of the warmup {trainable_params}")

        if self.use_mixed_precision:
            self.scaler.scale(scaled_loss.float()).backward(retain_graph=False)
            if should_step:
                
                
                ### Debug params:
                tracked_before_val = None
                if self._debug_tracked_param_resolved_name:
                    tracked_param = dict(self.model.denoiser.named_parameters()).get(
                        self._debug_tracked_param_resolved_name
                    )
                    if tracked_param is not None and tracked_param.numel() > 0:
                        tracked_before_val = tracked_param.detach().reshape(-1)[0].item()

                self._apply_lr_for_next_optimizer_step()
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.denoiser.parameters(), self.max_grad_norm
                    )
                metrics["grad_norm"] = _compute_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._lr_update_step = min(
                    self._lr_update_step + 1, self.total_train_steps)
                
                self._maybe_log_debug_param_update(self._lr_update_step, tracked_before_val)
        else:
            scaled_loss.backward(retain_graph=False)
            if should_step:

                tracked_before_val = None
                if self._debug_tracked_param_resolved_name:
                    tracked_param = dict(self.model.denoiser.named_parameters()).get(
                        self._debug_tracked_param_resolved_name
                    )
                    if tracked_param is not None and tracked_param.numel() > 0:
                        tracked_before_val = tracked_param.detach().reshape(-1)[0].item()
                
                self._apply_lr_for_next_optimizer_step()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.denoiser.parameters(), self.max_grad_norm
                    )
                metrics["grad_norm"] = _compute_grad_norm()
                self.optimizer.step()
                self._lr_update_step = min(
                    self._lr_update_step + 1, self.total_train_steps
                )

                self._maybe_log_debug_param_update(self._lr_update_step, tracked_before_val)

        if should_step:
            self.optimizer.zero_grad(set_to_none=True)
            self._grad_accum_counter = 0
        
        self.update_train_state()

        return metrics

    def flush_grad_accumulation(self) -> None:
        """Steps optimizer if there are remaining accumulated gradients."""
        if self._grad_accum_counter == 0:
            return

        if self.use_mixed_precision:
            self._apply_lr_for_next_optimizer_step()
            self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.denoiser.parameters(), self.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._lr_update_step = min(self._lr_update_step + 1, self.total_train_steps)
        else:
            self._apply_lr_for_next_optimizer_step()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.denoiser.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self._lr_update_step = min(self._lr_update_step + 1, self.total_train_steps)

        self.optimizer.zero_grad(set_to_none=True)
        self._grad_accum_counter = 0

    def update_train_state(self) -> SD:
        """Update the training state, including optimizer and parameters."""
        next_step = self.train_state.step + 1
        if isinstance(next_step, Tensor):
            next_step = next_step.item()

        # update ema model
        if self.store_ema:
            self.train_state.ema_model.update_parameters(self.model.denoiser)
            ema_params = self.train_state.ema_parameters

        # Further states can be replaced at every training step
        return self.train_state.replace(
            step=next_step, ema=ema_params if self.store_ema else None
        )

    @staticmethod
    def inference_fn_from_state_dict(
        state: SD,
        denoiser: nn.Module,
        *args,
        use_ema: bool = False,
        lead_time: bool = False,
        **kwargs,
    ):
        denoiser.eval()
        if use_ema:
            if state.ema_model:
                denoiser.load_state_dict(state.ema_parameters)

            else:
                raise ValueError("EMA model is None or not initialized")

        return dfn_lib.DenoisingModel.inference_fn(
            denoiser, lead_time, *args, **kwargs
        )
