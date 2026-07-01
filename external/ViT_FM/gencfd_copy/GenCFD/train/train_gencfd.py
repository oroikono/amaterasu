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

"""Main File to run Training for GenCFD."""

import time
import os
import math
import wandb
import multiprocessing as mp

# Set the cache size and debugging for torch.compile before importing torch
# os.environ["TORCH_LOGS"] = "all"  # or any of the valid log settings
import torch
import torch.distributed as dist
from torch.distributed import is_initialized
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from GenCFD.train import training_loop
from GenCFD.utils.dataloader_builder import get_dataset_loader
from GenCFD.utils.gencfd_builder import (
    create_denoiser,
    create_callbacks,
    save_json_file,
)
from GenCFD.utils.parser_utils import train_args

torch.set_float32_matmul_precision("high")  # Better performance on newer GPUs!
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# Setting global seed for reproducibility
torch.manual_seed(SEED)  # For CPU operations
torch.cuda.manual_seed(SEED)  # For GPU operations
torch.cuda.manual_seed_all(SEED)  # Ensure all GPUs (if multi-GPU) are set

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/GenCFD/wandb_logs"


def init_distributed_mode(args):
    """Initialize a Distributed Data Parallel Environment"""

    args.local_rank = int(os.getenv("LOCAL_RANK", -1))  # Get from environment variable

    if args.local_rank == -1:
        raise ValueError(
            "--local_rank was not set. Ensure torchrun is used to launch the script."
        )

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend="nccl", rank=args.local_rank, world_size=args.world_size
    )

    device = torch.device(f"cuda:{args.local_rank}")
    print(" ")
    print(f"DDP initialized with rank {args.local_rank} and device {device}.")

    return args, device


if __name__ == "__main__":

    #mp.set_start_method("spawn", force=True)

    # get arguments for training
    args = train_args()

    # Initialize distributed mode (if multi-GPU)
    if args.world_size > 1:
        args, device = init_distributed_mode(args)
    else:
        print(" ")
        print(f"Used device: {device}")

    cwd = os.getcwd()
    if args.save_dir is None:
        raise ValueError("Save directory not specified in arguments!")
    savedir = os.path.join(cwd, args.save_dir)
    if not os.path.exists(savedir):
        if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
            os.makedirs(savedir)
            print(f"Created a directory to store metrics and models: {savedir}")

    
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        run = wandb.init(entity="bogdanraonic", 
                    project="diffusion-project", 
                    name= args.save_dir.strip().split("/")[-1], 
                    group=args.dataset,
                    tags = [args.dataset],
                    config=vars(args),
                    #sync_tensorboard=True,
                    dir = savedir)

    train_dataloader, eval_dataloader, dataset, time_cond = get_dataset_loader(
        args=args,
        name=args.dataset,
        batch_size=args.batch_size,
        num_worker=args.worker,
        prefetch_factor=2,  # Default DataLoader value
        split=True,
        split_ratio=0.9,
    )

    print(len(train_dataloader), "TRAIN")
    print(len(eval_dataloader), "EVAL")
    
    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        # Save parameters in a JSON File
        save_json_file(
            args=args,
            time_cond=time_cond,
            split_ratio=0.9,
            out_shape=dataset.output_shape,  # output shape of the prediction
            input_channel=dataset.input_channel,
            output_channel=dataset.output_channel,
            spatial_resolution=dataset.spatial_resolution,
            device=device,
            seed=SEED,
        )

    denoising_model = create_denoiser(
        args=args,
        input_channels=dataset.input_channel,
        out_channels=dataset.output_channel,
        spatial_resolution=dataset.spatial_resolution,
        time_cond=time_cond,
        device=device,
        dtype=args.dtype,
        use_ddp_wrapper=True,
        param_dim=getattr(dataset, "num_parameters", 0),
        param_embed_channels=getattr(dataset, "param_embed_channels", 0),
    )

    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        # Print number of Parameters:
        model_params = sum(
            p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad
        )
        print(" ")
        print(f"Total number of model parameters: {model_params}")
        print(" ")

    # Initialize optimizer
    optimizer = optim.AdamW(
        denoising_model.denoiser.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
        eps=1e-6,
    )

    max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0 else None

    trainer = training_loop.trainers.DenoisingTrainer(
        model=denoising_model,
        optimizer=optimizer,
        device=device,
        ema_decay=args.ema_decay,
        store_ema=True,  # Store ema model as well
        track_memory=args.track_memory,
        use_mixed_precision=args.use_mixed_precision,
        is_compiled=args.compile,
        world_size=args.world_size,
        local_rank=args.local_rank,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=max_grad_norm,
        total_train_steps=args.num_train_steps,
        warmup_steps=(args.warmup_steps if args.warmup_steps is not None else 1000),
        min_lr=(args.min_lr if args.min_lr is not None else 1e-6),
        freeze_param_weight=args.freeze_param_weight,
    )

    start_train = time.time()

    # Initialize the metric writer
    metric_writer = (
        SummaryWriter(log_dir=savedir) if args.local_rank in {0, -1} else None
    )

    training_loop.run(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=savedir,
        # DDP configs
        world_size=args.world_size,
        local_rank=args.local_rank,
        # Training configs
        total_train_steps=args.num_train_steps,
        metric_writer=metric_writer,
        metric_aggregation_steps=args.metric_aggregation_steps,
        # Evaluation configs:
        eval_dataloader=eval_dataloader,
        eval_every_steps=args.eval_every_steps,
        # Other configs
        num_batches_per_eval=args.num_batches_per_eval,
        callbacks=create_callbacks(args, savedir),
    )

    end_train = time.time()
    elapsed_train = end_train - start_train
    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        print(f"Done training. Elapsed time {elapsed_train / 3600} h")

    if args.world_size > 1:
        dist.destroy_process_group()
