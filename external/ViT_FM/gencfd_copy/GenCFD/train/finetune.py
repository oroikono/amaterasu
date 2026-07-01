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

"""Fine-tuning entrypoint for GenCFD.

This script mirrors ``train_gencfd.py`` but supports loading a pretrained
checkpoint while gracefully skipping parameters whose shapes are incompatible
with the current target resolution.
"""

import os
import time

import torch
import torch.distributed as dist
import wandb
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from GenCFD.train import training_loop
from GenCFD.utils.dataloader_builder import get_dataset_loader
from GenCFD.utils.gencfd_builder import create_callbacks, create_denoiser, save_json_file
from GenCFD.utils.parser_utils import train_args

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/GenCFD/wandb_logs"


def init_distributed_mode(args):
    """Initialize a Distributed Data Parallel Environment."""

    args.local_rank = int(os.getenv("LOCAL_RANK", -1))

    if args.local_rank == -1:
        raise ValueError(
            "--local_rank was not set. Ensure torchrun is used to launch the script."
        )

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend="nccl", rank=args.local_rank, world_size=args.world_size
    )

    used_device = torch.device(f"cuda:{args.local_rank}")
    print(" ")
    print(f"DDP initialized with rank {args.local_rank} and device {used_device}.")

    return args, used_device


def _resolve_checkpoint_path(path: str) -> str:
    """Return checkpoint path; if directory is passed, return latest *.pth file."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    checkpoints = [f for f in os.listdir(path) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoints found in directory: {path}")

    checkpoints.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    return os.path.join(path, checkpoints[-1])


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


def load_pretrained_compatible_weights(
    trainer,
    checkpoint_path: str,
    force_reinit_keys: list[str] | None = None,
) -> list[str]:
    """Load all compatible checkpoint weights and reinitialize everything else.

    Parameters are loaded only when the key exists in the current model and the
    tensor shape matches. Any non-matching or missing parameter is left at its
    freshly initialized value. Additional exact keys can be force-reinitialized
    through ``force_reinit_keys`` (defaults to empty).
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "model_state_dict" in checkpoint:
        loaded_state_dict = checkpoint["model_state_dict"]
        checkpoint_is_compiled = checkpoint.get("is_compiled", False)
        checkpoint_is_ddp = checkpoint.get("is_parallelized", False)
    else:
        loaded_state_dict = checkpoint
        checkpoint_is_compiled = False
        checkpoint_is_ddp = False

    model_state_dict = trainer.model.denoiser.state_dict()
    adapted_state_dict = _adapt_state_dict_keys_for_model(
        state_dict=loaded_state_dict,
        model_is_compiled=trainer.is_compiled,
        model_is_ddp=trainer.is_parallelized,
        checkpoint_is_compiled=checkpoint_is_compiled,
        checkpoint_is_ddp=checkpoint_is_ddp,
    )

    force_reinit_keys = [] if force_reinit_keys is None else force_reinit_keys
    force_reinit_keyset = set(force_reinit_keys)

    compatible_state_dict = {}
    shape_mismatch_keys = []
    forced_reinitialized_keys = []

    for key, value in adapted_state_dict.items():
        if key not in model_state_dict:
            continue
        if key in force_reinit_keyset:
            forced_reinitialized_keys.append(key)
            continue
        if model_state_dict[key].shape != value.shape:
            shape_mismatch_keys.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_state_dict[key].shape)}"
            )
            continue
        compatible_state_dict[key] = value

    reinitialized_keys = sorted(
        set(model_state_dict.keys()) - set(compatible_state_dict.keys())
    )
    unexpected_in_checkpoint = sorted(
        set(adapted_state_dict.keys()) - set(model_state_dict.keys())
    )
    trainer.model.denoiser.load_state_dict(compatible_state_dict, strict=False)

    print(" ")
    print(f"Loaded pretrained weights from: {checkpoint_path}")
    print(f"Compatible parameters loaded: {len(compatible_state_dict)} / {len(model_state_dict)}")
    #print(f"Re-initialized parameters: {len(missing_in_checkpoint)}")
    print(f"Re-initialized parameters: {len(reinitialized_keys)}")

    if shape_mismatch_keys:
        print("Shape-mismatched checkpoint parameters (reinitialized):")
        for msg in shape_mismatch_keys:
            print(f"  - {msg}")
    if forced_reinitialized_keys:
        print("Forced reinitialized checkpoint parameters (exact keys):")
        for key in sorted(forced_reinitialized_keys):
            print(f"  - {key}")
    if reinitialized_keys:
        print("All reinitialized model parameters:")
        for key in reinitialized_keys:
            print(f"  - {key}")
    
    if unexpected_in_checkpoint:
        print(f"Ignored unexpected checkpoint parameters: {len(unexpected_in_checkpoint)}")

    return reinitialized_keys

if __name__ == "__main__":
    args = train_args()

    if args.world_size > 1:
        args, device = init_distributed_mode(args)
    else:
        print(" ")
        print(f"Used device: {device}")

    if args.model_dir is None:
        raise ValueError(
            "--model_dir must point to a pretrained checkpoint (.pth) or checkpoint directory for fine-tuning."
        )

    if args.finetune_scratch_only_entire_training and "3D" not in args.model_type:
        raise ValueError(
            "`--finetune_scratch_only_entire_training` is supported only for 3D models."
        )
    if args.finetune_unfreeze_ustack3d_output_after_warmup and "3D" not in args.model_type:
        raise ValueError(
            "`--finetune_unfreeze_ustack3d_output_after_warmup` is supported only for 3D models."
        )

    cwd = os.getcwd()
    if args.save_dir is None:
        raise ValueError("Save directory not specified in arguments!")
    savedir = os.path.join(cwd, args.save_dir)
    if not os.path.exists(savedir):
        if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
            os.makedirs(savedir)
            print(f"Created a directory to store metrics and models: {savedir}")

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        wandb.init(
            entity="bogdanraonic",
            project="diffusion-project",
            name=args.save_dir.strip().split("/")[-1],
            group=args.dataset,
            tags=[args.dataset, "finetune"],
            config=vars(args),
            dir=savedir,
        )

    train_dataloader, eval_dataloader, dataset, time_cond = get_dataset_loader(
        args=args,
        name=args.dataset,
        batch_size=args.batch_size,
        num_worker=args.worker,
        prefetch_factor=2,
        split=True,
        split_ratio=0.9,
    )

    print(len(train_dataloader), "TRAIN")
    print(len(eval_dataloader), "EVAL")

    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        save_json_file(
            args=args,
            time_cond=time_cond,
            split_ratio=0.9,
            out_shape=dataset.output_shape,
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
        model_params = sum(
            p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad
        )
        print(" ")
        print(f"Total number of model parameters: {model_params}")
        print(" ")

    optimizer = optim.AdamW(
        denoising_model.denoiser.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )

    trainer = training_loop.trainers.DenoisingTrainer(
        model=denoising_model,
        optimizer=optimizer,
        device=device,
        ema_decay=args.ema_decay,
        store_ema=True,
        track_memory=args.track_memory,
        use_mixed_precision=args.use_mixed_precision,
        is_compiled=args.compile,
        world_size=args.world_size,
        local_rank=args.local_rank,
        grad_accum_steps=args.grad_accum_steps,
        total_train_steps=args.num_train_steps,
        warmup_steps=(args.warmup_steps if args.warmup_steps is not None else 1000),
        min_lr=(args.min_lr if args.min_lr is not None else 1e-6),
        freeze_param_weight=args.freeze_param_weight,
        warmup_finetune_scratch_only=args.warmup_finetune_scratch_only,
        finetune_scratch_only_entire_training=args.finetune_scratch_only_entire_training,
        finetune_unfreeze_ustack3d_output_after_warmup= args.finetune_unfreeze_ustack3d_output_after_warmup,
        debug_track_param_name=args.debug_track_param_name,
        debug_track_param_log_every=args.debug_track_param_log_every,
    )

    checkpoint_path = _resolve_checkpoint_path(args.model_dir)
    #load_pretrained_compatible_weights(trainer, checkpoint_path)
    reinitialized_keys = load_pretrained_compatible_weights(trainer, checkpoint_path)
    trainer.configure_warmup_scratch_only(reinitialized_keys)

    start_train = time.time()
    metric_writer = SummaryWriter(log_dir=savedir) if args.local_rank in {0, -1} else None

    training_loop.run(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=savedir,
        world_size=args.world_size,
        local_rank=args.local_rank,
        total_train_steps=args.num_train_steps,
        metric_writer=metric_writer,
        metric_aggregation_steps=args.metric_aggregation_steps,
        eval_dataloader=eval_dataloader,
        eval_every_steps=args.eval_every_steps,
        num_batches_per_eval=args.num_batches_per_eval,
        callbacks=create_callbacks(args, savedir),
    )

    end_train = time.time()
    elapsed_train = end_train - start_train
    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        print(f"Done fine-tuning. Elapsed time {elapsed_train / 3600} h")

    if args.world_size > 1:
        dist.destroy_process_group()