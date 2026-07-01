import argparse
import os
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import tqdm

# Add repository root to sys.path.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from regression.ViTModulev2 import Vit3_pl
from utils.spectral_utils import spectral_utils
from utils.utils_data import find_files_with_extension, get_loader, load_data, read_cli_inference
from utils.utils_finetune_3d import initialize_FT3d


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _normalize_cli_tokens(argv: list[str]) -> list[str]:
    normalized = []
    for token in argv:
        stripped = token.strip()
        if stripped.startswith(("–", "—", "−")):
            stripped = f"--{stripped[1:]}"
        normalized.append(stripped)
    return normalized


def _parse_legacy_export_args(parser: argparse.ArgumentParser, unknown_args: list[str]) -> argparse.Namespace:
    legacy_parser = argparse.ArgumentParser(add_help=False)
    legacy_parser.add_argument("--which_type", type=str)
    legacy_parser.add_argument("--output_nc_path", type=str)
    legacy_parser.add_argument("--channel_names", nargs="+")
    legacy_parser.add_argument("--overwrite", type=_str2bool)
    legacy_args, remaining = legacy_parser.parse_known_args(unknown_args)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    return legacy_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spectral-training data from a 3D regression model.")
    parser = read_cli_inference(parser)
    parser.add_argument("--which_type", type=str, default="test", help="Dataset split to use: train/val/test.")
    parser.add_argument("--output_nc_path", type=str, default=None, help="Optional absolute path to the output NetCDF file.")
    parser.add_argument("--channel_names", nargs="+", default=None, help="Optional output channel names stored in the NetCDF file.")
    parser.add_argument("--overwrite", type=_str2bool, default=False, help="Overwrite an existing output file instead of resuming.")
    parser.add_argument("--max_z", type=int, default=None, help="Optional z-resolution crop applied before inference/export.")
    parser.add_argument("--member_start", type=int, default=None, help="Inclusive global member index assigned to this worker.")
    parser.add_argument("--member_end", type=int, default=None, help="Exclusive global member index assigned to this worker.")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of workers/jobs used for the export.")
    parser.add_argument("--shard_index", type=int, default=None, help="0-based worker index when `--num_shards` is provided.")

    cli_tokens = _normalize_cli_tokens(sys.argv[1:])
    args, unknown_args = parser.parse_known_args(cli_tokens)
    if not unknown_args:
        return args

    legacy_args = _parse_legacy_export_args(parser, unknown_args)
    for key, value in vars(legacy_args).items():
        if value is not None:
            setattr(args, key, value)
    return args


def _flag_was_provided(flag: str) -> bool:
    return flag in _normalize_cli_tokens(sys.argv[1:])


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args

    if os.path.isdir(args.config):
        raise IsADirectoryError(
            f"`--config` must point to a JSON file, but received the directory `{args.config}`."
        )
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Could not find config JSON: `{args.config}`")
    
    config_dict = load_data(args.config)

    cli_overrides = {
        "--device": ("device", args.device),
        "--batch_size": ("batch_size", args.batch_size),
        "--N_samples": ("N_samples", args.N_samples),
        "--which_type": ("which_type", args.which_type),
        "--output_nc_path": ("output_nc_path", args.output_nc_path),
        "--channel_names": ("channel_names", args.channel_names),
        "--overwrite": ("overwrite", args.overwrite),
        "--inference_tag": ("inference_tag", args.inference_tag),
        "--which_data": ("which_data", args.which_data),
        "--config_regression": ("config_regression", args.config_regression),
        "--max_z": ("max_z", args.max_z),
        "--member_start": ("member_start", args.member_start),
        "--member_end": ("member_end", args.member_end),
        "--num_shards": ("num_shards", args.num_shards),
        "--shard_index": ("shard_index", args.shard_index),
    }
    for flag, (key, value) in cli_overrides.items():
        if _flag_was_provided(flag):
            config_dict[key] = value

    return argparse.Namespace(**config_dict)


def get_model_root(config: argparse.Namespace) -> str:
    model_root = getattr(config, "config_regression_folder", None)
    if model_root is None:
        model_root = getattr(config, "config_regression", None)
    if model_root is None:
        raise ValueError("Expected `config_regression_folder` or `config_regression` in the config.")
    return model_root


def _is_model_run_folder(path: str) -> bool:
    if not os.path.isdir(path):
        return False

    model_dir = os.path.join(path, "model")
    has_model_dir = os.path.isdir(model_dir)
    has_checkpoint = bool(find_files_with_extension(model_dir, "ckpt", [], is_pl=True)) if has_model_dir else False
    has_config = bool(find_files_with_extension(path, "json", ["param"]))
    return has_model_dir and has_checkpoint and has_config


def get_candidate_model_folders(model_root: str, tags: Optional[Iterable[str]], exclude: Optional[Iterable[str]]) -> tuple[list[str], bool]:
    tags = list(tags or [])
    exclude = list(exclude or [])
    if _is_model_run_folder(model_root):
        return [model_root], True

    if ("finetuned" in model_root) or (("scratch" in model_root or "Scratch" in model_root) and "scratch_Base" not in model_root):
        folders = []
        for entry in os.scandir(model_root):
            if not entry.is_dir():
                continue
            folder_name = entry.name
            if any(tag not in folder_name for tag in tags):
                continue
            if any(tag_exclude in folder_name for tag_exclude in exclude):
                continue
            if not _is_model_run_folder(entry.path):
                continue
            folders.append(folder_name)
        return sorted(folders), False

    return [model_root], True


def load_regression_model(model_folder: str, device: str) -> tuple[torch.nn.Module, dict, str]:
    checkpoint_candidates = find_files_with_extension(os.path.join(model_folder, "model"), "ckpt", [], is_pl=True)
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"Could not find a Lightning checkpoint under `{os.path.join(model_folder, 'model')}`."
        )

    config_candidates = find_files_with_extension(model_folder, "json", ["param"])
    if not config_candidates:
        raise FileNotFoundError(
            f"Could not find a regression config JSON containing `param` in `{model_folder}`."
        )

    checkpoint_path = str(checkpoint_candidates[0])
    config_path = str(config_candidates[0])
    model_config = vars(argparse.Namespace(**load_data(config_path)))

    arch = model_config["config_arch"]
    if isinstance(arch, dict):
        arch_config = arch
    else:
        arch_config = load_data(arch)

    model_config["workdir"] = None
    init_new = model_config.get("init_new", False)
    target_s = model_config.get("s_new", model_config["s"])

    if model_config.get("patch_size_new") is not None:
        patch_size = model_config["patch_size_new"]
    elif isinstance(target_s, (list, tuple)):
        patch_size = [8, 8, 8]
    elif target_s == 64:
        patch_size = 4
    else:
        patch_size = 8

    regression_model = Vit3_pl(
        in_dim=model_config["in_dim"],
        out_dim=model_config["out_dim"],
        loss_fn=None,
        config_train=model_config,
        config_arch=arch_config,
    )
    regression_model = initialize_FT3d(
        model=regression_model,
        new_in_dim=model_config["in_dim"],
        new_out_dim=model_config["out_dim"],
        new_s=target_s,
        new_patch_size=patch_size,
        dims=arch_config["dims"],
        latent_channels=arch_config["latent_channels"],
        init_new=init_new,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    if any("_orig_mod." in k for k in state_dict):
        state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}
    if any("_orig_mod" in k for k in state_dict):
        state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}

    regression_model.load_state_dict(state_dict)
    model = regression_model.model.to(device).eval()
    return model, model_config, checkpoint_path


def build_loader(config: argparse.Namespace, model_config: Optional[dict] = None):
    masked_input = getattr(config, "is_masked", False)
    if not masked_input:
        masked_input = None

    which_type = getattr(config, "which_type", "traon")
    if which_type not in {"train", "val", "test"}:
        which_type = "train"
    
    if model_config is None:
        err_group = np.array(config.err_group)
        input_dim = int(np.sum(err_group))
        output_dim = input_dim
    else:
        input_dim = int(model_config.get("in_dim", np.sum(np.array(config.err_group))))
        output_dim = int(model_config.get("out_dim", np.sum(np.array(config.err_group))))
    return get_loader(
        which_data=config.which_data,
        which_type=which_type,
        in_dim=input_dim,
        out_dim=output_dim,
        N_samples=config.N_samples,
        batch_size=config.batch_size,
        masked_input=masked_input,
        is_time=config.is_time,
        max_num_time_steps=config.max_num_time_steps,
        time_step_size=config.time_step_size,
        fix_input_to_time_step=config.fix_input_to_time_step,
        allowed_transitions=config.allowed_transitions,
        ood_tag=None,
        shuffle_=False,
        macro_id=getattr(config, "macro_id", 2),
    )


def build_relevant_mask(config: argparse.Namespace) -> torch.Tensor:
    mask = []
    for size, is_kept in zip(config.err_group, config.err_mask_group):
        mask.extend([bool(is_kept)] * size)
    return torch.tensor(mask, dtype=torch.bool)


def infer_channel_names(config: argparse.Namespace, dataset, num_channels: int) -> list[str]:
    if getattr(config, "channel_names", None):
        names = list(config.channel_names)
    elif hasattr(dataset, "output_variable_names"):
        names = list(dataset.output_variable_names)
    elif hasattr(dataset, "variable_names"):
        names = list(dataset.variable_names)
    else:
        names = []

    if len(names) < num_channels:
        names = names + [f"channel_{idx}" for idx in range(len(names), num_channels)]
    return names[:num_channels]


def build_output_path(config: argparse.Namespace, model_folder: str) -> str:
    if getattr(config, "output_nc_path", None):
        return config.output_nc_path

    inference_tag = getattr(config, "inference_tag", "")
    suffix = f"_{inference_tag}" if inference_tag else ""
    output_dir = os.path.join(model_folder, f"predictions_{config.which_data}{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        f"{config.which_data}__{config.which_type}__members_{config.N_samples}__spectral_regression.nc",
    )


def unpack_batch(batch, is_time_regression: bool, masked_input_enabled: bool):
    if is_time_regression:
        if masked_input_enabled:
            t_batch, input_batch, output_batch, _ = batch
        else:
            t_batch, input_batch, output_batch = batch
        return t_batch, input_batch, output_batch
    input_batch, output_batch = batch
    return None, input_batch, output_batch


def infer_target_spatial_shape(config: argparse.Namespace, model_config: dict, dataset) -> Optional[tuple[int, int, int]]:
    native_shape = getattr(dataset, "native_spatial_shape", None)
    if native_shape is None:
        return None

    target_shape = getattr(dataset, "spatial_shape", native_shape)
    target_s = model_config.get("s_new", model_config.get("s"))
    if isinstance(target_s, (list, tuple)) and len(target_s) == 3:
        target_shape = tuple(int(v) for v in target_s)
    elif isinstance(target_s, int):
        target_shape = (int(target_s), int(target_s), native_shape[2])

    if getattr(config, "max_z", None) is not None:
        target_shape = (int(target_shape[0]), int(target_shape[1]), int(config.max_z))

    return (
        min(int(target_shape[0]), int(native_shape[0])),
        min(int(target_shape[1]), int(native_shape[1])),
        min(int(target_shape[2]), int(native_shape[2])),
    )


def configure_dataset_spatial_shape(dataset, target_spatial_shape: Optional[tuple[int, int, int]]) -> None:
    if target_spatial_shape is None or not hasattr(dataset, "native_spatial_shape"):
        return

    native_shape = tuple(int(v) for v in dataset.native_spatial_shape)
    clipped = (
        min(int(target_spatial_shape[0]), native_shape[0]),
        min(int(target_spatial_shape[1]), native_shape[1]),
        min(int(target_spatial_shape[2]), native_shape[2]),
    )
    dataset.spatial_shape = clipped
    if hasattr(dataset, "max_z"):
        dataset.max_z = clipped[2]


def resolve_member_range(config: argparse.Namespace) -> tuple[int, int]:
    total_members = int(config.N_samples)

    start = getattr(config, "member_start", None)
    end = getattr(config, "member_end", None)
    num_shards = getattr(config, "num_shards", None)
    shard_index = getattr(config, "shard_index", None)

    has_explicit_range = start is not None or end is not None
    has_shard_spec = num_shards is not None or shard_index is not None
    if has_explicit_range and has_shard_spec:
        raise ValueError("Use either explicit member_start/member_end or num_shards/shard_index, not both.")

    if has_shard_spec:
        if num_shards is None or shard_index is None:
            raise ValueError("Both --num_shards and --shard_index must be provided together.")
        if num_shards <= 0:
            raise ValueError("--num_shards must be positive.")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards.")
        start = (total_members * shard_index) // num_shards
        end = (total_members * (shard_index + 1)) // num_shards
    else:
        if start is None:
            start = 0
        if end is None:
            end = total_members

    if start < 0 or end < 0 or start > end or end > total_members:
        raise ValueError(
            f"Invalid member range [{start}, {end}) for N_samples={total_members}."
        )
    return int(start), int(end)


def retarget_dataset_members(dataset, member_start: int, member_end: int) -> None:
    local_member_count = member_end - member_start
    dataset.start += member_start
    dataset.num_trajectories = local_member_count
    if hasattr(dataset, "multiplier"):
        dataset.length = local_member_count * dataset.multiplier
    else:
        dataset.length = local_member_count


def next_write_position_in_range(ds, member_start: int, member_end: int, num_time_pairs: int) -> tuple[int, int]:
    if member_start >= member_end:
        return member_end, 0

    if "written_steps" not in ds.variables:
        if "written_mask" not in ds.variables:
            return member_start, 0
        written_mask = np.asarray(ds.variables["written_mask"][member_start:member_end], dtype=np.int64)
        unfinished = np.where(written_mask == 0)[0]
        if len(unfinished) == 0:
            return member_end, 0
        return member_start + int(unfinished[0]), 0

    written_steps = np.asarray(ds.variables["written_steps"][member_start:member_end], dtype=np.int64)
    for offset, steps_done in enumerate(written_steps):
        if int(steps_done) < num_time_pairs:
            return member_start + offset, int(steps_done)
    return member_end, 0


def ensure_output_file(
    output_path: str,
    total_members: int,
    time_indices: list[tuple[int, int]],
    spatial_shape: tuple[int, int, int],
    num_channels: int,
    variable_names: list[str],
    attrs: dict,
    overwrite: bool,
) -> None:
    lock_path = f"{output_path}.lock"
    if overwrite and os.path.exists(output_path):
        with open(lock_path, "a+") as lock_file:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        if os.path.exists(lock_path):
            os.remove(lock_path)

    with open(lock_path, "a+") as lock_file:
        import fcntl
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            ds = spectral_utils.create_nc_if_needed(
                nc_path=output_path,
                N_members=total_members,
                time_indices=time_indices,
                s=int(spatial_shape[0]),
                s_z=int(spatial_shape[2]),
                compression_level=3,
                C=int(num_channels),
                variable_names=variable_names,
                attrs=attrs,
            )
            ds.close()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def generate_predictions_for_folder(config: argparse.Namespace, model_root: str, folder_name: str, is_zero_shot: bool) -> None:
    model_folder = model_root if is_zero_shot else os.path.join(model_root, folder_name)
    print(f"\nProcessing model folder: {model_folder}\n")

    model, model_config, checkpoint_path = load_regression_model(model_folder, config.device)
    loader = build_loader(config, model_config=model_config)
    dataset = loader.dataset

    if not hasattr(dataset, "time_indices"):
        raise AttributeError("The selected dataset does not expose `time_indices`, which are required for spectral export.")

    target_spatial_shape = infer_target_spatial_shape(config, model_config, dataset)
    configure_dataset_spatial_shape(dataset, target_spatial_shape)

    member_start, member_end = resolve_member_range(config)
    retarget_dataset_members(dataset, member_start, member_end)
    local_member_count = member_end - member_start
    if local_member_count == 0:
        print("No members assigned to this worker; exiting early.")
        return
    
    relevant_mask = build_relevant_mask(config)
    model_output_channels = int(model_config.get("out_dim", relevant_mask.numel()))
    if relevant_mask.numel() != model_output_channels:
        raise ValueError(
            f"Relevant output mask has length {relevant_mask.numel()}, but the loaded model outputs {model_output_channels} channels. "
            "Please align `err_group`/`err_mask_group` with the regression model output configuration."
        )

    masked_input_enabled = bool(getattr(config, "is_masked", False))
    dtype = next(model.parameters()).dtype
    time_indices = list(dataset.time_indices)
    chunk_size = len(time_indices)
    output_path = build_output_path(config, model_folder)

    sample_batch = next(iter(loader))
    t_batch, input_batch, _ = unpack_batch(sample_batch, config.is_time, masked_input_enabled)
    input_batch = input_batch.to(device=config.device, dtype=dtype, non_blocking=True)
    if t_batch is not None:
        t_batch = t_batch.to(device=config.device, dtype=dtype, non_blocking=True)
    with torch.no_grad():
        sample_preds_np = model(input_batch, t_batch)[:, relevant_mask].detach().to(torch.float32).cpu().numpy()

    spatial_shape = tuple(int(v) for v in sample_preds_np.shape[-3:])
    variable_names = infer_channel_names(config, dataset, sample_preds_np.shape[1])
    attrs = {
        "source_dataset": config.which_data,
        "source_split": getattr(config, "which_type", "test"),
        "source_file": getattr(dataset, "file_path", None),
        "model_checkpoint": checkpoint_path,
        "model_folder": model_folder,
        "num_members_requested": int(config.N_samples),
        "num_time_pairs": int(chunk_size),
        "assigned_member_start": int(member_start),
        "assigned_member_end": int(member_end),
    }

    ensure_output_file(
        output_path=output_path,
        total_members=int(config.N_samples),
        time_indices=time_indices,
        spatial_shape=spatial_shape,
        num_channels=int(sample_preds_np.shape[1]),
        variable_names=variable_names,
        attrs=attrs,
        overwrite=bool(getattr(config, "overwrite", False)),
    )

    with spectral_utils.open_nc_locked(output_path, mode="a") as ds:
        resume_member, resume_time_offset = next_write_position_in_range(ds, member_start, member_end, chunk_size)

    if resume_member >= member_end:
        print(f"Assigned member range [{member_start}, {member_end}) is already complete in {output_path}.")
        return

    resume_steps = (resume_member - member_start) * chunk_size + resume_time_offset
    print(f"Writing predictions to {output_path}")
    print(f"Assigned member range: [{member_start}, {member_end})")
    print(f"Resuming from global member index {resume_member}, time offset {resume_time_offset}")

    current_member = member_start
    current_time_offset = 0
    global_step = 0

    def advance_position(member_idx: int, time_idx: int) -> tuple[int, int]:
        time_idx += 1
        if time_idx >= chunk_size:
            member_idx += 1
            time_idx = 0
        return member_idx, time_idx

    def write_prediction(pred_single: np.ndarray, member_idx: int, time_idx: int) -> None:
        with spectral_utils.open_nc_locked(output_path, mode="a") as ds:
            spectral_utils.write_time_slice(
                ds,
                member_idx,
                time_idx,
                pred_single,
                variable_names=variable_names,
                sync=True,
            )
    
    with torch.no_grad():
        progress = tqdm.tqdm(total=local_member_count * chunk_size, desc="Generating predictions")
        if resume_steps:
            progress.update(resume_steps)
            for _ in range(resume_steps):
                current_member, current_time_offset = advance_position(current_member, current_time_offset)
                global_step += 1

        for batch in loader:
            t_batch, input_batch, _ = unpack_batch(batch, config.is_time, masked_input_enabled)
            input_batch = input_batch.to(device=config.device, dtype=dtype, non_blocking=True)
            if t_batch is not None:
                t_batch = t_batch.to(device=config.device, dtype=dtype, non_blocking=True)
            preds_np = model(input_batch, t_batch)[:, relevant_mask].detach().to(torch.float32).cpu().numpy()

            for pred_single in preds_np:
                if global_step < resume_steps:
                    global_step += 1
                    continue

                if current_member >= member_end:
                    break
                
                write_prediction(pred_single, current_member, current_time_offset)
                current_member, current_time_offset = advance_position(current_member, current_time_offset)
                global_step += 1
                progress.update(1)

            if current_member >= member_end:
                break

        progress.close()


def main() -> None:
    torch.cuda.empty_cache()
    args = parse_args()
    config = load_config(args)

    model_root = get_model_root(config)
    folders, is_zero_shot = get_candidate_model_folders(
        model_root=model_root,
        tags=getattr(config, "tags", []),
        exclude=getattr(config, "exclude", []),
    )

    print("\nCandidate model folders:")
    for folder in folders:
        print(f"- {folder}")

    for folder_name in folders:
        generate_predictions_for_folder(config, model_root, folder_name, is_zero_shot)


if __name__ == "__main__":
    main()
