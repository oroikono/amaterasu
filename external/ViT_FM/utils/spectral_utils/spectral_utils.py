import fcntl
import os
from contextlib import contextmanager
from typing import Optional

import numpy as np
from netCDF4 import Dataset


def _lock_path(nc_path: str) -> str:
    return f"{nc_path}.lock"


@contextmanager
def open_nc_locked(nc_path: str, mode: str = "a"):
    lock_path = _lock_path(nc_path)
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        ds = Dataset(nc_path, mode)
        try:
            yield ds
        finally:
            ds.close()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _default_variable_names(num_channels: int) -> list[str]:
    mapping = {
        2: ["u", "v"],
        3: ["u", "v", "w"],
        4: ["rho", "u", "v", "p"],
        5: ["rho", "u", "v", "w", "p"],
    }
    if num_channels not in mapping:
        raise ValueError(
            f"Unsupported C={num_channels}. Pass `variable_names` explicitly for custom channel layouts."
        )
    return mapping[num_channels]


def _resolve_variable_names(ds: Optional[Dataset], num_channels: int, variable_names: Optional[list[str]]) -> list[str]:
    if variable_names is not None:
        channel_names = list(variable_names)
    elif ds is not None and "channel_names" in ds.ncattrs():
        channel_names = [name for name in ds.getncattr("channel_names").split(",") if name]
    else:
        channel_names = _default_variable_names(num_channels)

    if len(channel_names) != num_channels:
        raise ValueError(f"Expected {num_channels} variable names, got {len(channel_names)}: {channel_names}")
    return channel_names


def create_nc_if_needed(
    nc_path: str,
    N_members: int,
    time_indices: list[tuple[int, int]],
    s: int = 128,
    s_z: Optional[int] = None,        # if None -> 2D; if int -> 3D
    compression_level: int = 3,
    C: int = 4,
    variable_names: Optional[list[str]] = None,
    attrs: Optional[dict] = None,
):
    """
    Create the NetCDF file with fixed-size dims and chunked, compressed variables.
    Safe to call repeatedly; if the file exists, it is opened in append mode.

    Spatial layout:
      - 2D: (x, y) with size (s, s)
      - 3D: (x, y, z) with size (s, s, s_z)
    Channels default to the legacy PDE layouts unless `variable_names` is provided.
    """
    channel_names = _resolve_variable_names(None, C, variable_names)

    is_3d = s_z is not None
    B = len(time_indices)
    file_exists = os.path.exists(nc_path)

    if not file_exists:
        os.makedirs(os.path.dirname(nc_path), exist_ok=True)
        ds = Dataset(nc_path, "w", format="NETCDF4")

        # Dimensions
        ds.createDimension("member", N_members)
        ds.createDimension("time", B)
        ds.createDimension("x", s)
        ds.createDimension("y", s)
        if is_3d:
            ds.createDimension("z", s_z)

        # Coordinates
        member_var = ds.createVariable("member", "i4", ("member",))
        member_var[:] = np.arange(N_members, dtype=np.int32)

        time_i = ds.createVariable("time_i", "i4", ("time",))
        time_j = ds.createVariable("time_j", "i4", ("time",))
        ti = np.array([ij[0] for ij in time_indices], dtype=np.int32)
        tj = np.array([ij[1] for ij in time_indices], dtype=np.int32)
        time_i[:] = ti
        time_j[:] = tj

        # Select spatial dims & chunking
        if is_3d:
            data_dims = ("member", "time", "x", "y", "z")
            chunks = (1, 1, s, s, s_z)
        else:
            data_dims = ("member", "time", "x", "y")
            chunks = (1, 1, s, s)

        kw = dict(
            zlib=True,
            complevel=compression_level,
            chunksizes=chunks,
            shuffle=True,
        )

        for var_name in channel_names:
            ds.createVariable(var_name, "f4", data_dims, **kw)

        # Meta
        ds.setncattr("description", "PDEGym+ predictions")
        ds.setncattr("channel_names", ",".join(channel_names))
        ds.setncattr("note", "Spatial dims: 2D=(x,y) or 3D=(x,y,z). Time indices stored in time_i/time_j.")
        ds.setncattr("created_by", "streaming-writer")
        ds.setncattr("layout", f"variables: {channel_names} with dims {data_dims}")

        # Optional progress/helper variable to skip already-written members on resume
        written = ds.createVariable("written_mask", "i1", ("member",))
        written[:] = 0
        written_steps = ds.createVariable("written_steps", "i4", ("member",))
        written_steps[:] = 0

        if attrs is not None:
            for key, value in attrs.items():
                if value is not None:
                    ds.setncattr(key, value)

        ds.sync()
        ds.close()

    ds = Dataset(nc_path, "a")

    if len(ds.dimensions["member"]) != N_members:
        raise ValueError(f"NetCDF member dim mismatch: expected {N_members}, found {len(ds.dimensions['member'])}")
    if len(ds.dimensions["time"]) != B:
        raise ValueError(f"NetCDF time dim mismatch: expected {B}, found {len(ds.dimensions['time'])}")
    if len(ds.dimensions["x"]) != s or len(ds.dimensions["y"]) != s:
        raise ValueError(
            f"NetCDF x/y dims mismatch: expected ({s}, {s}), found ({len(ds.dimensions['x'])}, {len(ds.dimensions['y'])})"
        )
    has_z = "z" in ds.dimensions
    if is_3d != has_z:
        raise ValueError(
            f"NetCDF dimensionality mismatch: expected {'3D' if is_3d else '2D'}, found {'3D' if has_z else '2D'}."
        )
    if is_3d and len(ds.dimensions["z"]) != s_z:
        raise ValueError(f"NetCDF z dim mismatch: expected {s_z}, found {len(ds.dimensions['z'])}")
    
    if "written_steps" not in ds.variables:
        written_steps = ds.createVariable("written_steps", "i4", ("member",))
        written_steps[:] = 0
    if attrs is not None:
        for key, value in attrs.items():
            if value is not None:
                ds.setncattr(key, value)
    return ds

def next_write_position(ds: Dataset, num_time_pairs: int) -> tuple[int, int]:
    """
    Return the next (member_idx, time_idx) position for streaming writes.

    Files created before `written_steps` existed fall back to member-level resume.
    """
    num_members = len(ds.dimensions["member"])

    if "written_steps" not in ds.variables:
        if "written_mask" not in ds.variables:
            return 0, 0

        mask = np.asarray(ds.variables["written_mask"][:])
        hits = np.where(mask == 0)[0]
        if len(hits) == 0:
            return num_members, 0
        return int(hits[0]), 0

    written_steps = np.asarray(ds.variables["written_steps"][:], dtype=np.int64)
    for member_idx in range(num_members):
        steps_done = int(written_steps[member_idx])
        if steps_done < num_time_pairs:
            return member_idx, steps_done
    return num_members, 0


def write_time_slice(
    ds: Dataset,
    member_idx: int,
    time_idx: int,
    pred_Cxyz: np.ndarray,
    variable_names: Optional[list[str]] = None,
    sync: bool = False,
):
    """
    Write a single time slice for one member.

    For 2D:
      pred_Cxyz: (C, s, s)
    For 3D:
      pred_Cxyz: (C, s, s, s_z)
    """
    assert pred_Cxyz.ndim in (3, 4), f"Expected (C,s,s) or (C,s,s,s_z), got {pred_Cxyz.shape}"

    is_3d = pred_Cxyz.ndim == 4
    if is_3d:
        C, sx, sy, sz = pred_Cxyz.shape
    else:
        C, sx, sy = pred_Cxyz.shape

    channel_names = _resolve_variable_names(ds, C, variable_names)

    if pred_Cxyz.dtype != np.float32:
        pred_Cxyz = pred_Cxyz.astype(np.float32, copy=False)

    assert "x" in ds.dimensions and "y" in ds.dimensions, "NetCDF missing x/y dimensions"
    assert sx == len(ds.dimensions["x"]), f"x dim mismatch: data={sx}, nc={len(ds.dimensions['x'])}"
    assert sy == len(ds.dimensions["y"]), f"y dim mismatch: data={sy}, nc={len(ds.dimensions['y'])}"

    has_z = "z" in ds.dimensions
    assert has_z == is_3d, (
        f"Dimensionality mismatch: data is {'3D' if is_3d else '2D'}, "
        f"but NetCDF is {'3D' if has_z else '2D'}."
    )
    if is_3d:
        assert sz == len(ds.dimensions["z"]), f"z dim mismatch: data={sz}, nc={len(ds.dimensions['z'])}"

    for channel_idx, var_name in enumerate(channel_names):
        if is_3d:
            ds.variables[var_name][member_idx, time_idx, :, :, :] = pred_Cxyz[channel_idx]
        else:
            ds.variables[var_name][member_idx, time_idx, :, :] = pred_Cxyz[channel_idx]

    if "written_steps" in ds.variables:
        next_step = max(int(ds.variables["written_steps"][member_idx]), time_idx + 1)
        ds.variables["written_steps"][member_idx] = next_step
        if next_step >= len(ds.dimensions["time"]) and "written_mask" in ds.variables:
            ds.variables["written_mask"][member_idx] = 1

    if sync:
        ds.sync()


def write_member(
    ds: Dataset,
    member_idx: int,
    pred_BCxyz: np.ndarray,
    variable_names: Optional[list[str]] = None,
):
    """
    Write a single member's predictions.

    For 2D:
      pred_BCxyz: (B, C, s, s)
    For 3D:
      pred_BCxyz: (B, C, s, s, s_z)

    Channel ordering follows `variable_names` when provided, otherwise the
    legacy PDE layouts are used.
    """
    assert pred_BCxyz.ndim in (4, 5), f"Expected (B,C,s,s) or (B,C,s,s,s_z), got {pred_BCxyz.shape}"

    is_3d = pred_BCxyz.ndim == 5

    if is_3d:
        B, C, sx, sy, sz = pred_BCxyz.shape
    else:
        B, C, sx, sy = pred_BCxyz.shape

    channel_names = _resolve_variable_names(ds, C, variable_names)
    
    if len(channel_names) != C:
        raise ValueError(f"Expected {C} variable names, got {len(channel_names)}: {channel_names}")
    
     # Check that ds spatial dims match the data
    assert "x" in ds.dimensions and "y" in ds.dimensions, "NetCDF missing x/y dimensions"
    
    assert sx == len(ds.dimensions["x"]), f"x dim mismatch: data={sx}, nc={len(ds.dimensions['x'])}"
    assert sy == len(ds.dimensions["y"]), f"y dim mismatch: data={sy}, nc={len(ds.dimensions['y'])}"

    has_z = "z" in ds.dimensions
    assert has_z == is_3d, (
        f"Dimensionality mismatch: data is {'3D' if is_3d else '2D'}, "
        f"but NetCDF is {'3D' if has_z else '2D'}."
    )
    if is_3d:
        assert sz == len(ds.dimensions["z"]), f"z dim mismatch: data={sz}, nc={len(ds.dimensions['z'])}"

    dataset_variables = [ds.variables[name] for name in channel_names]

    # Ensure float32
    if pred_BCxyz.dtype != np.float32:
        pred_BCxyz = pred_BCxyz.astype(np.float32, copy=False)
    
    # Helper index function for slicing
    def assign_slice(var, arr):
        if is_3d:
            var[member_idx, t, :, :, :] = arr
        else:
            var[member_idx, t, :, :] = arr

    # Write per-time slice to keep memory footprint low and support partial retries
    for t in range(B):
        for channel_idx, var in enumerate(dataset_variables):
            assign_slice(var, pred_BCxyz[t, channel_idx])

    if "written_mask" in ds.variables:
        ds.variables["written_mask"][member_idx] = 1
    ds.sync()
