import re
import glob
import numpy as np
import dask.array as da
import xarray as xr

# --- 1. Get sorted prediction files ---
base_dir = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/pdegym_plus/PDEGYM_PLUS_10ep_ViTB_regression/predictions_ns_shear_gencfd_generated_data"
N = 12000
files = sorted(glob.glob(f"{base_dir}/samples_steps_1_*_{N}_pred.npy"))
dim = 2

print(files)
# --- 2. Extract time values ---
def extract_time(fname):
    m = re.search(r"samples_steps_1_([0-9.]+)_12000_pred.npy", fname)
    return float(m.group(1)) if m else None

times = [extract_time(f) for f in files]
print(times)

# --- 3. Wrap each file as Dask array ---
arrays = [da.from_array(np.load(f, mmap_mode='r'), chunks=(500, 9, 128, 128)) for f in files]

# --- 4. Stack along new 'time' dimension ---
# shape → (time, member, channel, x, y)
stacked = da.stack(arrays, axis=0)

# --- 5. Select only first 4 channels and rearrange ---
# Convert to (member, time, x, y)
u   = stacked[:, :, 1, :, :].transpose(1, 0, 2, 3)
v   = stacked[:, :, 2, :, :].transpose(1, 0, 2, 3)

print(u.shape, v.shape)

# --- 6. Build Dataset ---
if dim == 4:
    rho = stacked[:, :, 0, :, :].transpose(1, 0, 2, 3)
    p   = stacked[:, :, 3, :, :].transpose(1, 0, 2, 3)

    ds = xr.Dataset(
        {
            "rho": (("member", "time", "x", "y"), rho),
            "u":   (("member", "time", "x", "y"), u),
            "v":   (("member", "time", "x", "y"), v),
            "p":   (("member", "time", "x", "y"), p),
        },
        coords={
            "member": np.arange(rho.shape[0]),
            "time": np.array(times, dtype=np.float32),
            "x": np.arange(128),
            "y": np.arange(128),
        },
    )
elif dim == 2:
    ds = xr.Dataset(
        {
            "u":   (("member", "time", "x", "y"), u),
            "v":   (("member", "time", "x", "y"), v),
        },
        coords={
            "member": np.arange(u.shape[0]),
            "time": np.array(times, dtype=np.float32),
            "x": np.arange(128),
            "y": np.arange(128),
        },
    )


# --- 7. Write to NetCDF efficiently ---
ds.to_netcdf(
    f"{base_dir}/{riemann_curved}_pdegym_plus_output.nc",
    compute=True,  # set to False to build lazily and compute later
    engine="netcdf4",
    encoding={var: {"zlib": True, "complevel": 4, "chunksizes": (1, 1, 128, 128)} for var in ds.data_vars},
)

