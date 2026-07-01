import os
import pandas as pd

def append_unique_dicts_to_csv(dicts, fn="output.csv", check_columns = 3):
    df_new = pd.DataFrame(dicts)
    if os.path.exists(fn):
        df_old = pd.read_csv(fn)
        # ensure same columns & order
        all_cols = list(dict.fromkeys(df_old.columns.tolist() + df_new.columns.tolist()))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        
        # identify the first four columns
        firstn = all_cols[:check_columns]
        # build a set of existing n-tuples
        seen = set(df_old[firstn].itertuples(index=False, name=None))
        # filter out duplicates
        mask = df_new[firstn].apply(tuple, axis=1).isin(seen)
        df_to_add = df_new.loc[~mask]
        
        # concat and write back
        df = pd.concat([df_old, df_to_add], ignore_index=True, sort=False)
    else:
        df = df_new

    df.to_csv(fn, index=False)

def filter_rows_by_keys(csv_file, d_keys):
    """
    Read `csv_file` and return only rows where each key in `d_keys`
    has the corresponding value.

    Parameters
    ----------
    csv_file : str
        Path to your CSV.
    d_keys : dict
        Column→value filters.  E.g. {"model":"ViT-B", "l1_u":0.123}

    Returns
    -------
    pd.DataFrame
        Subset of rows matching all filters.
    """
    # 1) load
    df = pd.read_csv(csv_file)
    
    # 2) check that all keys exist
    missing = [k for k in d_keys if k not in df.columns]
    if missing:
        raise KeyError(f"These columns are not in {csv_file}: {missing}")
    
    # 3) build mask
    mask = pd.Series(True, index=df.index)
    for key, val in d_keys.items():
        mask &= (df[key] == val)
    
    # 4) return filtered DataFrame
    frame = df[mask]
    Ns = []
    Errs = []
    if not frame.empty:
        rows = frame.to_dict(orient="records")
        for d in rows:
            Ns.append(d["num_trajectories"])
            Errs.append(d["err_final_l1_rel"])
    return Ns, Errs


'''
    Extract the meanings from the variables of interest:
'''

def extract_meaning_variables(out_dim, groups = [1,2,1], which_data = "eul_ns_mix"):

    meaning = None
    if out_dim == 9:
        if which_data == "mhd_orszag8" or which_data == "mhd_orszag8_long":
            meaning = ["rho", "uv", "p", "b", "u_h", "coeff", "bc"]
        else:
            meaning = ["rho", "uv", "p", "u_w", "u_ac", "u_h", "coeff", "bc"]
    elif out_dim == 6:
        if which_data == "mhd_orszag8" or which_data=="mhd_orszag8_long":
            meaning = ["rho", "uv", "p", "b"]
    elif out_dim == 5:
        if which_data == "eul_riemann_kh3d":
            meaning = ["rho", "uvz", "p"]
        if which_data == "eul_riemann3d":
            meaning = ["rho", "uvz", "p"]
        else:
            meaning = ["rho", "uvz", "p"]
    elif out_dim == 4:
        if "ns" in which_data and "tracer" not in which_data:
            meaning = ["rho", "uv", "p"]
        elif "eul" in which_data and "tracer" not in which_data:
            meaning = ["rho", "uv", "p"]
        elif "wave" in which_data:
            meaning = ["u", "x", "y", "coeff"]
        elif "poisson" in which_data:
            meaning = ["x", "u", "coeff", "y"]
        elif "allen" in which_data:
            meaning = ["u", "x", "y", "z"]
    elif out_dim == 2:
        if "wave" in which_data:
            meaning = ["u", "coeff"]
        elif "ns_pwc" in which_data:
            meaning = ["uv"]
    elif out_dim == 1:
        if "poisson" in which_data:
            meaning = ["u"]
        elif "allen_cahn" in which_data:
            meaning = ["u"]

    if meaning is not None:
        assert len(meaning) == len(groups)
    return meaning

