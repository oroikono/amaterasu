import os
import sys

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go 2 directories up
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add that directory to sys.path (if not already there)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import torch
import tqdm
import argparse
import functools
import numpy as np
import time
import copy
from regression.EmbeddingModule import FourierEmbedding, AdaptiveScale


from utils.utils_data import get_loader, load_data, read_cli_inference, find_files_with_extension, save_errors
from utils.utils_inference import append_unique_dicts_to_csv, extract_meaning_variables
import matplotlib.pyplot as plt

from utils.utils_finetune import initialize_FT
from regression.ViTModulev2 import MultiVit3_pl, MultiVit2_pl, Vit3_pl
from utils.spectral_utils.spectral_utils import create_nc_if_needed, write_member, next_unwritten_member

import sys
sys.path.append("/path/to/site-packages")  # Replace with the correct path
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="load parameters for training")
    params = read_cli_inference(parser).parse_args()
    
    if params.config is None:
        config = params
    else:
        config = argparse.Namespace(**load_data(params.config))
    
    folder_ = config.config_regression_folder
    error_file = config.error_file
    tags = config.tags

    save_data = config.save_data
    inference_tag = config.inference_tag
    device = config.device

    err_group = config.err_group
    err_mask_group = config.err_mask_group
    groups = [0]
    variable_meaning = extract_meaning_variables(np.sum(np.array(err_group)), groups = err_group, which_data = config.which_data)
    
    const_dim = []
    for i,e in enumerate(err_mask_group):
        const_dim = const_dim + err_group[i] * [1.0 * err_mask_group[i]]
    print(" ")
    print("CONST DIM ARRAY: ", const_dim)
    print(" ")

    is_time_reg = config.is_time
    masked_input = config.is_masked
    if not masked_input:
        masked_input = None

    regression_scheme = config.regression_scheme
    dt = config.dt
    if "merra" in config.which_data:
        dt = 1/24.
        print("HERE")

    p = 1

    print(dt, "DT")

    num_unmasked = 0
    for i,g in enumerate(err_group):
        groups.append(groups[-1]+g)
        if err_mask_group[i]>0:
            num_unmasked+=1
    num_groups = len(err_group)

    if ("finetuned" in folder_) or ("scratch" in folder_ or "Scratch" in folder_):
        subfolders = [f.name for f in os.scandir(folder_) if f.is_dir()]
        folders = []
        for f in subfolders:
            is_valid = True
            for tag in tags:
                if tag not in f:
                    is_valid = False
            if is_valid:
                folders.append(f)
        is_zero_shot = False
    else:
        folders = [folder_]
        is_zero_shot = True

    print(" ")
    print("Folders: ")
    print(folders)
    print(" ")

    if hasattr(config, "is_persistence"):
        is_persistence = config.is_persistence
    else:
        is_persistence = False

    if hasattr(config, "in_one_array"):
        in_one_array = config.in_one_array
    else:
        in_one_array = False
    if in_one_array:
        inp_array = []
        out_array = []
        pred_array = []

    for folder in folders:
        
        if not is_zero_shot:
            path =  f"{folder_}/{folder}"
        else:
            path = folder
        
        regression_model_path = str(find_files_with_extension(path + "/model", "ckpt", [], is_pl = True)[0])
        regression_config_path = str(find_files_with_extension(path, "json", ["param"])[0])
        config_reg  = argparse.Namespace(**load_data(regression_config_path))

        print("MODEL PATH: " + regression_model_path)
        print(" ")


        arch = config_reg.config_arch
        if "finetuned" not in regression_model_path:
            config_reg_arch = load_data(arch)
        else:
            config_reg_arch = dict(arch)

        config_reg = vars(config_reg)
        config_reg["workdir"] = None

        
        regression_model = Vit3_pl(in_dim = config_reg["in_dim"], 
                                    out_dim = config_reg["out_dim"],
                                    loss_fn = None,
                                    config_train = config_reg,
                                    config_arch = config_reg_arch)
    

        if "init_new" in config_reg:
            init_new = config_reg["init_new"]
        else:
            init_new = False
        
        if "reinit_film" in config_reg:
            reinit_film = config_reg["reinit_film"]
        else:
            reinit_film = False
            
        if "finetuned" in regression_model_path and 'reinit_ft' in config_reg and config_reg['reinit_ft']:
            regression_model = initialize_FT(regression_model, config_reg["in_dim"],config_reg["out_dim"], latent_channels = config_reg_arch["latent_channels"], init_new = init_new, init_film = reinit_film)
            
        checkpoint = torch.load(regression_model_path, map_location = device)
        regression_model.load_state_dict(checkpoint["state_dict"])
        regression_model = regression_model.model.to(device).eval()

        data_folder = f"{path}/errors_reg/errors_{config.which_data}_{config.N_samples}s"
        if not os.path.exists(f"{data_folder}"):
            os.makedirs(f"{data_folder}")

        if save_data:
            counter = 0
            if not os.path.exists(f"{path}/predictions_{config.which_data}_{inference_tag}"):
                os.makedirs(f"{path}/predictions_{config.which_data}_{inference_tag}")
            folder_pred = f"{path}/predictions_{config.which_data}_{inference_tag}"
    
        cnt_pred = 0
        errors_lp = np.zeros((0, num_groups))
        errors_lp_rel = np.zeros((0, num_groups))

        print("GROUPS, ", groups, "UNMASKED, ", num_unmasked, "MASKED GROUPS, ", err_mask_group)        
        
        if hasattr(config, "which_type") and config.which_type in ["train", "test", "val"]:
            which_type = config.which_type
        else:
            which_type = "test"
        
        test_loader = get_loader(which_data = config.which_data,
                            which_type = which_type,
                            in_dim = np.sum(np.array(err_group)),
                            out_dim = np.sum(np.array(err_group)),
                            N_samples = config.N_samples,
                            batch_size = config.batch_size,
                            masked_input = masked_input,
                            is_time = is_time_reg,
                            max_num_time_steps = config.max_num_time_steps,
                            time_step_size = config.time_step_size,
                            fix_input_to_time_step = config.fix_input_to_time_step,
                            allowed_transitions = config.allowed_transitions,
                            ood_tag = None,
                            shuffle_ = False,
                            set_batch_to_full_traj = True)

        tqdm_data = tqdm.tqdm(test_loader)
        print(len(tqdm_data), "DATA LEN")
        print(" ")

        relevant_mask = []
        for size, m in zip(err_group, err_mask_group):
            relevant_mask.extend([m] * size)
        relevant_mask = torch.tensor(relevant_mask, dtype=torch.bool)
        

        if hasattr(test_loader.dataset, 'time_indices'):
            time_indices = test_loader.dataset.time_indices
        else:
            raise Exception("No time_indices attribute in dataset")
        
        nc_path = os.path.join(folder_pred, f"{config.which_data}_pdegym_plus_pred_{config.N_samples}.nc")
        ds = create_nc_if_needed(nc_path, N_members=len(test_loader), time_indices=time_indices, s=128, compression_level=3, C = np.sum(np.array(config.err_mask_group)*np.array(config.err_group)))
        print("GENERATING THE DATASET OF ",len(test_loader)," SAMPLES,  PATH IS", nc_path)
        with torch.no_grad():
            for step, batch in enumerate(tqdm_data):
                
                '''
                    Unpack:
                '''
                
                if is_time_reg:
                    if masked_input is not None:
                        t_batch, input_batch, output_batch, mask = batch
                    else:
                        t_batch, input_batch, output_batch = batch
                        mask = None
                else:
                    input_batch, output_batch = batch
                    t_batch = None
                                
                '''
                    Send to device & make a copy for later use
                '''

                t_batch = t_batch
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)[:, relevant_mask]
                t_batch = t_batch.type(torch.float32).to(device)
                #print(t_batch)
                output_pred_batch = regression_model(input_batch, t_batch)[:, relevant_mask]

                B,C,s,s = output_pred_batch.shape

                if isinstance(output_pred_batch, torch.Tensor):
                    output_pred_batch = output_pred_batch.detach().to(torch.float32).cpu().numpy()
                
                #print(relevant_mask, output_pred_batch.shape)
                write_member(ds, step, output_pred_batch)

                #np.save(f"{folder_pred}/sample_{step}_pred.npy", output_pred_batch[0])
                #np.save(f"{folder_pred}/sample_{step}_out.npy", output_batch.detach().cpu().numpy()[0])
                #np.save(f"{folder_pred}/sample_{step}_inp.npy",input_batch.detach().cpu().numpy()[0])


                '''
                    Make predictions (regression) - AR prediction based on regression_scheme and dt
                '''

                """if not is_persistence:
                    if is_time_reg:
                        for i, reg_step in enumerate(regression_scheme):
                            t = dt * reg_step
                            t_batch = t * torch.ones_like(t_batch).to(device)
                            t_batch = t_batch.type(torch.float32)
                            output_pred_batch = regression_model(input_batch, t_batch)         
                            
                            input_batch[:,:config_reg["out_dim"]] = output_pred_batch

                            if mask is not None:
                                for m in range(len(const_dim)):
                                    if const_dim[m] == 0:
                                        input_batch[:,m] = input_batch_copy[:,m]
                else:
                    output_pred_batch = input_batch_copy
                """
                