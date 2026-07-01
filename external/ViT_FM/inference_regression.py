import torch
import tqdm
import argparse
import functools
import numpy as np
import os
import time
import copy
from regression.EmbeddingModule import FourierEmbedding, AdaptiveScale


from utils.utils_data import get_loader, load_data, read_cli_inference, find_files_with_extension, save_errors
from utils.utils_inference import append_unique_dicts_to_csv, extract_meaning_variables
import matplotlib.pyplot as plt

from utils.utils_finetune import initialize_FT
from regression.ViTModulev2 import MultiVit3_pl, MultiVit2_pl, Vit3_pl

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
                            rel_time = False,
                            ood_tag = None,
                            shuffle_ = False)
        
        mean = test_loader.dataset.mean.unsqueeze(0).to(device)
        std = test_loader.dataset.std.unsqueeze(0).to(device)

        tqdm_data = tqdm.tqdm(test_loader)
        print(len(tqdm_data), "DATA LEN")
        print(" ")

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
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                input_batch_copy = copy.deepcopy(input_batch).to(device)
                output_copy = copy.deepcopy(output_batch).detach().cpu().numpy()

                '''
                    Make predictions (regression) - AR prediction based on regression_scheme and dt
                '''

                if not is_persistence:
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
                
                '''
                    Save the predictions if the flag is up
                '''
                if save_data:
                    pred_copy = copy.deepcopy(output_pred_batch).detach().cpu().numpy()
                    if not in_one_array:
                        for i in range(pred_copy.shape[0]):
                            time_ = str(np.sum(regression_scheme)*0.1)
                            np.save(f"{folder_pred}/sample_{cnt_pred}_steps_{len(regression_scheme)}_{time_}_pred.npy", pred_copy[i])
                            np.save(f"{folder_pred}/sample_{cnt_pred}_steps_{len(regression_scheme)}_{time_}_out.npy", output_copy[i])
                            np.save(f"{folder_pred}/sample_{cnt_pred}_steps_{len(regression_scheme)}_{time_}_inp.npy",input_batch_copy.detach().cpu().numpy()[i])
                            cnt_pred+=1
                        del pred_copy
                    else:
                        inp_array.append(input_batch_copy.detach().cpu().numpy())
                        out_array.append(output_copy)
                        pred_array.append(pred_copy)
                    

                '''
                    Concatenate the errors
                '''

                errs_rel = np.zeros((input_batch.shape[0],num_groups))
                errs = np.zeros((input_batch.shape[0], num_groups))
                for d in range(num_groups):
                    dim_in = groups[d]
                    dim_out = groups[d+1]

                    loss_rel_lp = torch.mean(abs(output_pred_batch[:,dim_in:dim_out] - output_batch[:,dim_in:dim_out]), dim = [1,2,3])/ torch.mean(abs(output_batch[:,dim_in:dim_out]) + 1e-10, dim = [1,2,3])
                    loss_lp = (torch.mean(abs(output_pred_batch[:,dim_in:dim_out] - output_batch[:,dim_in:dim_out]) ** p, (-3, -2, -1))) ** (1 / p)
                    
                    if err_mask_group[d]>0 and step == 0:
                        print(dim_in, dim_out, loss_rel_lp)
                    loss_rel_lp = loss_rel_lp.reshape(-1).detach().cpu().numpy()
                    loss_lp = loss_lp.reshape(-1).detach().cpu().numpy()
                    
                    errs_rel[:,d] = loss_rel_lp 
                    errs[:,d] =loss_lp

                errors_lp = np.concatenate((errors_lp, errs), axis = 0)
                errors_lp_rel = np.concatenate((errors_lp_rel, errs_rel), axis = 0)

        median_errs = np.median(errors_lp_rel, axis = 0)
        median_errs_l1 = np.median(errors_lp, axis = 0)

        if save_data and in_one_array:
            time_ = round(np.sum(np.array(regression_scheme))*dt,1)
            inp_array = np.concatenate(inp_array, axis = 0)
            out_array = np.concatenate(out_array, axis = 0)
            pred_array = np.concatenate(pred_array, axis = 0)
            np.save(f"{folder_pred}/samples_steps_{len(regression_scheme)}_{time_}_{inp_array.shape[0]}_pred.npy", pred_array)
            np.save(f"{folder_pred}/samples_steps_{len(regression_scheme)}_{time_}_{inp_array.shape[0]}_out.npy", out_array)
            np.save(f"{folder_pred}/samples_steps_{len(regression_scheme)}_{time_}_{inp_array.shape[0]}_inp.npy", inp_array)

        d_load = dict()
        check_columns = 20

        if  "mlp_dim" in config_reg["config_arch"] and config_reg["config_arch"]["mlp_dim"] == 2048 and config_reg["config_arch"]["heads"] == 16:
            d_load["model"] = f"{tags[0]}ViT_B"
        else:
            d_load["model"] = "ViT_S"
        
        if is_persistence:
            d_load["model"] = f"persistence"

        d_load["experiment"] = config.which_data

        time_ = np.sum(np.array(regression_scheme))*dt
        d_load["time"] = round(time_,2)
        if not is_zero_shot:
            d_load["num_trajectories"] = int(folder.strip().split("_")[-1])
            d_load["ar_steps"] = len(regression_scheme)
        else:
            if "scratch" in error_file or "scratch" in folder:
                d_load["num_trajectories"] = config_reg["N_train"]
                d_load["ar_steps"] = len(regression_scheme)
                d_load["fourier_emb"] = True 
                if "is_fourier_emb" in config_reg:
                    d_load["fourier_emb"] = config_reg["is_fourier_emb"]
                check_columns = 6
            else:
                d_load["num_trajectories"] = 0.5
        
        d_load["native"] = "non_native" not in folder

        err_final = 0.
        for d in range(num_groups):
            err_final+= (err_mask_group[d] * median_errs[d])

            if err_mask_group[d]>0:
                d_load[variable_meaning[d]+"_l1_rel"] =  median_errs[d]
                d_load[variable_meaning[d]+"_l1"] =  median_errs_l1[d]
        err_final = err_final / num_unmasked

        d_load["err_final_l1_rel"] = err_final
        append_unique_dicts_to_csv([d_load], fn = error_file, check_columns = check_columns)

        print(median_errs, err_final, "MEAN OVER MEDIAN")
        print(" ")
